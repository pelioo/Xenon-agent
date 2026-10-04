# -*- coding: utf-8 -*-
"""用户配置写回（保留注释的文本级合并）。

webui 设置面板把表单保存到 xenon.yml 时使用本模块：
- 只替换/追加目标段下的键值行，不重排、不丢注释、不动其他段；
- 失败前先备份为 xenon.yml.bak，可人工恢复；
- 返回值给出每个键是否发生变更，便于调用方提示"需重启生效"。

注意：llm 客户端 / 上下文上限等是启动快照（进程级配置），
写回后需要重启 Xenon / webui 进程才会生效。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from xenon_core.settings import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)

# 顶层段名：形如 "llm:"、"context:" 的非注释行（允许段名含 -）
_SECTION_RE = re.compile(r"^(?P<name>[A-Za-z0-9_-]+):\s*(?P<rest>.*)$")
# 段内已激活键：形如 "  key: value"（2 空格缩进，非注释）
_KEY_RE = re.compile(r"^  (?P<key>[A-Za-z0-9_.-]+):(?P<rest>.*)$")
# 常见布尔/数值/列表值直接写；字符串需要引号时用 JSON 风格
_NEEDS_QUOTE_RE = re.compile(r"[:#\[\]{},&*!|>'\"%@`]|^\s|\s$|^[-\d.]")


def _format_yaml_value(value: Any) -> str:
    """把 Python 值序列化为单行 YAML 值（flow 风格，安全）。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        items = ", ".join(_format_yaml_value(item) for item in value)
        return f"[{items}]"
    if isinstance(value, dict):
        items = ", ".join(
            f"{_format_yaml_value(str(k))}: {_format_yaml_value(v)}"
            for k, v in value.items()
        )
        return f"{{{items}}}"
    text = str(value)
    if text == "":
        return '""'
    # 含 YAML 特殊字符时用双引号包起来（JSON 引号风格在 YAML 中合法）
    if _NEEDS_QUOTE_RE.search(text):
        return json.dumps(text, ensure_ascii=False)
    return text


def _find_section_ranges(lines: List[str]) -> Dict[str, Tuple[int, int]]:
    """返回 {段名: (起始行号, 结束行号)}；结束行号为下一段起始或文件末尾。"""
    starts: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _SECTION_RE.match(line)
        if match and not line.startswith(" "):
            starts.append((idx, match.group("name")))
    ranges: Dict[str, Tuple[int, int]] = {}
    for pos, (start, name) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        ranges[name] = (start, end)
    return ranges


def _apply_section_patch(
    lines: List[str], start: int, end: int, patch: Dict[str, Any]
) -> Dict[str, bool]:
    """在 [start, end) 区间内应用段内补丁，返回 {key: changed}。"""
    changed: Dict[str, bool] = {}
    active: Dict[str, int] = {}  # key -> 行号（仅已激活键）
    for idx in range(start, end):
        match = _KEY_RE.match(lines[idx])
        if match and not lines[idx].strip().startswith("#"):
            active[match.group("key")] = idx

    for key, value in patch.items():
        rendered = _format_yaml_value(value)
        if key in active:
            line_no = active[key]
            old = lines[line_no]
            new = f"  {key}: {rendered}"
            if old.rstrip("\n") != new:
                lines[line_no] = new + "\n"
                changed[key] = True
        else:
            # 追加到段尾（段内最后一个非空行之后）
            insert_at = end
            for idx in range(end - 1, start - 1, -1):
                if lines[idx].strip():
                    insert_at = idx + 1
                    break
            lines.insert(insert_at, f"  {key}: {rendered}\n")
            changed[key] = True
            end += 1
    return changed


def update_user_config(
    patch: Dict[str, Dict[str, Any]],
    config_path: Optional[Path] = None,
    *,
    backup: bool = True,
) -> Dict[str, Dict[str, bool]]:
    """把补丁写回用户配置文件（保留注释），返回 {段: {键: changed}}。

    patch 形如：{"llm": {"model": "x", "api_key": "sk-..."}, "context": {...}}
    若某顶层段不存在会追加到文件末尾。
    """
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        lines: List[str] = []
    else:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    ranges = _find_section_ranges(lines)
    result: Dict[str, Dict[str, bool]] = {}

    for section, section_patch in patch.items():
        if not isinstance(section_patch, dict) or not section_patch:
            continue
        if section in ranges:
            start, end = ranges[section]
            changed = _apply_section_patch(lines, start, end, section_patch)
        else:
            # 段不存在：追加新段
            changed = {}
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f"\n{section}:\n")
            for key, value in section_patch.items():
                lines.append(f"  {key}: {_format_yaml_value(value)}\n")
                changed[key] = True
        result[section] = changed

    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text("".join(lines), encoding="utf-8")
        logger.info("配置备份已写入 %s", backup_path)

    path.write_text("".join(lines), encoding="utf-8")
    logger.info("配置已写回 %s: %s", path, result)
    return result


def read_user_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """读取用户配置文件为 dict（不合并默认值，仅文件内容）。"""
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as error:  # noqa: BLE001 - 配置损坏不应炸掉设置页
        logger.warning("读取配置文件失败 %s: %s", path, error)
        return {}
