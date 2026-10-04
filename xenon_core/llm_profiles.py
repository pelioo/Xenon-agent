# -*- coding: utf-8 -*-
"""LLM 配置档案（profiles）存储。

WebUI 设置面板支持"增量添加多个供应商配置，一键切换"：
- 档案列表存放在独立 JSON（config/llm_profiles.json），不污染带注释的 xenon.yml；
- 每个档案包含 provider / base_url / api_key / model / available_models /
  model_contexts（逐模型上下文容量）/ context_max_tokens / thinking_enabled 全套字段；
- "激活"由调用方（webui/main.py）负责把档案字段合并写回 xenon.yml（走
  update_user_config），本模块只负责档案本身的持久化与查询。

JSON 结构：
{
  "active": "DeepSeek",           # 当前激活档案名（可为空字符串）
  "profiles": [
    {"name": "DeepSeek", "provider": "openai_compat", ...}
  ]
}
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from xenon_core.settings import PROJECT_ROOT
from xenon_core.media_capability import normalize_input_modalities

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_PATH = PROJECT_ROOT / "config" / "llm_profiles.json"

# 档案字段全集（顺序即展示顺序）
PROFILE_FIELDS = (
    "name",
    "provider",
    "base_url",
    "api_key",
    "model",
    "available_models",
    "model_contexts",
    "context_max_tokens",
    "thinking_enabled",
    "thinking_mode",
    "reasoning_effort",
    "input_modalities",
)

# 思考模式方言（档案级可选，auto = 按 base_url 自动适配）
THINKING_MODES = ("auto", "enabled_disabled", "adaptive_disabled", "off")

# 思考等级可选值（空串 = 跟随全局默认 max）
REASONING_EFFORTS = ("", "off", "minimal", "low", "medium", "high", "max")

# 单模型上下文容量的合法范围（与 webui 请求校验一致）
MODEL_CONTEXT_MIN = 8192
MODEL_CONTEXT_MAX = 10_000_000


def normalize_model_contexts(
    model_contexts: Any,
    available_models: Optional[List[str]] = None,
) -> Dict[str, int]:
    """规整逐模型上下文容量表：{模型名: token 数}。

    - 只保留在 available_models 中出现的模型（给定时）；
    - 非法/越界的值被丢弃（运行时回退到档案级 context_max_tokens）。
    """
    if not isinstance(model_contexts, dict):
        return {}
    allow = set(available_models) if available_models else None
    cleaned: Dict[str, int] = {}
    for key, value in model_contexts.items():
        name = str(key).strip()
        if not name or (allow is not None and name not in allow):
            continue
        try:
            tokens = int(value)
        except (TypeError, ValueError):
            continue
        if MODEL_CONTEXT_MIN <= tokens <= MODEL_CONTEXT_MAX:
            cleaned[name] = tokens
    return cleaned

_lock = threading.RLock()


def _path() -> Path:
    return Path(DEFAULT_PROFILES_PATH)


def _empty_doc() -> Dict[str, Any]:
    return {"active": "", "profiles": []}


def _load_doc(path: Optional[Path] = None) -> Dict[str, Any]:
    """读取档案文件；不存在或损坏时返回空结构（不抛异常）。"""
    p = Path(path or _path())
    if not p.exists():
        return _empty_doc()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty_doc()
        profiles = data.get("profiles")
        if not isinstance(profiles, list):
            profiles = []
        return {
            "active": str(data.get("active") or ""),
            "profiles": [pr for pr in profiles if isinstance(pr, dict) and pr.get("name")],
        }
    except Exception as error:  # noqa: BLE001 - 档案损坏不应炸掉设置页
        logger.warning("读取 LLM 档案失败 %s: %s", p, error)
        return _empty_doc()


def _save_doc(doc: Dict[str, Any], path: Optional[Path] = None) -> None:
    """原子写回档案文件（先写临时文件再替换，防中途断电损坏）。"""
    p = Path(path or _path())
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def _normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """规整档案字段：只保留已知字段、补默认值。"""
    name = str(profile.get("name") or "").strip()
    if not name:
        raise ValueError("档案名称不能为空")
    normalized: Dict[str, Any] = {
        "name": name,
        "provider": str(profile.get("provider") or "openai_compat").strip(),
        "base_url": str(profile.get("base_url") or "").strip().rstrip("/"),
        "api_key": str(profile.get("api_key") or "").strip(),
        "model": str(profile.get("model") or "").strip(),
        "available_models": [
            str(item).strip()
            for item in (profile.get("available_models") or [])
            if str(item).strip()
        ],
        "model_contexts": {},  # 占位，下方按 available_models 过滤后填入
        "context_max_tokens": int(profile.get("context_max_tokens") or 1_000_000),
        "thinking_enabled": bool(profile.get("thinking_enabled", True)),
    }
    # 思考模式方言：非法值回退 auto
    thinking_mode = str(profile.get("thinking_mode") or "auto").strip()
    normalized["thinking_mode"] = thinking_mode if thinking_mode in THINKING_MODES else "auto"
    # 思考等级：非法值回退空串（跟随全局默认）
    effort = str(profile.get("reasoning_effort") or "").strip().lower()
    normalized["reasoning_effort"] = effort if effort in REASONING_EFFORTS else ""
    # 原生多模态：输入模态（空列表 = 未配置，运行时按模型名启发式推断）
    normalized["input_modalities"] = normalize_input_modalities(
        profile.get("input_modalities")
    )
    normalized["model_contexts"] = normalize_model_contexts(
        profile.get("model_contexts"), normalized["available_models"]
    )
    if not normalized["base_url"]:
        raise ValueError("API Base URL 不能为空")
    if not normalized["model"]:
        raise ValueError("模型名不能为空")
    return normalized


def list_profiles(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """返回全部档案（含 api_key 明文，仅限服务端使用）。"""
    with _lock:
        return list(_load_doc(path)["profiles"])


def get_active_profile_name(path: Optional[Path] = None) -> str:
    with _lock:
        return _load_doc(path)["active"]


def get_profile(name: str, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    with _lock:
        for profile in _load_doc(path)["profiles"]:
            if profile["name"] == name:
                return dict(profile)
    return None


def save_profile(
    profile: Dict[str, Any],
    *,
    activate: bool = False,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """新增或更新档案（同名覆盖）。返回规整后的档案。

    activate=True 时同时把该档案设为当前激活项。
    """
    normalized = _normalize_profile(profile)
    with _lock:
        doc = _load_doc(path)
        replaced = False
        for idx, existing in enumerate(doc["profiles"]):
            if existing["name"] == normalized["name"]:
                doc["profiles"][idx] = normalized
                replaced = True
                break
        if not replaced:
            doc["profiles"].append(normalized)
        if activate:
            doc["active"] = normalized["name"]
        _save_doc(doc, path)
    logger.info("LLM 档案已保存: %s (activate=%s, replaced=%s)", normalized["name"], activate, replaced)
    return normalized


def delete_profile(name: str, path: Optional[Path] = None) -> bool:
    """删除档案；若删除的是当前激活项，active 清空。返回是否删除了档案。"""
    with _lock:
        doc = _load_doc(path)
        remaining = [pr for pr in doc["profiles"] if pr["name"] != name]
        if len(remaining) == len(doc["profiles"]):
            return False
        doc["profiles"] = remaining
        if doc["active"] == name:
            doc["active"] = ""
        _save_doc(doc, path)
    logger.info("LLM 档案已删除: %s", name)
    return True


def activate_profile(name: str, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """把指定档案设为激活项（不写回 xenon.yml，那是调用方职责）。"""
    with _lock:
        doc = _load_doc(path)
        for profile in doc["profiles"]:
            if profile["name"] == name:
                doc["active"] = name
                _save_doc(doc, path)
                logger.info("LLM 档案已激活: %s", name)
                return dict(profile)
    return None
