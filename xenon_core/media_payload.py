# -*- coding: utf-8 -*-
"""附件解析与 content parts 构造（原生多模态改造 Phase 1）。

职责（方案 §3.2b + 决策 D2/D4）：

1. ``parse_attachments`` —— 从用户文本识别附件引用（显式 ``@path`` /
   谨慎的裸路径启发式），调用 media_codec 编码，返回
   ``(清理后的文本, MediaRef 列表)``；
2. ``build_user_content`` —— MediaRef → OpenAI content parts；
   **无媒体时 content 保持 str**，存量行为 100% 不变（纯增量改造）；
3. 媒体索引 —— data URI → 展示占位（display）的进程内注册表，供历史
   落盘 / 日志红action / 上下文压缩 / 能力降级时替换 base64；
4. 落盘与红action辅助 —— ``messages_for_persistence``（base64 不落盘）、
   ``redact_media_for_log``（日志中 data URI 截断为 ``<N chars>``）；
5. 能力降级辅助 —— ``degrade_media_parts_for_text_only``（L4：纯文本
   档案自动改写为文字占位 + 工具提示，避免 400）。

设计原则：
- 显式 ``@path`` 与裸路径都要求文件真实存在（宁可漏识别，不可错识别）；
- 识别成功才从文本中移除引用 token，其余文本零改动；
- 单次消息最多附带 ``MAX_ATTACHMENTS_PER_MESSAGE`` 个附件（默认 4）。
"""
from __future__ import annotations

import json
import os
import re
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from xenon_core import media_codec


# ── 附件约束（SVG 设计稿约定：单次 ≤4 张 · 单张 ≤10MB，可配置）────────
MAX_ATTACHMENTS_PER_MESSAGE = 4
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
# v1 仅图片原生接入（音频/视频待 Phase 3 API 实测后再放开识别）
SUPPORTED_ATTACHMENT_EXTENSIONS = tuple(media_codec.IMAGE_EXTENSIONS)

_KIND_LABELS = {"image": "图片", "audio": "音频", "video": "视频"}
_KIND_TOOL_HINTS = {"image": "vision_tool", "audio": "asr 工具", "video": "video 工具"}
_TRAILING_PUNCT = "。，；：、！？,.!?;:"


@dataclass
class MediaRef:
    """一条附件引用（编码结果 + 展示占位）。"""

    kind: str          # "image" | "audio" | "video"
    source: str        # 原始引用（本地路径 / URL）
    data_uri: str      # 编码结果（发请求时注入）
    display: str       # 落盘/展示用占位符，如 "[图片: shot.png (45KB)]"
    size_bytes: int = 0
    mime: str = ""


# ── 媒体索引：data URI 指纹 → 展示元数据（供落盘/日志/降级替换）────────
_MEDIA_INDEX: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_MEDIA_INDEX_LIMIT = 256


def _media_key(data_uri: str) -> str:
    return f"{len(data_uri)}|{data_uri[:24]}|{data_uri[-120:]}"


def register_media(ref: MediaRef) -> None:
    """登记一条媒体引用的展示元数据（data URI → display）。"""
    if not ref or not ref.data_uri:
        return
    _MEDIA_INDEX[_media_key(ref.data_uri)] = {
        "display": ref.display,
        "kind": ref.kind,
        "source": ref.source,
        "size_bytes": ref.size_bytes,
    }
    while len(_MEDIA_INDEX) > _MEDIA_INDEX_LIMIT:
        _MEDIA_INDEX.popitem(last=False)


def lookup_media(data_uri: str) -> Optional[Dict[str, Any]]:
    if not isinstance(data_uri, str) or not data_uri:
        return None
    return _MEDIA_INDEX.get(_media_key(data_uri))


def clear_media_index() -> None:
    """清空媒体索引（测试用）。"""
    _MEDIA_INDEX.clear()


def indexed_media_count() -> int:
    return len(_MEDIA_INDEX)


# ── content part 解析与展示 ─────────────────────────────────────────
def _extract_part_media(part: Any) -> Tuple[Optional[str], str]:
    """从 content part 提取 (kind, uri)；非媒体 part 返回 (None, "")。"""
    if not isinstance(part, dict):
        return None, ""
    ptype = str(part.get("type") or "").strip()
    if ptype == "image_url":
        value = part.get("image_url")
        if isinstance(value, dict):
            return "image", str(value.get("url") or "")
        return "image", str(value or "")
    if ptype == "input_audio":
        value = part.get("input_audio")
        if isinstance(value, dict):
            return "audio", str(value.get("data") or "")
        return "audio", str(value or "")
    if ptype in ("video_url", "input_video"):
        value = part.get("video_url") or part.get("input_video")
        if isinstance(value, dict):
            return "video", str(value.get("url") or value.get("data") or "")
        return "video", str(value or "")
    return None, ""


def describe_media_part(part: Any) -> Optional[str]:
    """给出媒体 part 的展示占位；非媒体 part 返回 None。

    - 命中媒体索引：返回注册的 display（如 ``[图片: shot.png (45KB)]``）；
    - 未命中（如进程重启后加载的历史）：返回通用占位 ``[图片]``。
    """
    kind, uri = _extract_part_media(part)
    if kind is None:
        return None
    meta = lookup_media(uri) if uri else None
    if meta and meta.get("display"):
        return str(meta["display"])
    return f"[{_KIND_LABELS.get(kind, '媒体')}]"


def _source_of_part(part: Any) -> str:
    _, uri = _extract_part_media(part)
    meta = lookup_media(uri) if uri else None
    return str(meta.get("source") or "") if meta else ""


def _format_display(name: str, size_bytes: int, kind: str) -> str:
    label = _KIND_LABELS.get(kind, "媒体")
    kb = max(1, int(round(size_bytes / 1024)))
    return f"[{label}: {name} ({kb}KB)]"


def _mime_of_data_uri(data_uri: str) -> str:
    prefix = str(data_uri or "")[:64]
    if prefix.startswith("data:") and ";" in prefix:
        return prefix[5:prefix.find(";")]
    return ""


# ── 附件解析 ────────────────────────────────────────────────────────
_AT_TOKEN_RE = re.compile(r'@(?:"([^"]+)"|\'([^\']+)\'|(\S+))')
_PATHISH_RE = re.compile(r'(?:[A-Za-z]:\\[^\s]+|/[^\s]+)')


def _resolve_candidate(raw: str) -> Optional[Path]:
    """把候选字符串规整为真实存在的文件路径；否则返回 None。"""
    candidate = str(raw or "").strip().strip('"').strip("'")
    if not candidate:
        return None
    candidate = candidate.rstrip(_TRAILING_PUNCT)
    try:
        path = Path(os.path.expanduser(candidate))
    except Exception:
        return None
    try:
        if not path.exists() or not path.is_file():
            return None
    except OSError:
        return None
    return path


def parse_attachments(
    user_input: str,
    *,
    attachment_paths: Optional[Iterable[str]] = None,
) -> Tuple[str, List[MediaRef]]:
    """解析用户输入中的附件引用。

    返回 ``(清理后的文本, MediaRef 列表)``：

    - 显式语法 ``@path`` / ``@"path with space"``（推荐，无歧义）；
    - 裸路径：以图片扩展名结尾且文件真实存在才识别（谨慎启发式，
      防止“D:\\\\notes\\\\todo.png 这个需求”这类文本被误吞）；
    - 不存在的路径不识别为附件，原样留在文本中（边界原则）；
    - ``attachment_paths`` 为调用方显式传入的附件（如 WebUI 上传）。

    识别成功的引用会从文本中移除（展示信息由 MediaRef.display 承载，
    随 content parts 一起进入消息）；无附件时文本原样返回。
    """
    if not isinstance(user_input, str):
        user_input = "" if user_input is None else str(user_input)

    refs: List[MediaRef] = []
    seen_sources: set = set()
    remove_spans: List[Tuple[int, int]] = []

    def _attach(path: Path) -> Optional[MediaRef]:
        key = str(path).lower()
        if key in seen_sources:
            return None
        if len(refs) >= MAX_ATTACHMENTS_PER_MESSAGE:
            return None
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_ATTACHMENT_EXTENSIONS:
            return None
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if size > MAX_ATTACHMENT_BYTES:
            print(
                f"[media] 附件超过单张上限（{size / 1024 / 1024:.1f}MB > "
                f"{MAX_ATTACHMENT_BYTES // (1024 * 1024)}MB），已忽略: {path}"
            )
            return None
        data_uri = media_codec.encode_to_data_uri(str(path))
        if not data_uri:
            return None
        if len(data_uri) > media_codec.MAX_BASE64_CHARS:
            print(
                f"[media] 附件编码后超限（{len(data_uri) / 1024 / 1024:.1f}MB base64），"
                f"已忽略: {path}"
            )
            return None
        ref = MediaRef(
            kind="image",
            source=str(path),
            data_uri=data_uri,
            display=_format_display(path.name, size, "image"),
            size_bytes=size,
            mime=_mime_of_data_uri(data_uri),
        )
        refs.append(ref)
        seen_sources.add(key)
        register_media(ref)
        return ref

    def _eat_trailing_punct(text: str, start: int, end: int) -> int:
        while end < len(text) and text[end] in _TRAILING_PUNCT:
            end += 1
        return end

    # 1) 显式 @path / @"path with space"
    for match in _AT_TOKEN_RE.finditer(user_input):
        raw = match.group(1) or match.group(2) or match.group(3) or ""
        path = _resolve_candidate(raw)
        if path is None:
            continue
        if _attach(path) is not None:
            remove_spans.append((match.start(), _eat_trailing_punct(user_input, match.start(), match.end())))

    # 2) 裸路径启发式（跳过已被 @ 语法消费的区域）
    def _inside_occupied(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in remove_spans)

    for token_match in re.finditer(r"\S+", user_input):
        token_start, token_end = token_match.span()
        if _inside_occupied(token_start, token_end):
            continue
        token = token_match.group(0)
        for path_match in _PATHISH_RE.finditer(token):
            path = _resolve_candidate(path_match.group(0))
            if path is None:
                continue
            if _attach(path) is not None:
                start = token_start + path_match.start()
                end = token_start + path_match.end()
                remove_spans.append((start, _eat_trailing_punct(user_input, start, end)))
            break  # 一个 token 内最多识别一个路径

    # 3) 调用方显式附件（如 WebUI 上传）
    for raw_path in attachment_paths or []:
        path = _resolve_candidate(str(raw_path))
        if path is None:
            continue
        _attach(path)

    if not remove_spans:
        return user_input, refs

    remove_spans.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in remove_spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    chunks: List[str] = []
    cursor = 0
    for start, end in merged:
        chunks.append(user_input[cursor:start])
        cursor = end
    chunks.append(user_input[cursor:])
    cleaned = re.sub(r"[ \t]{2,}", " ", "".join(chunks)).strip()
    return cleaned, refs


def append_attachment_tokens(text: str, attachment_paths: Optional[Iterable[str]]) -> str:
    """把显式附件路径转成 ``@token`` 追加到文本（用于排队消息）。

    排队注入链路（pending input）复用同一套 ``parse_attachments`` 识别逻辑，
    避免为队列单独维护第二种附件传递结构。
    """
    tokens: List[str] = []
    for path in attachment_paths or []:
        value = str(path or "").strip()
        if not value:
            continue
        if re.search(r"\s", value):
            tokens.append(f'@"{value}"')
        else:
            tokens.append(f"@{value}")
    if not tokens:
        return text if isinstance(text, str) else ""
    base = (text or "").strip()
    return (base + " " if base else "") + " ".join(tokens)


# ── content 构造 ────────────────────────────────────────────────────
def build_user_content(text: str, media: Optional[List[MediaRef]]) -> Any:
    """构造用户消息 content。

    - 无媒体：返回 str（存量行为 100% 不变）；
    - 有媒体：返回 OpenAI content parts；文本非空时作为首个 text part。
    """
    if not media:
        return text

    media_parts: List[Dict[str, Any]] = []
    for ref in media:
        part = _ref_to_content_part(ref)
        if part is None:
            # 音频/视频原生协议未实测（Phase 3）：降级为文字占位，不丢信息
            media_parts.append({"type": "text", "text": _ref_fallback_text(ref)})
        else:
            media_parts.append(part)

    if not media_parts:
        return text

    parts: List[Dict[str, Any]] = []
    if isinstance(text, str) and text.strip():
        parts.append({"type": "text", "text": text})
    parts.extend(media_parts)
    return parts


def _ref_to_content_part(ref: MediaRef) -> Optional[Dict[str, Any]]:
    if ref.kind == "image":
        return {"type": "image_url", "image_url": {"url": ref.data_uri}}
    # 音频 input_audio / 视频 video_url 的具体协议字段需 Phase 3 API 实测，
    # 测通前不盲接（方案 §5 Phase 3）。此处返回 None 由调用方降级。
    return None


def _ref_fallback_text(ref: MediaRef) -> str:
    hint = _KIND_TOOL_HINTS.get(ref.kind, "相应工具")
    return f"{ref.display}（该媒体暂未接入原生输入；如需分析可用 {hint} 处理文件：{ref.source}）"


# ── 落盘 / 日志 / 降级 辅助 ─────────────────────────────────────────
def content_list_to_text(content: List[Any]) -> str:
    """content parts → 纯文本（媒体 part → display 占位符）。"""
    pieces: List[str] = []
    for item in content:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            described = describe_media_part(item)
            if described is not None:
                text = described
            elif "text" in item:
                text = str(item.get("text") or "")
            else:
                text = json.dumps(item, ensure_ascii=False, default=str)
        else:
            text = str(item)
        if text:
            pieces.append(text)
    return "\n".join(pieces)


def content_to_safe_text(content: Any) -> str:
    """任意 content（str / list / None）→ 纯文本；媒体 part → display 占位。

    供任何"文本化"路径使用，保证 base64 绝不进入日志/摘要/去重键等。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content_list_to_text(content)
    return str(content)


def messages_for_persistence(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """历史落盘前转换：list content → 纯文本（媒体 → display 占位符）。

    base64 绝不落盘（方案 D2 / 空间维度强约束）。返回新列表，不改原对象。
    """
    out: List[Dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            out.append(message)
            continue
        content = message.get("content")
        if isinstance(content, list):
            new_message = dict(message)
            new_message["content"] = content_list_to_text(content)
            out.append(new_message)
        else:
            out.append(message)
    return out


def extract_media_attachments(content: Any) -> List[Dict[str, Any]]:
    """从 content parts 提取媒体附件元数据（供 WebUI 消息回放显示缩略图）。

    返回 ``[{kind, name, path, size, display}]``；无媒体返回空列表。
    路径来自媒体索引（进程内）；未命中索引时 path 为空、display 为通用占位。
    """
    if not isinstance(content, list):
        return []
    items: List[Dict[str, Any]] = []
    for part in content:
        kind, uri = _extract_part_media(part)
        if kind is None:
            continue
        meta = lookup_media(uri) if uri else None
        source = str(meta.get("source") or "") if meta else ""
        display = str(meta.get("display") or "") if meta else ""
        size = int(meta.get("size_bytes") or 0) if meta else 0
        name = Path(source).name if source else ""
        items.append(
            {
                "kind": kind,
                "name": name,
                "path": source,
                "size": size,
                "display": display or f"[{_KIND_LABELS.get(kind, '媒体')}]",
            }
        )
    return items


def messages_for_session_store(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """WebUI 会话落盘转换：list content → 纯文本 + attachments 元数据。

    - 与 ``messages_for_persistence`` 同为"base64 绝不落盘"约束的落点；
    - 额外保留附件元数据（name/path/size/display），供消息回放渲染缩略图；
    - 无媒体的消息原样返回（零行为变化）；返回新列表，不改原对象。
    """
    out: List[Dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            out.append(message)
            continue
        content = message.get("content")
        if isinstance(content, list) and _content_has_media(content):
            new_message = dict(message)
            new_message["content"] = content_list_to_text(content)
            attachments = extract_media_attachments(content)
            if attachments:
                new_message["attachments"] = attachments
            out.append(new_message)
        else:
            out.append(message)
    return out


def redact_media_for_log(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """日志红action：data URI 替换为 ``data:<mime>;base64,<N chars>``。

    保留 content parts 结构（便于排障），但日志中不再出现完整 base64 串。
    返回新列表，不改原对象。
    """
    out: List[Dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            out.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        new_parts: List[Any] = []
        changed = False
        for item in content:
            redacted = _redact_part(item) if isinstance(item, dict) else None
            if redacted is not None:
                new_parts.append(redacted)
                changed = True
            else:
                new_parts.append(item)
        if changed:
            new_message = dict(message)
            new_message["content"] = new_parts
            out.append(new_message)
        else:
            out.append(message)
    return out


def _redact_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    kind, uri = _extract_part_media(part)
    if kind is None or not uri or not uri.startswith("data:"):
        return None
    prefix = uri[:64]
    mime = prefix[5:prefix.find(";")] if ";" in prefix else "application/octet-stream"
    redacted_uri = f"data:{mime};base64,<{len(uri)} chars>"
    if kind == "image":
        return {"type": "image_url", "image_url": {"url": redacted_uri}}
    if kind == "audio":
        return {"type": "input_audio", "input_audio": {"data": redacted_uri}}
    if kind == "video":
        return {"type": "video_url", "video_url": {"url": redacted_uri}}
    return None


def has_media_parts(messages: List[Dict[str, Any]]) -> bool:
    """判断消息列表中是否存在媒体 part（能力协商 gate / 保留策略用）。"""
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if _content_has_media(message.get("content")):
            return True
    return False


def degrade_media_parts_for_text_only(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """L4 能力降级：媒体 part → 文字占位（附工具提示）。

    用于当前档案 ``input_modalities`` 不含 image 的场景：图片不上行，
    改为“占位 + vision_tool 提示”，确保请求不因不支持的 content part 而 400。
    返回新列表，不改原对象。
    """
    out: List[Dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            out.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        has_media = any(
            _extract_part_media(item)[0] is not None for item in content if isinstance(item, dict)
        )
        if not has_media:
            out.append(message)
            continue

        new_parts: List[Any] = []
        for item in content:
            kind, _ = _extract_part_media(item) if isinstance(item, dict) else (None, "")
            if kind is None:
                new_parts.append(item)
                continue
            display = describe_media_part(item) or f"[{_KIND_LABELS.get(kind, '媒体')}]"
            kind_label = _KIND_LABELS.get(kind, "媒体")
            tool_hint = _KIND_TOOL_HINTS.get(kind, "相应工具")
            source = _source_of_part(item)
            if source:
                note = (
                    f"{display}\n[系统提示] 当前模型不支持原生{kind_label}输入，"
                    f"如需查看内容请使用 {tool_hint} 分析文件：{source}"
                )
            else:
                note = f"{display}\n[系统提示] 当前模型不支持原生{kind_label}输入。"
            new_parts.append({"type": "text", "text": note})

        new_message = dict(message)
        new_message["content"] = new_parts
        out.append(new_message)
    return out


# ── WebUI 上传存储 ─────────────────────────────────────────────────
def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", str(name or "")).strip(" .")
    return cleaned or "upload"


def store_uploaded_media(filename: str, data: bytes, dest_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """保存 WebUI 上传的媒体文件（校验扩展名/大小）。

    返回 ``(保存路径, 错误信息)``；成功时错误信息为 None。
    """
    suffix = Path(str(filename or "")).suffix.lower()
    if suffix not in SUPPORTED_ATTACHMENT_EXTENSIONS:
        supported = "/".join(ext.lstrip(".") for ext in SUPPORTED_ATTACHMENT_EXTENSIONS)
        return None, f"不支持的图片格式: {suffix or '(无扩展名)'}（支持 {supported}）"
    if not data:
        return None, "文件内容为空"
    if len(data) > MAX_ATTACHMENT_BYTES:
        return None, f"文件超过单张上限 {MAX_ATTACHMENT_BYTES // (1024 * 1024)}MB"

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(Path(str(filename)).name)
    unique = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}_{safe_name}"
    )
    target = dest / unique
    try:
        target.write_bytes(data)
    except OSError as error:
        return None, f"写入失败: {error}"
    return target, None



# ── 媒体保留策略（compact 后 / 发送前调用，方案 D2）────────────────
def _content_has_media(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and _extract_part_media(item)[0] is not None
        for item in content
    )


def _replace_media_parts_with_display(content: List[Any]) -> List[Any]:
    new_parts: List[Any] = []
    for item in content:
        if isinstance(item, dict):
            described = describe_media_part(item)
            if described is not None:
                new_parts.append({"type": "text", "text": described})
                continue
        new_parts.append(item)
    return new_parts


def apply_media_retention_policy(
    messages: List[Dict[str, Any]],
    *,
    keep_recent_rounds: int = 3,
) -> List[Dict[str, Any]]:
    """媒体保留策略：最近 N 轮的媒体保留，更早轮次替换为 display 占位文本。

    - "轮"按含媒体的 user 消息从后往前计数（对应"当前轮 + 最近 N 轮"语义）；
    - 更早轮次的媒体 part → 文本占位（保留文本 part；上下文经济性，方案 D2）；
    - 幂等：已是文本占位的消息不再变化；返回新列表，不改原对象。
    """
    keep = max(0, int(keep_recent_rounds or 0))
    degrade_indexes: List[int] = []
    media_rank = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or not _content_has_media(message.get("content")):
            continue
        if message.get("role") == "user":
            media_rank += 1
            if media_rank > keep:
                degrade_indexes.append(index)
        elif media_rank > keep:
            # 非 user 消息携带媒体（防御性；当前管线不会产生）跟随更早轮降级
            degrade_indexes.append(index)

    if not degrade_indexes:
        return list(messages)

    out: List[Dict[str, Any]] = list(messages)
    for index in degrade_indexes:
        message = out[index]
        if not isinstance(message, dict):
            continue
        new_message = dict(message)
        new_message["content"] = _replace_media_parts_with_display(
            message.get("content") or []
        )
        out[index] = new_message
    return out


# ── 统一安全入口（chat_entry / 排队注入 / 自主循环共用）────────────
def build_user_content_safely(
    user_input: str,
    *,
    attachment_paths: Optional[Iterable[str]] = None,
) -> Any:
    """解析附件并构造用户消息 content，失败时降级为原文本（不阻断对话）。

    无附件时返回 str——存量行为 100% 不变；附件解析失败返回原文本。
    """
    try:
        cleaned_text, media_refs = parse_attachments(
            user_input, attachment_paths=attachment_paths
        )
        return build_user_content(cleaned_text, media_refs)
    except Exception:  # noqa: BLE001 - 解析失败不得阻断对话
        return user_input
