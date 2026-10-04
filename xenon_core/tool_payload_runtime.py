from __future__ import annotations

import ast
import json
from typing import Any, Callable, Dict, List, Optional, Set


TOOL_RESULT_COMPRESSED_PREFIX = "[TOOL_RESULT_COMPRESSED]"
TOOL_RESULT_ARCHIVE_NOTE = "[Full raw tool result is preserved in the archived conversation history.]"
TOOL_ARGUMENT_OMISSION_KEY = "__xenon_omitted_after_successful_write__"
WRITE_PAYLOAD_COMPACTED_KEY = "context_compacted"


class CompactedToolArgumentError(ValueError):
    """Kept for import compatibility; write payload compaction is disabled."""

    def __init__(self, tool_name: str, fields: List[str]):
        self.tool_name = tool_name
        self.fields = fields
        joined_fields = ", ".join(fields)
        super().__init__(
            f"Tool arguments contain compacted context placeholders in: {joined_fields}."
        )


def summarize_tool_payload_for_context(
    content: str,
    *,
    max_chars: int = 240,
    max_lines: int = 8,
) -> str:
    if not content:
        return ""

    def _truncate(text: str, truncate_to: int = max_chars) -> str:
        text = text.strip()
        if len(text) > truncate_to:
            return text[:truncate_to].rstrip() + "..."
        return text

    text = str(content).strip()
    if not text:
        return ""

    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        interesting_keys = [
            "success",
            "message",
            "summary",
            "error",
            "status",
            "path",
            "file_path",
            "count",
            "line_count",
            "size",
            "chunk_index",
            "total_chunks",
            "chunk_size",
            "has_more",
            "is_last_chunk",
            "next_chunk_index",
            "remaining_chunks",
            "content_complete",
        ]
        parts = []
        for key in interesting_keys:
            if key in parsed and parsed[key] not in (None, "", [], {}):
                value = parsed[key]
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                parts.append(f"{key}={_truncate(str(value), 80)}")

        if not parts:
            remaining_keys = list(parsed.keys())[:6]
            parts.append(f"keys={', '.join(remaining_keys)}")

        return "; ".join(parts)

    if isinstance(parsed, list):
        preview_items = []
        for item in parsed[:3]:
            if isinstance(item, (dict, list)):
                preview_items.append(_truncate(json.dumps(item, ensure_ascii=False), 80))
            else:
                preview_items.append(_truncate(str(item), 80))
        return f"list[{len(parsed)}]: " + " | ".join(preview_items)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return _truncate(text)

    condensed = "\n".join(lines[:max_lines])
    return _truncate(condensed)


def compact_successful_write_tool_context(
    *,
    messages: List[Dict[str, Any]],
    tool_call_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
    result: Any,
    min_chars: int = 480,
    preview_chars: int = 160,
) -> Dict[str, Any]:
    """No-op: write payload compaction is intentionally disabled."""
    return {
        "compacted": False,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "argument_fields": [],
        "result_fields": [],
        "reason": "write_compaction_disabled",
    }


def build_compact_tool_arguments_preview(
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    min_chars: int = 240,
    preview_chars: int = 120,
) -> Dict[str, Any]:
    """Return arguments unchanged; write payload compaction is disabled."""
    return dict(arguments or {})


def build_compact_tool_result_preview(
    tool_name: str,
    arguments: Dict[str, Any],
    result: Any,
    *,
    min_chars: int = 240,
    preview_chars: int = 120,
) -> str:
    """Return results unchanged; write payload compaction is disabled."""
    return str(result)


# 等待类工具的默认等待超时（与 Tools 层默认值一致；单位：秒）。
# 仅注入"等待总时长"语义的 timeout，不触碰连接超时语义的工具（如 download 的
# timeout 是 HTTP 连接超时，保持原默认，由框架级兜底负责）。
_WAIT_TOOL_TIMEOUT_SUFFIXES = (
    ("_ocr_wait", 300.0),
    ("_download_wait", 300.0),
)


def _inject_default_wait_timeout(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """对等待类工具 setdefault(timeout)，防止 timeout=None 无限等待。

    仅在调用方未显式传 timeout（键不存在）时注入；显式传值（含 null）不覆盖。
    """
    if not isinstance(arguments, dict):
        return arguments
    lowered = (tool_name or "").lower()
    for suffix, default_timeout in _WAIT_TOOL_TIMEOUT_SUFFIXES:
        if lowered.endswith(suffix):
            result = dict(arguments)
            result.setdefault("timeout", default_timeout)
            return result
    return arguments


def sanitize_tool_arguments_for_execution(tool_name: str, arguments: Any) -> Any:
    """Remove old compaction metadata keys without changing executable payloads.

    同时对等待类工具（ocr_wait / download_wait）注入默认 timeout，
    防止 timeout 缺失时无限等待（tool_runtime.py 与 tool_execution.py 两处
    调用同时受益）。
    """
    if not isinstance(arguments, dict):
        return arguments

    cleaned = {
        key: value
        for key, value in arguments.items()
        if key not in (WRITE_PAYLOAD_COMPACTED_KEY, TOOL_ARGUMENT_OMISSION_KEY)
    }
    return _inject_default_wait_timeout(tool_name, cleaned)


def compress_tool_messages_in_place(
    messages: List[Dict[str, Any]],
    *,
    summarize_tool_payload_fn: Callable[[str], str],
    protected_indices: Optional[Set[int]] = None,
    allow_protected: bool = False,
) -> int:
    protected_indices = protected_indices or set()
    compressed_count = 0

    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        if not allow_protected and index in protected_indices:
            continue

        content = (message.get("content", "") or "").strip()
        if not content or content.startswith(TOOL_RESULT_COMPRESSED_PREFIX):
            continue

        summary = summarize_tool_payload_fn(content)
        if not summary:
            continue

        compressed = (
            f"{TOOL_RESULT_COMPRESSED_PREFIX}\n"
            f"{summary}\n"
            f"{TOOL_RESULT_ARCHIVE_NOTE}"
        )
        if len(compressed) >= len(content):
            continue

        message["content"] = compressed
        compressed_count += 1

    return compressed_count
