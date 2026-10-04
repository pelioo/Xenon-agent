from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from xenon_core.media_payload import build_user_content, parse_attachments


INTERRUPTED_TOOL_CALL_MESSAGE = "[系统自动补全] 用户发起新输入，之前的工具调用已被中断"


def handle_user_chat_entry(
    *,
    user_input: str,
    attachments: Optional[List[str]] = None,
    current_context: List[Dict[str, Any]],
    decay_recent_tool_results_fn: Callable[[str], None],
    set_interrupted_fn: Callable[[bool], None],
    set_active_user_input_fn: Callable[[str], None],
    reset_loaded_tools_for_new_turn_fn: Callable[[], Optional[str]],
    find_pending_tool_call_ids_fn: Callable[[List[Dict[str, Any]]], Any],
    append_conversation_message_fn: Callable[[List[Dict[str, Any]], Dict[str, Any]], None],
    cleanup_reasoning_content_fn: Callable[[List[Dict[str, Any]]], None],
    process_chat_with_context_fn: Callable[[str], None],
    logger: Any,
) -> None:
    set_interrupted_fn(False)
    decay_recent_tool_results_fn(user_input)
    set_active_user_input_fn(user_input)

    unload_notice = reset_loaded_tools_for_new_turn_fn()
    if unload_notice:
        append_conversation_message_fn(
            current_context,
            {"role": "system", "content": unload_notice},
        )

    pending_tool_calls = find_pending_tool_call_ids_fn(current_context)
    if pending_tool_calls:
        logger.warning(
            "检测到 %s 个未响应的 tool_calls，在新用户输入前补全响应",
            len(pending_tool_calls),
        )
        for tool_call_id in pending_tool_calls:
            append_conversation_message_fn(
                current_context,
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": INTERRUPTED_TOOL_CALL_MESSAGE,
                },
            )

    cleanup_reasoning_content_fn(current_context)
    append_conversation_message_fn(
        current_context,
        {
            "role": "user",
            "content": _build_user_message_content(user_input, attachments, logger),
        },
    )
    process_chat_with_context_fn(user_input)


def _build_user_message_content(
    user_input: str,
    attachments: Optional[List[str]],
    logger: Any,
) -> Any:
    """解析附件并构造用户消息 content（原生多模态 Phase 1）。

    无附件时返回原始字符串——存量行为 100% 不变；
    附件解析失败不得阻断对话（降级为纯文本，记录 warning）。
    """
    try:
        cleaned_text, media_refs = parse_attachments(
            user_input, attachment_paths=attachments
        )
        return build_user_content(cleaned_text, media_refs)
    except Exception as error:  # noqa: BLE001 - 解析失败不得阻断对话
        logger.warning("附件解析失败，按纯文本处理: %s", error)
        return user_input
