from __future__ import annotations

import copy
import inspect
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

CHECKPOINT_SYSTEM_PREFIX = "【任务状态检查点】"


def run_chat_turn(
    *,
    user_input: str,
    internal_context: Optional[Dict[str, Any]],
    current_context: List[Dict[str, Any]],
    context_manager: Any,
    prepare_orchestration_decision_fn: Callable[..., Any],
    get_actual_context_status_fn: Callable[[], Dict[str, Any]],
    get_current_tool_names_fn: Callable[[], List[str]],
    get_static_system_prompt_fn: Callable[[], str],
    get_context_token_info_fn: Callable[..., str],
    get_current_tools_fn: Callable[[], List[Dict[str, Any]]],
    ensure_context_size_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Any],
    chat_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]], str], None],
    cleanup_old_summaries_if_healthy_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], None],
    save_memory_log_fn: Callable[..., None],
    set_current_context_fn: Callable[[List[Dict[str, Any]]], None],
    interrupted_exception_cls: type[BaseException],
    model_for_chat: str,
    logger: Any,
    print_fn: Callable[..., Any] = print,
    reset_loaded_tools_for_next_turn_fn: Optional[Callable[[], Optional[str]]] = None,
    compact_turn_after_commit_fn: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    stream_callback_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    save_memory_log_fn("user", content=user_input)
    # 编排决策在此生成并写入 self.orchestration_decision（供工具路由/恢复等使用）；
    # 决策内容不再注入对话消息。
    prepare_orchestration_decision_fn(user_input, internal_context=internal_context)

    context_copy = copy.deepcopy(current_context)
    static_content = get_static_system_prompt_fn()

    _replace_leading_system_messages(context_copy)
    context_copy.insert(0, {"role": "system", "content": static_content})

    tools = get_current_tools_fn()
    ensure_context_size_fn(context_copy, tools)

    # 当前时间信息和上下文 Token 状态不放入 API 请求（按用户要求删除）。
    # token_info 仅用于本地日志记录，便于排查上下文占用。
    token_info = _get_context_token_info(get_context_token_info_fn, context_copy, tools)
    dynamic_for_log = _get_leading_system_content(context_copy)
    if token_info:
        dynamic_for_log = dynamic_for_log + "\n\n" + token_info
    save_memory_log_fn("system", content=dynamic_for_log)

    context_committed = False
    try:
        chat_fn(context_copy, tools, model_for_chat)
        _print_answer_timestamp(print_fn)
        set_current_context_fn(_context_for_next_turn(context_copy, compact_turn_after_commit_fn))
        context_committed = True
        cleanup_old_summaries_if_healthy_fn(context_copy, tools)
    except interrupted_exception_cls:
        print_fn("\n\033[93m[对话已中断，工具调用结果已保留]\033[0m")
        set_current_context_fn(_context_for_next_turn(context_copy, compact_turn_after_commit_fn))
        context_committed = True
    except Exception as error:
        logger.error("对话过程发生错误: %s", error)
        print_fn(f"\n\033[91m错误: {error}\033[0m")
    finally:
        if reset_loaded_tools_for_next_turn_fn is not None:
            unload_notice = reset_loaded_tools_for_next_turn_fn()
            if unload_notice and context_committed and compact_turn_after_commit_fn is None:
                context_copy.append({"role": "system", "content": unload_notice})
                set_current_context_fn(context_copy)


def _get_context_token_info(
    get_context_token_info_fn: Callable[..., str],
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
) -> str:
    if _accepts_context_token_args(get_context_token_info_fn):
        return get_context_token_info_fn(
            messages=messages,
            tools=tools,
            system_message="",
        )
    return get_context_token_info_fn()


def _accepts_context_token_args(get_context_token_info_fn: Callable[..., str]) -> bool:
    try:
        signature = inspect.signature(get_context_token_info_fn)
    except (TypeError, ValueError):
        return True

    parameters = signature.parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True
    return {"messages", "tools", "system_message"}.issubset(parameters)


def _replace_leading_system_messages(messages: List[Dict[str, Any]]) -> None:
    preserved_checkpoints: List[Dict[str, Any]] = []
    while messages and messages[0].get("role") == "system":
        message = messages.pop(0)
        if str(message.get("content", "")).startswith(CHECKPOINT_SYSTEM_PREFIX):
            preserved_checkpoints.append(message)
    if preserved_checkpoints:
        messages[:0] = preserved_checkpoints


def _context_for_next_turn(
    live_messages: List[Dict[str, Any]],
    compact_turn_after_commit_fn: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    if compact_turn_after_commit_fn is None:
        return live_messages
    return compact_turn_after_commit_fn(live_messages)


def _get_leading_system_content(messages: List[Dict[str, Any]]) -> str:
    """读取首条 system 消息内容用于日志记录；无 system 消息时返回空串。"""
    if messages and messages[0].get("role") == "system":
        return str(messages[0].get("content", ""))
    return ""


def _print_answer_timestamp(
    print_fn: Callable[..., Any] = print,
) -> None:
    """每轮回答完成时，仅在终端打印完成时间用于调试。

    按用户要求（2026-08-19），提问时间/回答完成时间不再以 system 消息
    写入对话上下文，也不会出现在 API 请求中。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_fn(f"[回答完成时间: {timestamp}]")
