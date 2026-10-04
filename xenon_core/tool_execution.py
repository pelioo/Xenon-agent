from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from xenon_core.tool_payload_runtime import (
    build_compact_tool_arguments_preview,
    build_compact_tool_result_preview,
    sanitize_tool_arguments_for_execution,
)


MUTATING_TOOL_NAME_PARTS = (
    "write",
    "create",
    "append",
    "insert",
    "str_replace",
    "replace",
    "delete",
    "move",
    "copy",
)


# ═══════════════════════════════════════════════════════════════════ #
#  框架级工具执行超时兜底
#  背景：同步工具调用无 timeout 时可能永久卡死（见审计 2026-08-13）。
#  方案：线程池 + future.result(timeout)，超时抛 ToolExecutionTimeoutError，
#        走现有 except Exception 错误链路（recovery_plan），不新增通道。
#  异步路径（background=True / execute_command_async / 轮询池）不经过这里。
# ═══════════════════════════════════════════════════════════════════ #

# 框架默认超时（无 per-tool 规则时的兜底），支持环境变量覆盖
DEFAULT_TOOL_TIMEOUT = float(os.environ.get("XENON_DEFAULT_TOOL_TIMEOUT", "120"))

# per-tool 兜底超时（秒）：框架值 ≥ 工具内部超时，让内部机制先触发。
# 后缀匹配按元组顺序执行，长后缀在前。
_PER_TOOL_TIMEOUT_SUFFIXES: tuple = (
    ("_download_exec", 3700.0),   # 内部 3600s（download_exec）
    ("_download_wait", 300.0),    # 本次修复：内部默认 300s
    ("_download", 300.0),         # 内部仅 30s 连接超时 → 300s 总兜底
    ("_ocr_wait", 300.0),         # 本次修复：内部默认 300s
)
# 前缀匹配（模块级）
_PER_TOOL_TIMEOUT_PREFIXES: tuple = (
    ("terminal_handler_", 320.0),       # 内部 subprocess 300s
    ("video_handler_", 1900.0),         # 内部 1800s
    ("web_video_renderer_", 1900.0),    # 内部 1800s
)


class ToolExecutionTimeoutError(TimeoutError):
    """框架级工具执行超时。

    后台线程仍在运行（不可杀），结果已丢弃；调用方应提示模型
    "可能已部分执行，重试前先检查状态"。
    """


class _DaemonThreadPoolExecutor:
    """固定大小 daemon 线程池。

    不依赖 ThreadPoolExecutor 内部实现（跨 Python 版本稳定）；
    线程全部 daemon：程序退出时不会等待卡死的工具线程。
    """

    def __init__(self, max_workers: int = 16, thread_name_prefix: str = "tool_exec"):
        self._max_workers = max(1, int(max_workers))
        self._work_queue: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
        self._threads: List[threading.Thread] = []
        self._shutdown = False
        for index in range(self._max_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"{thread_name_prefix}_{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _worker_loop(self) -> None:
        while True:
            item = self._work_queue.get()
            if item is None:
                return  # shutdown 哨兵
            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # 线程内异常必须传给 Future
                future.set_exception(exc)
            else:
                future.set_result(result)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        future = Future()
        self._work_queue.put((future, fn, args, kwargs))
        return future

    def shutdown(self, wait: bool = False) -> None:
        self._shutdown = True
        for _ in self._threads:
            self._work_queue.put(None)
        if wait:
            for thread in self._threads:
                thread.join(timeout=5.0)


_executor: Optional[_DaemonThreadPoolExecutor] = None
_executor_lock = threading.Lock()


def _get_tool_executor() -> _DaemonThreadPoolExecutor:
    """模块级单例线程池（懒初始化）。"""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = _DaemonThreadPoolExecutor(
                    max_workers=int(os.environ.get("XENON_TOOL_EXEC_MAX_WORKERS", "16")),
                    thread_name_prefix="tool_exec",
                )
    return _executor


def _resolve_effective_timeout(tool_name: str, arguments: Dict[str, Any]) -> float:
    """解析工具调用的有效超时（秒）。

    规则：
      1. 显式传了 timeout 且 > 0 → 尊重模型的显式值（不 cap）
      2. 未传 / timeout<=0 / null → per-tool 默认表（后缀 → 前缀）
      3. 都不匹配 → DEFAULT_TOOL_TIMEOUT
    """
    explicit = arguments.get("timeout") if isinstance(arguments, dict) else None
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool) and explicit > 0:
        return float(explicit)

    lowered = (tool_name or "").lower()
    for suffix, timeout in _PER_TOOL_TIMEOUT_SUFFIXES:
        if lowered.endswith(suffix):
            return float(timeout)
    for prefix, timeout in _PER_TOOL_TIMEOUT_PREFIXES:
        if lowered.startswith(prefix):
            return float(timeout)
    return float(DEFAULT_TOOL_TIMEOUT)


def _log_timeout_discard(
    future: Future,
    tool_name: str,
    effective_timeout: float,
) -> None:
    """超时后注册回调：后台线程结束时记录结果被丢弃的 warning。"""

    def _on_done(done_future: Future) -> None:
        try:
            _ = done_future.result()
        except BaseException as exc:
            logging.getLogger(__name__).warning(
                "工具 %s 超时后线程结束（异常）: %s", tool_name, exc
            )
        else:
            logging.getLogger(__name__).warning(
                "工具 %s 超时后线程才完成（结果已丢弃，耗时 >%.0fs）",
                tool_name,
                effective_timeout,
            )

    future.add_done_callback(_on_done)


def _execute_tool_with_framework_timeout(
    tool_name: str,
    arguments: Dict[str, Any],
    execute_tool_fn: Callable[[str, Dict[str, Any]], Any],
) -> Any:
    """带框架级超时的工具执行（仅同步调用路径使用）。

    超时 → 抛 ToolExecutionTimeoutError（后台线程继续运行，结果丢弃）。
    任务自身抛出的异常（含内部 TimeoutError）原样转发，与同步调用行为一致。
    """
    effective_timeout = _resolve_effective_timeout(tool_name, arguments)
    future = _get_tool_executor().submit(execute_tool_fn, tool_name, arguments)
    try:
        return future.result(timeout=effective_timeout)
    except concurrent.futures.TimeoutError as error:
        if future.done():
            # 任务已完成但自身抛出了 TimeoutError（如 subprocess 内部超时），原样转发
            raise
        _log_timeout_discard(future, tool_name, effective_timeout)
        raise ToolExecutionTimeoutError(
            f"工具 {tool_name} 执行超时（>{effective_timeout:g}s），"
            f"后台线程仍在运行，结果已丢弃；"
            f"注意：可能已部分执行，重试前请先检查状态。"
        ) from error


def touch_tool_usage(
    tool_name: str,
    *,
    touch_single_tool_fn: Callable[[str], None],
    loaded_modules: Dict[str, Dict[str, Any]],
    now_fn: Callable[[], Any] = datetime.now,
) -> None:
    touch_single_tool_fn(tool_name)
    last_used = now_fn()
    for module_info in loaded_modules.values():
        if tool_name in module_info.get("tool_names", set()):
            module_info["last_used"] = last_used


def execute_tool_call(
    *,
    tool_call_id: str,
    tool_name: str,
    arguments_str: str,
    messages: List[Dict[str, Any]],
    parse_arguments_fn: Callable[[str], Dict[str, Any]],
    execute_tool_fn: Callable[[str, Dict[str, Any]], Any],
    record_tool_outcome_fn: Callable[..., None],
    build_tool_call_snapshot_fn: Callable[..., Dict[str, Any]],
    push_recent_tool_result_fn: Callable[[Dict[str, Any]], None],
    monitor_tool_result_snapshot_fn: Callable[[Dict[str, Any]], None],
    add_tool_message_fn: Callable[[List[Dict[str, Any]], str, str], None],
    handle_tool_error_fn: Callable[[List[Dict[str, Any]], str, Exception, str], None],
    build_recovery_plan_fn: Callable[..., Optional[Dict[str, Any]]],
    touch_single_tool_fn: Callable[[str], None],
    loaded_modules: Dict[str, Dict[str, Any]],
    set_tool_executing_fn: Callable[[bool], None],
    stream_callback: Optional[Callable[[Dict[str, Any]], None]],
    current_phase: Optional[str],
    logger: Any,
    print_fn: Callable[..., Any] = print,
    now_fn: Callable[[], Any] = datetime.now,
) -> None:
    arguments: Optional[Dict[str, Any]] = None
    try:
        arguments = sanitize_tool_arguments_for_execution(
            tool_name,
            parse_arguments_fn(arguments_str),
        )
        display_arguments = build_compact_tool_arguments_preview(tool_name, arguments)
        print_fn(f"\n\033[38;2;86;114;79m调用工具: \033[0m\033[38;2;86;114;79m{tool_name}\033[0m")
        print_fn(f"\033[38;2;86;114;79m参数: \033[0m\033[38;2;86;114;79m{display_arguments}\033[0m")

        touch_tool_usage(
            tool_name,
            touch_single_tool_fn=touch_single_tool_fn,
            loaded_modules=loaded_modules,
            now_fn=now_fn,
        )

        set_tool_executing_fn(True)
        if stream_callback:
            stream_callback(
                {
                    "type": "tool_progress",
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "content": build_tool_execution_progress_message(tool_name, arguments),
                }
            )
        _start_time = time.perf_counter()
        result = _execute_tool_with_framework_timeout(tool_name, arguments, execute_tool_fn)
        _elapsed = time.perf_counter() - _start_time
        result_text = str(result)
        result_text += f"\n\n[⏱ 执行耗时: {_elapsed:.2f}s]"
        display_result_text = build_compact_tool_result_preview(tool_name, arguments, result)
        print_fn(f"\033[38;2;86;114;79m结果: \033[0m\033[38;2;86;114;79m{display_result_text}\033[0m\n")

        record_tool_outcome_fn(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=True,
        )
        snapshot = build_tool_call_snapshot_fn(
            tool_name=tool_name,
            arguments=arguments,
            success=True,
            result=result,
        )
        push_recent_tool_result_fn(snapshot)
        monitor_tool_result_snapshot_fn(snapshot)

        if stream_callback:
            stream_callback(
                {
                    "type": "tool_result",
                    "tool_call_id": tool_call_id,
                    "content": display_result_text,
                }
            )

        add_tool_message_fn(messages, tool_call_id, result_text)

    except json.JSONDecodeError as error:
        logger.error("JSON解析失败: %s", error)
        logger.error("原始参数字符串: %r", arguments_str)
        error_msg = f"JSON解析错误: {error}\n原始参数: {arguments_str[:200]}..."
        recovery_plan = build_recovery_plan_fn(
            tool_name=tool_name,
            error=error,
            phase=current_phase,
        )
        failure_arguments = {"raw_arguments": arguments_str[:200]}
        _record_failed_tool_call(
            tool_name=tool_name,
            arguments=failure_arguments,
            result=error_msg,
            error_text=str(error),
            recovery_plan=recovery_plan,
            record_tool_outcome_fn=record_tool_outcome_fn,
            build_tool_call_snapshot_fn=build_tool_call_snapshot_fn,
            push_recent_tool_result_fn=push_recent_tool_result_fn,
            monitor_tool_result_snapshot_fn=monitor_tool_result_snapshot_fn,
        )
        print_fn(f"错误: 参数格式错误 - {error}")
        add_tool_message_fn(messages, tool_call_id, error_msg)

    except Exception as error:
        recovery_plan = build_recovery_plan_fn(
            tool_name=tool_name,
            error=error,
            phase=current_phase,
        )
        failure_arguments = arguments if arguments is not None else {"raw_arguments": arguments_str[:200]}
        _record_failed_tool_call(
            tool_name=tool_name,
            arguments=failure_arguments,
            result=str(error),
            error_text=str(error),
            recovery_plan=recovery_plan,
            record_tool_outcome_fn=record_tool_outcome_fn,
            build_tool_call_snapshot_fn=build_tool_call_snapshot_fn,
            push_recent_tool_result_fn=push_recent_tool_result_fn,
            monitor_tool_result_snapshot_fn=monitor_tool_result_snapshot_fn,
        )
        handle_tool_error_fn(messages, tool_call_id, error, "工具执行")

    finally:
        set_tool_executing_fn(False)


def build_tool_execution_progress_message(tool_name: str, arguments: Dict[str, Any]) -> str:
    if not _looks_like_mutating_tool(tool_name):
        return f"开始执行工具: {tool_name}"

    target_path = _extract_target_path(arguments)
    payload = _extract_text_payload(arguments)
    details = []
    if target_path:
        details.append(_truncate_text(target_path, 120))
    if payload:
        details.append(f"{len(payload):,} 字符 / {_count_lines(payload):,} 行")

    suffix = f" ({', '.join(details)})" if details else ""
    return f"开始执行文件操作: {tool_name}{suffix}"


def _looks_like_mutating_tool(tool_name: str) -> bool:
    lowered = (tool_name or "").lower()
    return any(part in lowered for part in MUTATING_TOOL_NAME_PARTS)


def _extract_target_path(arguments: Dict[str, Any]) -> str:
    for key in ("file_path", "path", "destination_path", "source_path"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _extract_text_payload(arguments: Dict[str, Any]) -> str:
    for key in ("content", "new_str"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _record_failed_tool_call(
    *,
    tool_name: str,
    arguments: Dict[str, Any],
    result: Any,
    error_text: str,
    recovery_plan: Optional[Dict[str, Any]],
    record_tool_outcome_fn: Callable[..., None],
    build_tool_call_snapshot_fn: Callable[..., Dict[str, Any]],
    push_recent_tool_result_fn: Callable[[Dict[str, Any]], None],
    monitor_tool_result_snapshot_fn: Callable[[Dict[str, Any]], None],
) -> None:
    record_tool_outcome_fn(
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        success=False,
        recovery_plan=recovery_plan,
    )
    snapshot = build_tool_call_snapshot_fn(
        tool_name=tool_name,
        arguments=arguments,
        success=False,
        result=result,
        error=error_text,
        recovery_summary=(recovery_plan or {}).get("summary", ""),
    )
    push_recent_tool_result_fn(snapshot)
    monitor_tool_result_snapshot_fn(snapshot)
