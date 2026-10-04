#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公共异步执行辅助：后台线程池 + 轮询池结果推送。

供 video_handler / web_video_renderer / program_packager / asr_handler 等
长耗时工具模块复用。核心思路与 terminal_handler 一致：

    - 长耗时操作提交到后台线程池，立即返回 task_id（调用方零阻塞）
    - 完成后结果自动推入全局轮询池（xenon_core.polling_pool），
      下一回合 peek() 即可取到，无需轮询查询接口
    - 通过 @async_capable 装饰器给 ToolManager 方法增加 mode 参数：
        mode='auto'  → 依据模块声明的 _ASYNC_ACTIONS 自动路由（默认）
        mode='sync'  → 强制同步（保持原有行为，立即返回结果）
        mode='async' → 强制后台执行（立即返回 task_id）
"""

from __future__ import annotations

import functools
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Sequence

# ── 全局后台任务注册表（进程内） ──────────────────────────────────────
_async_tasks: Dict[str, Dict[str, Any]] = {}
_async_tasks_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="tool_async")

# 结果保留上限（防内存膨胀）
_MAX_TASKS = 200


def _now() -> float:
    return time.time()


def submit_background(
    fn: Callable[[], Any],
    *,
    source: str,
    scenario: str,
    action: str = "",
    priority: int = 1,
    ttl: int = 600,
) -> Dict[str, Any]:
    """提交任务到后台线程池，立即返回 task_id；完成时自动推入轮询池。

    Args:
        fn: 无参可调用对象（用 lambda 绑定参数）
        source: 轮询池消息来源，如 "video" / "packager" / "asr"
        scenario: 轮询池消息场景，如 "video_result" / "package_result"
        action: 操作名（用于任务清单展示）
        priority: 轮询池优先级（1 最高）
        ttl: 轮询池消息存活秒数
    """
    task_id = f"t-{uuid.uuid4().hex[:8]}"
    entry: Dict[str, Any] = {
        "task_id": task_id,
        "status": "running",
        "source": source,
        "scenario": scenario,
        "action": action,
        "submitted_at": _now(),
        "finished_at": None,
        "result": None,
    }
    with _async_tasks_lock:
        _async_tasks[task_id] = entry
        # 防膨胀：超出上限时丢弃最旧已完成任务
        if len(_async_tasks) > _MAX_TASKS:
            finished = sorted(
                (k for k, v in _async_tasks.items() if v["status"] != "running"),
                key=lambda k: _async_tasks[k]["submitted_at"],
            )
            for old_key in finished[: len(_async_tasks) - _MAX_TASKS]:
                _async_tasks.pop(old_key, None)

    def _runner() -> None:
        status = "done"
        result: Any = None
        try:
            result = fn()
            if isinstance(result, dict) and result.get("success") is False:
                status = "failed"
        except Exception as exc:  # noqa: BLE001 - 兜底捕获，任务结果必须落盘
            status = "failed"
            result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}

        with _async_tasks_lock:
            if task_id in _async_tasks:
                _async_tasks[task_id].update(
                    {"status": status, "result": result, "finished_at": _now()}
                )

        # 推入全局轮询池（失败不影响任务本身）
        try:
            from xenon_core.polling_pool import get_pool

            pool = get_pool()
            pool.push_result(
                source=source,
                scenario=scenario,
                priority=priority,
                ttl=ttl,
                result={
                    "task_id": task_id,
                    "action": action,
                    "status": status,
                    "result": result,
                },
            )
        except Exception:  # noqa: BLE001 - 池不可用时静默
            pass

    _executor.submit(_runner)
    return {
        "task_id": task_id,
        "status": "running",
        "source": source,
        "scenario": scenario,
        "action": action,
        "note": "任务已在后台执行，完成后结果将自动推入轮询池（source=source, scenario=scenario）",
    }


def list_async_tasks(include_done: bool = True) -> Dict[str, Any]:
    """列出所有后台任务状态。"""
    with _async_tasks_lock:
        tasks = []
        for task_id, entry in _async_tasks.items():
            if not include_done and entry["status"] != "running":
                continue
            item = {
                "task_id": task_id,
                "status": entry["status"],
                "source": entry["source"],
                "scenario": entry["scenario"],
                "action": entry["action"],
                "submitted_at": entry["submitted_at"],
                "finished_at": entry["finished_at"],
            }
            if entry["result"] is not None:
                item["has_result"] = True
            tasks.append(item)
        tasks.sort(key=lambda t: t["submitted_at"], reverse=True)
    return {"tasks": tasks, "count": len(tasks)}


def async_capable(
    action: str,
    *,
    async_actions: Sequence[str] = (),
    source: str = "",
    scenario: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """装饰器：为 ToolManager 方法增加 mode 参数并自动路由。

    要求被装饰方法的签名包含 async_mode: str = "auto"（schema 中可见）。
    用 async_mode 命名是为了避免与业务参数 mode 冲突（如视频混合模式）。

    - async_mode='sync'  → 直接调用原方法（立即返回结果）
    - async_mode='async' → 提交后台线程池（立即返回 task_id）
    - async_mode='auto'  → 若 action 在 async_actions 中则走后台，否则同步
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            mode = kwargs.pop("async_mode", "auto")
            if mode == "async" or (mode == "auto" and action in async_actions):
                return submit_background(
                    lambda: fn(self, *args, **kwargs),
                    source=source or action,
                    scenario=scenario or f"{action}_result",
                    action=action,
                )
            return fn(self, *args, **kwargs)

        return wrapper

    return decorator
