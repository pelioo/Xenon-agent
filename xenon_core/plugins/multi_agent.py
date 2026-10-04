# -*- coding: utf-8 -*-
"""multi-agent 插件：子代理调度服务。

激活时注册 MultiAgentService（惰性创建 MultiAgentRuntime 实例，文件态队列）。

P1 约定：AIAgent 的 multi_agent_runtime 属性（bootstrap 创建的实例）保持不变；
本服务是容器侧的能力入口，P2/P3 消费方切换后收敛为单一实例。
"""
from __future__ import annotations

from typing import Any

PLUGIN = {
    "id": "multi-agent",
    "name": "xenon_core.plugins.multi_agent",
    "inject": [],
}


class MultiAgentService:
    """子代理调度服务：惰性持有 MultiAgentRuntime，API 透传。"""

    def __init__(self, *, queue_path: str = "Tasks/multi_agent_queue.json", default_subtasks: int = 2) -> None:
        self._queue_path = queue_path
        self._default_subtasks = int(default_subtasks)
        self._runtime = None

    def get_runtime(self) -> Any:
        if self._runtime is None:
            from xenon_core.multi_agent_runtime import MultiAgentRuntime

            self._runtime = MultiAgentRuntime(queue_path=self._queue_path)
        return self._runtime

    @property
    def default_subtasks(self) -> int:
        return self._default_subtasks

    def __getattr__(self, name: str) -> Any:
        # create_run / get_status / load_state / ... 直接透传
        return getattr(self.get_runtime(), name)


def activate(ctx: Any) -> None:
    record = ctx.plugins().get("multi-agent")
    config = record.config if record else {}
    service = MultiAgentService(
        queue_path=str(config.get("queue_path", "Tasks/multi_agent_queue.json")),
        default_subtasks=int(config.get("default_subtasks", 2)),
    )
    ctx.provide("multi-agent", service)


def deactivate(ctx: Any) -> None:
    pass
