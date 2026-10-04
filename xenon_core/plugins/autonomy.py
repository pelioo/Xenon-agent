# -*- coding: utf-8 -*-
"""autonomy 插件：自主执行服务。

激活时封装 xenon_core.autonomy_runtime 的纯函数集合并注册为 "autonomy" 服务。

P1 约定：函数签名与 autonomy_runtime 一致（仍接受回调参数）；
P2/P3 与 AIAgent 接线后，回调将由服务内部从容器解析。
"""
from __future__ import annotations

from typing import Any

PLUGIN = {
    "id": "autonomy",
    "name": "xenon_core.plugins.autonomy",
    "inject": [],
}


class AutonomyService:
    """自主执行服务（函数命名空间 + 默认参数）。"""

    def __init__(self, config: dict, logger: Any) -> None:
        from xenon_core import autonomy_runtime as runtime

        self._runtime = runtime
        self._config = dict(config)
        self._logger = logger

    # ── 决策 ──
    def build_autonomous_decision(self, **kwargs: Any) -> Any:
        return self._runtime.build_autonomous_decision(**kwargs)

    def build_internal_resume_prompt(self, **kwargs: Any) -> str:
        return self._runtime.build_internal_resume_prompt(**kwargs)

    def select_active_goal(self, **kwargs: Any) -> Any:
        return self._runtime.select_active_goal(**kwargs)

    def should_resume_task(self, **kwargs: Any) -> bool:
        return self._runtime.should_resume_task(**kwargs)

    # ── 执行 ──
    def run_autonomous_tick(self, **kwargs: Any) -> Any:
        return self._runtime.run_autonomous_tick(**kwargs)

    def run_autonomous_cycle(self, **kwargs: Any) -> Any:
        return self._runtime.run_autonomous_cycle(**kwargs)

    def update_autonomous_progress(self, **kwargs: Any) -> Any:
        return self._runtime.update_autonomous_progress(**kwargs)

    # ── 状态 ──
    def enqueue_pending_user_input(self, **kwargs: Any) -> Any:
        return self._runtime.enqueue_pending_user_input(**kwargs)

    def get_phase_memory_snapshot(self, **kwargs: Any) -> Any:
        return self._runtime.get_phase_memory_snapshot(**kwargs)

    # ── 配置 ──
    @property
    def default_max_steps(self) -> int:
        return int(self._config.get("default_max_steps", 1))


def activate(ctx: Any) -> None:
    record = ctx.plugins().get("autonomy")
    service = AutonomyService(
        config=record.config if record else {},
        logger=__import__("logging").getLogger(__name__),
    )
    ctx.provide("autonomy", service)


def deactivate(ctx: Any) -> None:
    pass
