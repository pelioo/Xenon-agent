# -*- coding: utf-8 -*-
"""agent-loop 插件（P5）：装配 AIAgent 运行期服务。

激活时：
- 从容器取 "agent" 服务（bootstrap 在插件装配前提供）
- 构建 AgentRuntimeService 并注册为 "agent-loop" 服务
- bootstrap 装配完成后回填 agent.agent_runtime（AIAgent.__getattr__ 据此委托）

未装配本插件（XENON_PLUGINS=off 或清单禁用）时，bootstrap 直接构造服务（回退路径）。
"""
from __future__ import annotations

import logging
from typing import Any

PLUGIN = {
    "id": "agent-loop",
    "name": "xenon_core.plugins.agent_loop",
    "inject": [],
}


def activate(ctx: Any) -> None:
    from xenon_core.agent_runtime import AgentRuntimeService

    agent = ctx.get("agent")
    if agent is None:
        raise RuntimeError("agent-loop 依赖的服务缺失：agent（应由 bootstrap 提供）")
    runtime = AgentRuntimeService(agent)
    ctx.provide("agent-loop", runtime)
    logging.getLogger(__name__).info("agent-loop 已装配（运行期逻辑服务）")


def deactivate(ctx: Any) -> None:
    pass
