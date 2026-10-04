# -*- coding: utf-8 -*-
"""tools 插件（P4）：ToolManager 由加载器装配。

激活时：
- 从 settings 读取 tools.dir（空 = 默认 Tools/ 目录）构建 ToolManager（含 plugin.yml 元数据扫描）
- 注册 "tools" 服务（ToolManager）与 "tools.meta" 服务（4 个核心元工具 schema）

bootstrap 在插件装配完成后从容器取回并设置 agent.tool_manager 等兼容代理；
未装配本插件（XENON_PLUGINS=off 或清单禁用）时 bootstrap 回退直接创建 ToolManager（旧行为）。
"""
from __future__ import annotations

import logging
from typing import Any

PLUGIN = {
    "id": "tools",
    "name": "xenon_core.plugins.tools",
    "inject": [],
}


def activate(ctx: Any) -> None:
    from xenon_core.tool_runtime import ToolManager
    from xenon_core.tool_catalog import build_core_management_tools

    logger = logging.getLogger(__name__)
    tools_dir = None
    settings = ctx.get("settings")
    if settings is not None:
        configured = settings.get("tools.dir")
        if configured:
            tools_dir = configured

    manager = ToolManager(tools_dir=tools_dir)
    ctx.provide("tools", manager)
    ctx.provide("tools.meta", build_core_management_tools())
    logger.info(
        "tools 插件已装配：%d 个模块 %d 个工具（%s）",
        len(manager.module_names),
        len(manager.tool_schemas),
        manager.tools_dir,
    )


def deactivate(ctx: Any) -> None:
    manager = ctx.get("tools")
    if manager is not None:
        try:
            manager.stop_file_watcher()
        except Exception:
            pass
