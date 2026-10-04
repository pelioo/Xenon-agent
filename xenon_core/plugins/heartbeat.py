# -*- coding: utf-8 -*-
"""heartbeat 插件（P1 默认禁用，见 manifest.yml 注释）。

激活是幂等的：heartbeat 模块本身是全局单例（start_heartbeat 已启动则直接复用）。
deactivate 只停用"由本插件启动"的心跳，避免误杀 CLI/webui 入口管理的心跳实例。
"""
from __future__ import annotations

from typing import Any

PLUGIN = {
    "id": "heartbeat",
    "name": "xenon_core.plugins.heartbeat",
    "inject": [],
}

_started_by_plugin = False


def activate(ctx: Any) -> None:
    global _started_by_plugin
    from xenon_core import heartbeat as hb

    was_running = hb._manager is not None
    hb.start_heartbeat(mode="plugin")
    _started_by_plugin = not was_running

    ctx.provide("heartbeat", hb)
    ctx.provide("heartbeat.stats", hb._get_stats())


def deactivate(ctx: Any) -> None:
    global _started_by_plugin
    if _started_by_plugin:
        from xenon_core import heartbeat as hb

        hb.stop_heartbeat()
        _started_by_plugin = False
