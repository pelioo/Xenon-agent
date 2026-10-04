# -*- coding: utf-8 -*-
"""settings 插件：分层配置服务（xenon.yml + 机器层，watchdog 热重载）。

激活时：
- 构建 Settings 服务（默认值 ← xenon.yml ← ~/.xenon/config.yml）
- 注册为 "settings" 服务并启动热重载 watcher
- 配置变更时向容器发出 "settings:changed" 事件（消费方自行订阅）

停用时停止 watcher（配置文档本身保留在服务对象中）。
"""
from __future__ import annotations

import logging
from typing import Any

PLUGIN = {
    "id": "settings",
    "name": "xenon_core.plugins.settings",
    "inject": [],
}

_watching = False


def activate(ctx: Any) -> None:
    global _watching
    from xenon_core.settings import create_settings_service

    service = create_settings_service(
        on_change=lambda settings: ctx.emit("settings:changed"),
    )
    service.watch()
    ctx.provide("settings", service)
    _watching = True


def deactivate(ctx: Any) -> None:
    global _watching
    if _watching:
        service = ctx.get("settings")
        if service is not None:
            service.stop_watch()
        _watching = False
