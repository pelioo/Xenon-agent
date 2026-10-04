# -*- coding: utf-8 -*-
"""统一启动入口（P3）：CLI 与 webui 等外部界面都从这里装配。

本模块是 Xenon 的公共装配面：
- 公开常量与类型（AIAgent / 模型 / 版本 / 上下文上限等）
- create_agent()：构造一个装配完成的 agent（bootstrap + 插件树 + 清单热重载）

P3 说明：
- webui/main.py 不再直接 `from Xenon import ...`，改经本模块取同一装配入口；
- CLI（Xenon.py __main__）本身就是外壳，直接构造 AIAgent，与 webui 共享
  bootstrap_agent + load_profile 装配路径（同一装配实现，两个入口）；
- P5 将把 AIAgent 本体移入 xenon_core，届时本模块成为唯一装配点。
"""
from __future__ import annotations

from typing import Any

# 公共装配面：来自外壳 Xenon.py 的公开导出（webui 等外部界面的唯一来源）
from Xenon import (  # noqa: F401  (re-export)
    AIAgent,
    APP_VERSION,
    AVAILABLE_MODELS,
    BASE_URL,
    API_KEY,
    MAX_CONTEXT_TOKENS_DEFAULT,
    MODEL,
    get_system_prompt,
)


def create_agent(**kwargs: Any) -> AIAgent:
    """构造一个装配完成的 agent（bootstrap + 插件树 + 清单热重载）。"""
    return AIAgent(**kwargs)
