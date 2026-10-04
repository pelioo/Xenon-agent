# -*- coding: utf-8 -*-
"""prompts 插件（P3）：把 prompts/ 目录内容注册为系统提示词片段。

激活时：
- 读取 prompts/ 目录（沿用 prompt_runtime.load_prompts 的合并格式）
- 注册为容器提示词片段（首个片段，与旧行为"base + prompts/ 内容"一致）
- 提供 "prompts" 服务（目录路径等元信息）

其他插件可在 activate 中调用 ctx.add_prompt() 追加自己的片段，
最终由 Xenon.py 的 get_system_prompt(fragments=ctx.prompts()) 按注册顺序组装。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

PLUGIN = {
    "id": "prompts",
    "name": "xenon_core.plugins.prompts",
    "inject": [],
}


def _resolve_prompts_dir(ctx: Any) -> Path:
    settings = ctx.get("settings")
    if settings is not None:
        configured = settings.get("prompts.dir")
        if configured:
            return Path(configured)
    # 回退：settings 未装配或未配置时用项目默认
    from xenon_core.settings import PROJECT_ROOT

    return PROJECT_ROOT / "prompts"


def activate(ctx: Any) -> None:
    from xenon_core.prompt_runtime import load_prompts

    prompts_dir = _resolve_prompts_dir(ctx)
    logger = logging.getLogger(__name__)
    content = load_prompts(prompts_dir=prompts_dir, logger=logger)
    if content.strip():
        ctx.add_prompt(content)
    ctx.provide("prompts", {"dir": str(prompts_dir)})
    logger.info("prompts 插件已注册 %s 字节的提示词片段（%s）", len(content), prompts_dir)


def deactivate(ctx: Any) -> None:
    pass
