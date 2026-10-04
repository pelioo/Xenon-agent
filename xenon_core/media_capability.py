# -*- coding: utf-8 -*-
"""模型输入模态能力协商（原生多模态改造 Phase 3 / L4）。

三层推断（优先级从高到低）：
1. 运行时显式配置（``llm.input_modalities``，由激活档案写回 xenon.yml；
   或用户在设置面板/配置文件中直接指定）；
2. 模型名启发式（本模块维护的已知模态白名单）；
3. 保守默认：仅文本（``["text"]``）。

为什么未知模型默认 text-only（而非乐观放行）：
- 两个方向的判定错误代价不对称——
  "支持图却判成文本"：图片降级为文字占位 + 工具提示（soft fail，
  用户仍可通过 vision_tool 分析文件）；
  "不支持图却判成图像"：API 直接返回 400（hard fail，对话中断）。
- 若某模型支持图像而未被启发式识别，可在档案/设置中显式配置
  ``input_modalities: ["text", "image"]`` 覆盖。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

VALID_MODALITIES = ("text", "image", "video", "audio")

# MiMo 系：官方全模态输入（text/image/video/audio → text）
_FULL_MODALITY_PATTERNS = (
    r"mimo",
)

# 已知图像输入模型名模式（大小写不敏感，命中 → text + image）
_IMAGE_CAPABLE_PATTERNS = (
    # deepseek-flash / deepseek-v4-flash 系列（用户确认多模态，2026-10）
    r"deepseek-(?:v\d+(?:\.\d+)?-)?flash",
    # OpenAI 视觉/多模态系
    r"gpt-(?:4o|4\.1|4-turbo|4-vision|5)",
    # Anthropic Claude 3 系起支持图片输入
    r"claude-(?:3|4|opus|sonnet|haiku)",
    # Google Gemini 全系多模态
    r"gemini",
    # Qwen-VL 系列
    r"qwen[\w.\-]*vl",
    # GLM 视觉系（glm-4v / glm-4.5v ...）
    r"glm-[\w.\-]*v(?:ision)?\b",
    # 经典开源视觉模型
    r"internvl|llava|minicpm-v|cogvlm|moondream",
    # 命名惯例：*-vision
    r"vision",
)


def normalize_input_modalities(value: Any) -> List[str]:
    """把任意来源的模态配置规整为合法列表；空/非法返回 ``[]``（= 未配置）。

    接受形式：list/tuple/set，或逗号/空白分隔的字符串（如 ``"text,image"``）。
    """
    if isinstance(value, str):
        items: Iterable[Any] = re.split(r"[,\s]+", value)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []
    cleaned: List[str] = []
    for item in items:
        text = str(item or "").strip().lower()
        if text in VALID_MODALITIES and text not in cleaned:
            cleaned.append(text)
    return cleaned


def infer_input_modalities(model: str) -> List[str]:
    """按模型名启发式推断输入模态；未知模型返回 ``["text"]``（保守）。"""
    name = str(model or "").strip().lower()
    if not name:
        return ["text"]
    for pattern in _FULL_MODALITY_PATTERNS:
        if re.search(pattern, name):
            return ["text", "image", "video", "audio"]
    for pattern in _IMAGE_CAPABLE_PATTERNS:
        if re.search(pattern, name):
            return ["text", "image"]
    return ["text"]


def resolve_input_modalities(
    *,
    explicit: Any = None,
    model: str = "",
) -> List[str]:
    """解析"当前生效"的输入模态：显式配置优先，其次模型名启发式。"""
    configured = normalize_input_modalities(explicit)
    if configured:
        return configured
    return infer_input_modalities(model)


def supports_image(modalities: Optional[List[str]]) -> bool:
    return "image" in (modalities or [])


def describe_modalities(modalities: Optional[List[str]], *, model: str = "") -> str:
    """生成人类可读的模态摘要（设置面板/日志用）。"""
    resolved = list(modalities or []) or infer_input_modalities(model)
    return "+".join(resolved) if resolved else "text"
