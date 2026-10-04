"""OpenAI 兼容协议供应商（默认适配器）。

任何提供 OpenAI 兼容 /v1/chat/completions 接口的供应商都可直接使用本适配器：
OpenAI、DeepSeek、Kimi (Moonshot)、智谱 GLM、通义 Qwen、豆包、Ollama、
vLLM、OpenRouter、硅基流动、LM Studio、OneAPI 中转等。

本模块不实现任何协议翻译逻辑——openai SDK 本身就是适配器，
客户端工厂直接复用 openai.OpenAI。
"""
from __future__ import annotations

from typing import Any, Dict

from openai import OpenAI


def build_client(
    api_key: str,
    base_url: str,
    timeout: float,
    extra_options: Dict[str, Any] | None = None,
) -> OpenAI:
    """创建 OpenAI 兼容客户端。

    extra_options 可透传 openai.OpenAI 支持的其他参数
    （如 max_retries、http_client 等），默认空。
    """
    options: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout,
    }
    if extra_options:
        options.update(extra_options)
    return OpenAI(**options)


# 工厂别名：bootstrap 按 (api_key, base_url, timeout) 位置调用
OpenAICompatClient = OpenAI
