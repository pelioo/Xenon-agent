"""Provider 注册表：供应商适配器（积木盒）。

使用方式（xenon.yml）：
    llm:
      provider: openai_compat    # 默认；OpenAI 兼容接口的供应商都可用这个

内置积木：
    openai_compat  —— OpenAI 兼容协议（OpenAI / DeepSeek / Kimi / 智谱 / 通义 /
                       Ollama / vLLM / OpenRouter / 硅基流动 …）

新增供应商 = 新增一块积木，两步：
    1. 在 xenon_core/providers/ 下新建模块（如 anthropic.py），
       实现 base.py 里的契约（工厂 + OpenAI 兼容响应）
    2. 在本文件底部 register_provider("anthropic", ...) 一行注册

核心链路（agent_runtime / semantic_router_runtime / project_memory）无感知，
因为它们只依赖 OpenAI 兼容的 chat.completions.create 接口。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from xenon_core.providers.base import ChatCompletions, ProviderClient  # noqa: F401  # 再导出契约

_REGISTRY: Dict[str, Any] = {}


def register_provider(name: str, client_cls: Any) -> None:
    """注册一个供应商适配器（幂等：同名覆盖并告警）。"""
    if name in _REGISTRY:
        import warnings

        warnings.warn(f"Provider {name!r} 已注册，将被覆盖", RuntimeWarning, stacklevel=2)
    _REGISTRY[name] = client_cls


def resolve_client_class(provider: str) -> Any:
    """按名称解析供应商客户端工厂；未知名 fail-fast 报错。"""
    try:
        return _REGISTRY[provider]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(空)"
        raise ValueError(
            f"未知 LLM provider: {provider!r}（可用: {available}）"
            f"\n请在 xenon.yml 的 llm.provider 中指定已注册的供应商。"
        ) from None


def list_providers() -> Tuple[str, ...]:
    """列出已注册的供应商名称。"""
    return tuple(sorted(_REGISTRY))


# ── 内置积木注册 ──────────────────────────────────────────────
from xenon_core.providers.openai_compat import OpenAICompatClient  # noqa: E402

register_provider("openai_compat", OpenAICompatClient)
