from __future__ import annotations

from typing import Any, Dict, List, Optional


THINKING_ENABLED = "enabled"
THINKING_DISABLED = "disabled"


def normalize_thinking_type(enabled: bool, base_url: str = "", mode: str = "auto") -> str:
    """按供应商兼容 thinking.type 取值。

    mode（档案级配置，llm.thinking_mode）：
    - auto（默认）：按 base_url 自动适配（MiniMax→adaptive/disabled，
      其余→enabled/disabled）；
    - enabled_disabled：DeepSeek 风格；
    - adaptive_disabled：MiniMax 风格；
    - off：调用方应完全不发 thinking 参数（本函数不会被调到）。

    DeepSeek 用 enabled/disabled；MiniMax 只允许 adaptive/disabled
    （传 "enabled" 会 400: invalid thinking.type）。其余已验证供应商
    （DashScope/智谱/LM Studio）对 enabled/disabled 均耐受。
    """
    if mode == "enabled_disabled":
        return THINKING_ENABLED if enabled else THINKING_DISABLED
    if mode == "adaptive_disabled":
        return "adaptive" if enabled else "disabled"
    # auto：按 URL 启发式
    url = (base_url or "").lower()
    if "minimax" in url:
        return "adaptive" if enabled else "disabled"
    return THINKING_ENABLED if enabled else THINKING_DISABLED


def normalize_reasoning_effort(effort: Optional[str], base_url: str = "") -> Optional[str]:
    """按供应商兼容 reasoning_effort 取值。

    DeepSeek 支持 off/high/max；DashScope、智谱、LM Studio 等使用
    none/minimal/low/medium/high/xhigh 集合，传 "max"/"off" 会 400。
    2026-08-16 起反转映射方向：DeepSeek 系原样透传，其余供应商统一映射
    到通用集合（本地/新供应商默认安全）。
    """
    effort = (effort or "").strip()
    if not effort:
        return None
    url = (base_url or "").lower()
    if "deepseek" in url:
        return effort
    return {
        "off": "none",
        "none": "none",
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "high",
    }.get(effort, "high")


def build_chat_completion_kwargs(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    stream: Optional[bool] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    thinking_enabled: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    thinking_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Build OpenAI-compatible request kwargs for DeepSeek's thinking controls."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if tools is not None:
        kwargs["tools"] = tools
    if stream is not None:
        kwargs["stream"] = stream
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    request_extra_body = dict(extra_body or {})
    mode = (thinking_mode or "auto").strip() or "auto"
    thinking_off = mode == "off"
    if thinking_enabled is not None and not thinking_off:
        thinking_payload = dict(request_extra_body.get("thinking") or {})
        thinking_payload["type"] = normalize_thinking_type(
            bool(thinking_enabled), base_url or "", mode
        )
        request_extra_body["thinking"] = thinking_payload

    if request_extra_body:
        kwargs["extra_body"] = request_extra_body

    # thinking_mode=off 时连 reasoning_effort 一起抑制（等级是思考的子参数）
    if thinking_enabled and reasoning_effort and not thinking_off:
        kwargs["reasoning_effort"] = reasoning_effort

    return kwargs
