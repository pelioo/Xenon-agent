"""Provider 适配器契约（协议约定，非强制基类）。

一个供应商适配器 = 一块积木。要满足三个约定：

1. 导出一个可调用工厂（类或函数）：
       factory(api_key: str, base_url: str, timeout: float) -> client

2. client 必须暴露 chat.completions.create(**kwargs)，
   kwargs 为 OpenAI 兼容格式（由 xenon_core.model_request.build_chat_completion_kwargs 构建）。

3. 响应对象必须兼容 OpenAI 格式，与 xenon_core.response_runtime 的解析约定一致：
   - 非流式：response.choices[0].message.{content, reasoning_content, tool_calls}
   - 流式：  chunk.choices[0].delta.{content, reasoning_content, tool_calls}
   - 用量：  response.usage.{prompt_tokens, completion_tokens}

注册方式：在 xenon_core.providers 包底部调用
    register_provider("provider_name", factory)
一行注册，即插即用。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChatCompletions(Protocol):
    """OpenAI 兼容的 chat.completions 接口。"""

    def create(self, **kwargs: Any) -> Any:  # noqa: ANN401 - 响应对象为鸭子类型
        """发起一次对话补全请求，返回 OpenAI 格式响应。"""


@runtime_checkable
class ProviderClient(Protocol):
    """供应商客户端的最小形态。"""

    chat: ChatCompletions

    def close(self) -> None:  # noqa: D102
        """释放底层连接（可选实现）。"""
