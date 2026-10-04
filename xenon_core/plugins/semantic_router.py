# -*- coding: utf-8 -*-
"""semantic-router 插件：语义路由服务。

激活时从容器取 llm.routing（OpenAI 路由客户端）与 tools（模块列表来源），
构建 SemanticRouterService 并注册为 "semantic-router" 服务。

P1 约定：仅注册服务，AIAgent 的 _infer_semantic_route 调用路径保持不变；
P2/P3 消费方切换时改走 ctx.get("semantic-router")。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

PLUGIN = {
    "id": "semantic-router",
    "name": "xenon_core.plugins.semantic_router",
    "inject": ["llm.routing", "tools"],
}


class SemanticRouterService:
    """语义路由服务：封装 xenon_core.semantic_router_runtime 的纯函数。"""

    def __init__(
        self,
        *,
        routing_client: Any,
        model: str,
        timeout: int = 20,
        max_tokens: int = 450,
        thinking_enabled: bool = False,
        reasoning_effort: Optional[str] = None,
        get_module_list_fn: Callable[[], List[str]],
        logger: Any,
    ) -> None:
        from xenon_core.semantic_router_runtime import (
            build_semantic_router_catalog,
            infer_semantic_route,
            parse_semantic_route_response,
        )

        self._build_catalog = build_semantic_router_catalog
        self._infer = infer_semantic_route
        self._parse = parse_semantic_route_response
        self._routing_client = routing_client
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._thinking_enabled = thinking_enabled
        self._reasoning_effort = reasoning_effort
        self._get_module_list_fn = get_module_list_fn
        self._logger = logger

    def route(
        self,
        user_input: str,
        tool_schemas: List[Dict[str, Any]],
        current_task: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """推断一次语义路由；失败/不可解析返回 None（与现状一致）。"""
        return self._infer(
            user_input=user_input,
            tool_schemas=tool_schemas,
            current_task=current_task,
            get_module_list_fn=self._get_module_list_fn,
            routing_client=self._routing_client,
            router_model=self._model,
            router_max_tokens=self._max_tokens,
            logger=self._logger,
            router_thinking_enabled=self._thinking_enabled,
            router_reasoning_effort=self._reasoning_effort,
        )

    def build_catalog(self, tool_schemas: List[Dict[str, Any]]) -> str:
        return self._build_catalog(
            tool_schemas=tool_schemas,
            module_names=self._get_module_list_fn(),
        )

    def parse(self, content: str) -> Optional[Dict[str, Any]]:
        return self._parse(content)


def activate(ctx: Any) -> None:
    config = dict(ctx.plugins().get("semantic-router").config)
    routing_client = ctx.get("llm.routing")
    tool_manager = ctx.get("tools")
    if routing_client is None or tool_manager is None:
        raise RuntimeError("semantic-router 依赖的服务未就绪：llm.routing / tools")

    service = SemanticRouterService(
        routing_client=routing_client,
        model=str(config.get("model", "deepseek-v4-flash")),
        timeout=int(config.get("timeout", 20)),
        max_tokens=int(config.get("max_tokens", 450)),
        thinking_enabled=bool(config.get("thinking_enabled", False)),
        reasoning_effort=config.get("reasoning_effort"),
        get_module_list_fn=tool_manager.get_module_list,
        logger=__import__("logging").getLogger(__name__),
    )
    ctx.provide("semantic-router", service)


def deactivate(ctx: Any) -> None:
    pass
