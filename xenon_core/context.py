# -*- coding: utf-8 -*-
"""XenonContext —— Xenon 插件化改造的容器基座（P0）。

职责：
- 服务注册表：provide() / get() / has() / services()
- 事件钩子：on() / emit()（最小实现，供插件间解耦与轮次钩子使用）
- 插件生命周期：register_plugin() / activate() / deactivate() / dispose()

P0 约定：
- bootstrap 将现有"进程态"服务登记进 ctx（镜像 agent 属性，行为零变化）；
  AIAgent 上的同名属性保留为兼容代理，现有调用路径完全不受影响。
- 会话态/轮次态服务的迁移在后续阶段进行（见 docs/插件化改造设计草案.md 第 8 章）。
- P1 起由加载器按清单驱动插件激活；本类只提供容器侧能力，不做加载/扫描。

语义对齐 DSH 的 Cordis ctx：插件通过 provide() 提供服务、通过 get() 消费服务，
生命周期由 activate/deactivate 管理。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# 插件状态常量
PLUGIN_STATE_INACTIVE = "inactive"
PLUGIN_STATE_ACTIVE = "active"
PLUGIN_STATE_FAILED = "failed"


@dataclass
class PluginRecord:
    """一个插件行的登记记录。"""

    plugin_id: str
    name: str = ""
    state: str = PLUGIN_STATE_INACTIVE
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    activate_fn: Optional[Callable[["XenonContext"], None]] = None
    deactivate_fn: Optional[Callable[["XenonContext"], None]] = None
    error: Optional[str] = None


class XenonContext:
    """极简插件容器：服务注册表 + 事件钩子 + 插件生命周期。"""

    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}
        self._hooks: Dict[str, List[Callable[..., Any]]] = {}
        self._plugins: Dict[str, PluginRecord] = {}
        self._active_order: List[str] = []
        self._prompt_fragments: List[str] = []
        self._lock = threading.RLock()

    # ────────────────────────── 服务注册表 ──────────────────────────

    def provide(self, key: str, value: Any) -> None:
        """注册/覆盖一个服务。key 使用点分命名（如 'llm.routing'）。"""
        with self._lock:
            self._services[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """取服务；未注册返回 default。"""
        with self._lock:
            return self._services.get(key, default)

    def has(self, key: str) -> bool:
        """服务是否已注册。"""
        with self._lock:
            return key in self._services

    def services(self) -> Dict[str, Any]:
        """返回全部服务快照（拷贝，安全遍历）。"""
        with self._lock:
            return dict(self._services)

    # ────────────────────────── 事件钩子 ──────────────────────────

    def on(self, event: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """订阅事件；返回 fn 本身，便于用作装饰器。"""
        with self._lock:
            self._hooks.setdefault(event, []).append(fn)
        return fn

    def emit(self, event: str, *args: Any, **kwargs: Any) -> List[Any]:
        """触发事件；单个钩子异常不阻断其余钩子（该钩子的结果被跳过）。"""
        with self._lock:
            handlers = list(self._hooks.get(event, ()))
        results: List[Any] = []
        for fn in handlers:
            try:
                results.append(fn(*args, **kwargs))
            except Exception:
                continue
        return results

    # ────────────────────────── 插件生命周期 ──────────────────────────

    def register_plugin(
        self,
        plugin_id: str,
        *,
        name: str = "",
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        activate_fn: Optional[Callable[["XenonContext"], None]] = None,
        deactivate_fn: Optional[Callable[["XenonContext"], None]] = None,
    ) -> PluginRecord:
        """登记一个插件行（同 id 覆盖）。

        激活顺序由调用方（P1 加载器）按依赖拓扑保证；本方法只登记不激活。
        """
        with self._lock:
            record = PluginRecord(
                plugin_id=plugin_id,
                name=name or plugin_id,
                config=dict(config or {}),
                dependencies=list(dependencies or []),
                activate_fn=activate_fn,
                deactivate_fn=deactivate_fn,
            )
            self._plugins[plugin_id] = record
            return record

    def plugins(self) -> Dict[str, PluginRecord]:
        """插件表快照。"""
        with self._lock:
            return dict(self._plugins)

    def activate(self, plugin_id: str) -> bool:
        """激活一个已登记插件；失败置 FAILED 并记录错误，不抛出。

        返回是否本次真正执行了激活（重复激活返回 False）。
        """
        with self._lock:
            record = self._plugins.get(plugin_id)
            if record is None or record.state == PLUGIN_STATE_ACTIVE:
                return False
        try:
            if record.activate_fn is not None:
                record.activate_fn(self)
            record.state = PLUGIN_STATE_ACTIVE
            record.error = None
            with self._lock:
                self._active_order.append(plugin_id)
            return True
        except Exception as error:
            record.state = PLUGIN_STATE_FAILED
            record.error = str(error)
            return False

    def deactivate(self, plugin_id: str) -> bool:
        """停用一个插件；未登记返回 False，其余情况返回 True。"""
        with self._lock:
            record = self._plugins.get(plugin_id)
            if record is None:
                return False
        try:
            if record.deactivate_fn is not None:
                record.deactivate_fn(self)
        except Exception as error:
            record.error = str(error)
        record.state = PLUGIN_STATE_INACTIVE
        with self._lock:
            if plugin_id in self._active_order:
                self._active_order.remove(plugin_id)
        return True

    def dispose(self) -> None:
        """按激活逆序停用全部插件（进程退出 / 热重载整树重建时调用）。"""
        with self._lock:
            order = list(reversed(self._active_order))
        for plugin_id in order:
            self.deactivate(plugin_id)

    # ────────────────────────── 提示词片段（P3）──────────────────────────

    def add_prompt(self, fragment: str) -> None:
        """注册一段系统提示词片段（插件激活时调用）。

        片段按注册顺序拼接，最终由 get_system_prompt(fragments=...) 组装；
        prompts 插件把 prompts/ 目录内容注册为首个片段，保持与原行为一致。
        内容原样保存（不 strip）——与 load_prompts 的输出逐字节一致；仅跳过空白片段。
        """
        if not str(fragment or "").strip():
            return
        with self._lock:
            self._prompt_fragments.append(str(fragment))

    def prompts(self) -> List[str]:
        """已注册的提示词片段列表（拷贝）。"""
        with self._lock:
            return list(self._prompt_fragments)
