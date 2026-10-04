# -*- coding: utf-8 -*-
"""分层配置服务（P2）：Xenon.py 模块级常量迁入配置层。

合并顺序（后写覆盖先写）：
    内置默认（DEFAULT_CONFIG） ← xenon.yml 用户配置 ← ~/.xenon/config.yml 机器层

API key 解析顺序（固化到 llm.api_key）：
    xenon.yml/机器层 llm.api_key ← 环境变量（llm.api_key_env，默认 DEEPSEEK_API_KEY）
    全部缺失时返回空串并告警（2026-08-16 起不再 fail-fast：允许零配置首启，
    在 WebUI 设置面板中完成供应商配置；未配置时对话以 401 错误事件显式呈现）。

热重载：
    settings 插件（xenon_core/plugins/settings.py）持有带 watcher 的服务实例（watchdog 监听
    xenon.yml 与机器层配置，防抖后 reload 并触发 settings:changed 事件）。
    AIAgent 每次调用时经 _settings() 读取的配置（语义路由/摘要/流式/日志开关）热生效；
    进程级配置（客户端、上下文上限、tokenizer 等）为启动快照，重启生效。

P2 约定：
    - 默认值 = 迁移前的实际生效值（行为零变化）
    - deepseekconfig.py 回退层已于 2026-08-16 随文件一并移除
"""
from __future__ import annotations

import copy
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

logger = logging.getLogger(__name__)

# 默认路径：本项目根（xenon_core 的父目录）与用户主目录机器层
_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "xenon.yml"
DEFAULT_MACHINE_CONFIG_PATH = Path.home() / ".xenon" / "config.yml"

# 默认配置 = 迁移前的实际生效值（行为零变化）
DEFAULT_CONFIG: Dict[str, Any] = {
    "llm": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "available_models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "api_timeout": 120,
        "retry_attempts": 3,
        "network_retry_delay": 2,
        "thinking_enabled": True,
        "reasoning_effort": "max",
        "streaming_enabled": True,
    },
    "router": {
        # model 为空时回退到 llm.model
        "model": "",
        "timeout": 20,
        "max_tokens": 450,
        "thinking_enabled": False,
    },
    "summary": {
        # model 为空时回退到 llm.model
        "model": "",
        "max_tokens": 1200,
        "timeout": 45,
        "thinking_enabled": False,
        "recent_message_limit": 60,
        "max_tool_snippet": 800,
        "max_message_snippet": 1200,
        "injection_max_chars": 4000,
    },
    "context": {
        "max_tokens_default": 1000000,
        "output_token_reserve": 8000,
        "max_tokens_test": None,  # 设为数字可快速触发裁剪测试
        "protected_conversation_rounds": 3,
        "max_display_history_messages": 200,
        "compact_after_turn": True,
        "max_compact_history_turns": 20,
        "include_tool_results_in_next_turn": False,
        "include_reasoning_in_history": False,
    },
    "tools": {
        "dir": "",  # 空 = 默认 Tools/ 目录（P4：tools 插件读取）
        "max_loaded_modules": 5,
        "max_tool_recursion_depth": 200,
        "compressed_tool_result_max_chars": 240,
        "compressed_tool_result_max_lines": 8,
    },
    "logging": {
        "api_request_logging": True,  # 打印所有 API 请求/响应（含敏感信息，仅调试用）
        "memory_logging": False,      # 记录所有交互与推理过程（含敏感信息，仅调试用）
        "keep_raw_trace_log": True,
    },
    "prompts": {
        "dir": "prompts",
    },
    "memory": {
        "project_memory_filename": "project_context_latest.txt",
        "project_memory_snapshot_prefix": "project_context_snapshot_",
    },
}


# ────────────────────────── 文档构建 ──────────────────────────


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：override 覆盖 base（dict 深合并，其他类型整体替换）。

    YAML 中值为 None 的键（如只声明了段头 `llm:` 而没有子键）表示"不覆盖"，
    保留 base 原值——配置模板里空段是合法的。
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_document(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as error:
        logger.warning("读取配置文件失败 %s: %s", path, error)
        return {}


def resolve_api_key(document: Dict[str, Any]) -> str:
    """解析 API key：用户配置（yml/机器层） ← 环境变量。

    2026-08-16 两次调整：
    1. yml 显式 key 优先于环境变量——面板激活档案时会把该档案的 key 写进
       yml，保证"看到的就是生效的"（原来环境变量最优先，切换供应商后 401）；
    2. 全部缺失时不再抛错，返回空串并告警——允许零配置首启，在 WebUI 设置
       面板中完成配置；未配置前的对话以 401 错误事件显式呈现。
    同日 deepseekconfig.py 回退层已随文件一并移除。
    """
    llm = document.get("llm") or {}
    yaml_key = str(llm.get("api_key") or "").strip()
    if yaml_key:
        return yaml_key
    env_name = str(llm.get("api_key_env") or "DEEPSEEK_API_KEY")
    env_key = os.environ.get(env_name, "").strip()
    if env_key:
        return env_key
    logger.warning(
        "未配置任何 LLM API Key（xenon.yml 与环境变量 %s 均为空）；"
        "请启动后在 WebUI 设置面板中添加供应商配置，否则对话将返回 401。",
        env_name,
    )
    return ""


def build_document(
    *,
    config_path: Optional[Path] = None,
    machine_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """合并默认 + 用户/机器层 YAML，并固化 api_key。"""
    document = deep_merge(DEFAULT_CONFIG, {})
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    machine_path = Path(machine_path or DEFAULT_MACHINE_CONFIG_PATH)
    if config_path.exists():
        document = deep_merge(document, _load_yaml_document(config_path))
    if machine_path.exists():
        document = deep_merge(document, _load_yaml_document(machine_path))
    document["llm"]["api_key"] = resolve_api_key(document)
    return document


# ────────────────────────── Settings 对象 ──────────────────────────


class Settings:
    """分层配置对象：点分读取 + reload + watchdog 热重载。"""

    def __init__(self, document: Dict[str, Any]) -> None:
        self._document = document
        self._listeners: List[Callable[["Settings"], None]] = []
        self._lock = threading.RLock()
        self._observer: Any = None
        self._debounce_timer: Any = None
        self._debounce_lock = threading.Lock()

    # ── 读取 ──

    def get(self, key: str, default: Any = None) -> Any:
        """点分路径读取，如 get('llm.model')；未命中返回 default。"""
        with self._lock:
            node: Any = self._document
            for part in key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return node

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._document)

    # ── 变更订阅 ──

    def on_change(self, fn: Callable[["Settings"], None]) -> Callable[["Settings"], None]:
        with self._lock:
            self._listeners.append(fn)
        return fn

    def _notify(self) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(self)
            except Exception as error:
                logger.warning("settings 变更回调失败: %s", error)

    # ── 重载 ──

    def reload(
        self,
        *,
        config_path: Optional[Path] = None,
        machine_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """重新合并配置文档并通知监听者；返回新文档。"""
        document = build_document(
            config_path=config_path,
            machine_path=machine_path,
        )
        with self._lock:
            self._document = document
        self._notify()
        return document

    # ── watchdog 热重载 ──

    def watch(
        self,
        *,
        config_path: Optional[Path] = None,
        machine_path: Optional[Path] = None,
        debounce: float = 1.0,
        logger_override: Any = None,
    ) -> bool:
        """启动 watchdog 监听配置文件；变化防抖后 reload。

        返回是否成功启动（watchdog 不可用或目录不存在返回 False）。
        """
        if not WATCHDOG_AVAILABLE or self._observer is not None:
            return False
        config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        machine_path = Path(machine_path or DEFAULT_MACHINE_CONFIG_PATH)
        watched = sorted({p.parent for p in (config_path, machine_path) if p.parent.exists()})
        if not watched:
            return False

        settings = self
        log = logger_override or logger

        class _ConfigFileHandler(FileSystemEventHandler):
            def on_modified(self, event):
                self._handle(event)

            def on_created(self, event):
                self._handle(event)

            def on_moved(self, event):
                self._handle(event)

            def _handle(self, event):
                path = Path(getattr(event, "src_path", "") or "")
                if path.name not in {config_path.name, machine_path.name}:
                    return
                with settings._debounce_lock:
                    if settings._debounce_timer is not None:
                        settings._debounce_timer.cancel()
                    settings._debounce_timer = threading.Timer(
                        debounce, self._reload
                    )
                    settings._debounce_timer.daemon = True
                    settings._debounce_timer.start()

            def _reload(self):
                try:
                    settings.reload(config_path=config_path, machine_path=machine_path)
                except Exception as error:
                    log.warning("配置热重载失败: %s", error)

        self._observer = Observer()
        self._observer.daemon = True
        for directory in watched:
            self._observer.schedule(_ConfigFileHandler(), str(directory), recursive=False)
        self._observer.start()
        log.info("配置热重载已启动，监听: %s", ", ".join(str(p) for p in watched))
        return True

    def stop_watch(self) -> None:
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=3)
            except Exception:
                pass
            finally:
                self._observer = None


# ────────────────────────── 工厂 ──────────────────────────


def load_settings() -> Settings:
    """导入时快照（Xenon.py 模块常量用）；不含 watcher。"""
    return Settings(build_document())


def create_settings_service(
    *,
    config_path: Optional[Path] = None,
    machine_path: Optional[Path] = None,
    on_change: Optional[Callable[["Settings"], None]] = None,
) -> Settings:
    """settings 插件用：带 watcher 的服务实例。"""
    settings = Settings(
        build_document(config_path=config_path, machine_path=machine_path)
    )
    if on_change is not None:
        settings.on_change(on_change)
    return settings
