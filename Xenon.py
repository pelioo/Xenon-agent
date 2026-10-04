import os
import sys
import json
import logging
import hashlib
import copy
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

__version__ = "0.4.9"
APP_VERSION = __version__

from xenon_core.cognitive_network import CognitiveNetworkState
from Tools.memory_query_handler import SmartMemoryToolManager
from Tools.task_chain_handler import TaskChainToolManager
from xenon_core.agent_bootstrap import bootstrap_agent as core_bootstrap_agent
from xenon_core.agent_orchestrator import AgentOrchestrator, ActionDecision
from xenon_core.autonomy_runtime import (
    build_autonomous_decision as core_build_autonomous_decision,
    build_internal_resume_prompt as core_build_internal_resume_prompt,
    enqueue_pending_user_input as core_enqueue_pending_user_input,
    get_phase_memory_snapshot as core_get_phase_memory_snapshot,
    run_autonomous_cycle as core_run_autonomous_cycle,
    run_autonomous_tick as core_run_autonomous_tick,
    select_active_goal as core_select_active_goal,
    should_resume_task as core_should_resume_task,
    update_autonomous_progress as core_update_autonomous_progress,
)
from xenon_core.context_runtime import (
    ContextManager as CoreContextManager,
    TokenCounter as CoreTokenCounter,
    calculate_actual_tokens as core_calculate_actual_tokens,
    do_context_cleanup as core_do_context_cleanup,
    format_context_token_info as core_format_context_token_info,
    get_actual_context_status as core_get_actual_context_status,
)
from xenon_core.chat_entry import handle_user_chat_entry as core_handle_user_chat_entry
from xenon_core.cli_runtime import run_interactive_agent_session
from xenon_core.chat_runtime import run_chat_cycle as core_run_chat_cycle
from xenon_core.model_request import build_chat_completion_kwargs
from xenon_core.providers import resolve_client_class
from xenon_core.cognitive_signal_runtime import (
    get_cognitive_network_summary as core_get_cognitive_network_summary,
)
from xenon_core.execution_journal import ExecutionJournal
from xenon_core.history_runtime import (
    persist_full_history_snapshot as core_persist_full_history_snapshot,
    save_api_request as core_save_api_request,
    save_memory_log as core_save_memory_log,
    save_turn_debug_trace as core_save_turn_debug_trace,
)
from xenon_core.recovery_manager import RecoveryManager
from xenon_core.runtime_control import (
    handle_interrupt as core_handle_interrupt,
    interruptible_sleep as core_interruptible_sleep,
    restore_signal_handler as core_restore_signal_handler,
    retry_request as core_retry_request,
    setup_signal_handler as core_setup_signal_handler,
)
from xenon_core.recursion_detector import RecursionDetector
from xenon_core.context_tooling import handle_context_manager_tool_call as core_handle_context_manager_tool_call
from xenon_core.context_trim import (
    auto_trim_context as core_auto_trim_context,
    ensure_context_size as core_ensure_context_size,
)
from xenon_core.orchestration_runtime import (
    prepare_orchestration_decision as core_prepare_orchestration_decision,
)
from xenon_core.prompt_runtime import (
    build_system_prompt as core_build_system_prompt,
    load_prompts as core_load_prompts,
)
from xenon_core.message_flow import (
    append_conversation_message as core_append_conversation_message,
    clone_message as core_clone_message,
    ensure_message_integrity as core_ensure_message_integrity,
    extract_tool_call_id as core_extract_tool_call_id,
    find_pending_tool_call_ids as core_find_pending_tool_call_ids,
)
from xenon_core.multi_agent_runtime import build_subagent_prompt
from xenon_core.project_memory import (
    build_project_memory_text as core_build_project_memory_text,
    cleanup_old_summaries_if_healthy as core_cleanup_old_summaries_if_healthy,
    collect_summary_source_messages as core_collect_summary_source_messages,
    emergency_context_clear as core_emergency_context_clear,
    extract_current_checkpoint as core_extract_current_checkpoint,
    generate_smart_summary as core_generate_smart_summary,
    get_project_memory_dir as core_get_project_memory_dir,
    get_project_memory_path as core_get_project_memory_path,
    inject_recent_memory_summary as core_inject_recent_memory_summary,
    write_project_memory_files as core_write_project_memory_files,
)
from xenon_core.response_runtime import (
    cleanup_reasoning_content as core_cleanup_reasoning_content,
    process_non_streaming_response as core_process_non_streaming_response,
    process_streaming_response as core_process_streaming_response,
    validate_and_fix_json as core_validate_and_fix_json,
)
from xenon_core.semantic_router_runtime import (
    build_semantic_router_catalog as core_build_semantic_router_catalog,
    infer_semantic_route as core_infer_semantic_route,
    parse_semantic_route_response as core_parse_semantic_route_response,
)
from xenon_core.tool_payload_runtime import (
    compress_tool_messages_in_place as core_compress_tool_messages_in_place,
    summarize_tool_payload_for_context as core_summarize_tool_payload_for_context,
)
from xenon_core.turn_compactor import (
    TIMESTAMP_SYSTEM_PREFIXES as CORE_TIMESTAMP_SYSTEM_PREFIXES,
    compact_history_for_next_context as core_compact_history_for_next_context,
    compact_turn_for_next_context as core_compact_turn_for_next_context,
    sanitize_messages_for_api as core_sanitize_messages_for_api,
    trim_compact_history as core_trim_compact_history,
)
from xenon_core.tool_catalog import (
    authorize_single_tool as core_authorize_single_tool,
    build_current_tools as core_build_current_tools,
    handle_get_module_list_call as core_handle_get_module_list_call,
    handle_get_module_tools_call as core_handle_get_module_tools_call,
    handle_get_tool_description_call as core_handle_get_tool_description_call,
    handle_load_module_call as core_handle_load_module_call,
    is_single_tool_loaded as core_is_single_tool_loaded,
    is_tool_loaded as core_is_tool_loaded,
    reset_loaded_tools_for_new_turn as core_reset_loaded_tools_for_new_turn,
    touch_single_tool as core_touch_single_tool,
    unload_module as core_unload_module,
)
from xenon_core.tool_dispatch import handle_tool_call_batch as core_handle_tool_call_batch
from xenon_core.tool_feedback import (
    apply_tool_outcome_state,
    build_tool_outcome_state,
)
from xenon_core.tool_execution import execute_tool_call as core_execute_tool_call
from xenon_core.tool_observability import (
    append_recent_tool_result,
    build_tool_call_snapshot as build_core_tool_call_snapshot,
    collect_recent_failures,
    decay_recent_tool_results,
    get_recent_tool_results,
    safe_stringify_result,
)
from xenon_core.turn_runtime import run_chat_turn as core_run_chat_turn
from xenon_core.tool_runtime import ToolManager as CoreToolManager

try:
    from openai import OpenAI
except ImportError:
    print("\033[91m错误: openai 模块未安装。\033[0m")
    print("\033[91m安装命令: pip install openai\033[0m")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
HISTORY_DIR = PROJECT_ROOT / ".agent_history"
CONTEXT_DIR = PROJECT_ROOT / "Context"
TRACE_LOG_DIR = PROJECT_ROOT / "logs" / "api_traces"

# ── P2：分层配置（默认 ← xenon.yml ← 机器层）──
# SETTINGS 是导入时快照，供模块级常量与无 ctx 上下文使用；
# 按次读取的配置（路由/摘要/流式/日志开关）由 AIAgent._settings() 走热配置服务。
from xenon_core.settings import load_settings

SETTINGS = load_settings()

API_KEY = SETTINGS.get("llm.api_key", "")
BASE_URL = SETTINGS.get("llm.base_url", "https://api.deepseek.com")
MODEL = SETTINGS.get("llm.model", "deepseek-v4-flash")
PROVIDER = SETTINGS.get("llm.provider", "openai_compat")  # 供应商适配器（见 xenon_core.providers）
AVAILABLE_MODELS = list(dict.fromkeys(SETTINGS.get("llm.available_models") or [MODEL]))
if MODEL not in AVAILABLE_MODELS:
    AVAILABLE_MODELS.insert(0, MODEL)

# 导入tiktoken用于token计数
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("\033[93m警告: tiktoken 模块未安装，上下文管理功能将受限。\033[0m")
    print("\033[93m安装命令: pip install tiktoken\033[0m")

MAX_RETRY_ATTEMPTS = int(SETTINGS.get("llm.retry_attempts", 3))
NETWORK_RETRY_DELAY = int(SETTINGS.get("llm.network_retry_delay", 2))
API_TIMEOUT = int(SETTINGS.get("llm.api_timeout", 120))  # API 请求超时时间（秒）
MAX_TOOL_RECURSION_DEPTH = int(SETTINGS.get("tools.max_tool_recursion_depth", 100))  # 单轮内工具递归调用上限

ENABLE_STREAMING = bool(SETTINGS.get("llm.streaming_enabled", True))  # 是否启用流式响应
ENABLE_THINKING_MODE = bool(SETTINGS.get("llm.thinking_enabled", True))  # DeepSeek thinking 开关
REASONING_EFFORT = SETTINGS.get("llm.reasoning_effort", "max")
SUMMARY_THINKING_ENABLED = bool(SETTINGS.get("summary.thinking_enabled", False))
ROUTER_THINKING_ENABLED = bool(SETTINGS.get("router.thinking_enabled", False))
ENABLE_API_REQUEST_LOGGING = bool(SETTINGS.get("logging.api_request_logging", True))  # 调试用，含敏感信息
ENABLE_MEMORY_LOGGING = bool(SETTINGS.get("logging.memory_logging", False))  # 调试用，含敏感信息

# 上下文 token 上限配置
MAX_CONTEXT_TOKENS_DEFAULT = int(SETTINGS.get("context.max_tokens_default", 1000000))
OUTPUT_TOKEN_RESERVE = int(SETTINGS.get("context.output_token_reserve", 8000))

# 【测试用】上下文 token 上限，设为 None 使用默认值，设为较小值可快速触发裁剪测试
MAX_CONTEXT_TOKENS_TEST = SETTINGS.get("context.max_tokens_test")  # 例如: 10000

# 上下文保护配置（保留给旧接口兼容；当前压缩以 checkpoint 为主，不再依赖最近 N 轮）
PROTECTED_CONVERSATION_ROUNDS = int(SETTINGS.get("context.protected_conversation_rounds", 3))

# 智能摘要配置
SUMMARY_MODEL = SETTINGS.get("summary.model") or MODEL  # 摘要默认关闭 thinking，仍使用新版模型名
SUMMARY_MAX_TOKENS = int(SETTINGS.get("summary.max_tokens", 1200))  # 摘要最大token数
SUMMARY_TIMEOUT = int(SETTINGS.get("summary.timeout", 45))  # 摘要生成超时时间（秒）

CONTEXT_COMPACT_AFTER_TURN = bool(SETTINGS.get("context.compact_after_turn", True))
KEEP_RAW_TRACE_LOG = bool(SETTINGS.get("logging.keep_raw_trace_log", True))
INCLUDE_TOOL_RESULTS_IN_NEXT_TURN = bool(SETTINGS.get("context.include_tool_results_in_next_turn", False))
INCLUDE_REASONING_IN_HISTORY = bool(SETTINGS.get("context.include_reasoning_in_history", False))
MAX_COMPACT_HISTORY_TURNS = int(SETTINGS.get("context.max_compact_history_turns", 20))
MAX_DISPLAY_HISTORY_MESSAGES = int(SETTINGS.get("context.max_display_history_messages", 200))  # 显示用，与API提交无关
ROUTER_MODEL = SETTINGS.get("router.model") or MODEL
ROUTER_TIMEOUT = int(SETTINGS.get("router.timeout", 20))
ROUTER_MAX_TOKENS = int(SETTINGS.get("router.max_tokens", 450))
SUMMARY_RECENT_MESSAGE_LIMIT = int(SETTINGS.get("summary.recent_message_limit", 60))
SUMMARY_MAX_TOOL_SNIPPET = int(SETTINGS.get("summary.max_tool_snippet", 800))
SUMMARY_MAX_MESSAGE_SNIPPET = int(SETTINGS.get("summary.max_message_snippet", 1200))
SUMMARY_INJECTION_MAX_CHARS = int(SETTINGS.get("summary.injection_max_chars", 4000))
COMPRESSED_TOOL_RESULT_MAX_CHARS = int(SETTINGS.get("tools.compressed_tool_result_max_chars", 240))
COMPRESSED_TOOL_RESULT_MAX_LINES = int(SETTINGS.get("tools.compressed_tool_result_max_lines", 8))
PROJECT_MEMORY_FILENAME = SETTINGS.get("memory.project_memory_filename", "project_context_latest.txt")
PROJECT_MEMORY_SNAPSHOT_PREFIX = SETTINGS.get("memory.project_memory_snapshot_prefix", "project_context_snapshot_")

PROMPTS_DIR = PROJECT_ROOT / SETTINGS.get("prompts.dir", "prompts")

def load_prompts() -> str:
    """
    加载 prompts 文件夹中的所有文档内容
    
    Returns:
        合并后的提示词内容字符串
    """
    return core_load_prompts(prompts_dir=PROMPTS_DIR, logger=logger)

SYSTEM_PROMPT_BASE = r"""你是 Xenon，一个智能助手。

【静态底线】
- 保持自然、有帮助、可协作，不机械展示内部流程。
- 复杂任务先理解目标和约束，再选择工具或行动。
- 读取大文件前先检查大小；超过 50KB 时用搜索、分块读取或只读关键区域。
- 涉及无法核实、实时变化或专业高风险内容时，降低确定性并说明验证路径。
"""

def get_system_prompt(fragments: Any = None) -> str:
    """
    获取系统提示词。

    fragments 传入时（P3 插件路径）：base + 插件注册片段（按激活顺序）。
    未传时：base + prompts 文件夹内容（旧行为，供未装配插件的调用方使用）。
    """
    return core_build_system_prompt(
        system_prompt_base=SYSTEM_PROMPT_BASE,
        prompts_dir=PROMPTS_DIR,
        logger=logger,
        fragments=fragments,
    )


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.COLOR_CODE = '\033[38;2;86;114;79m'
        self.RESET_CODE = '\033[0m'

    def format(self, record):
        message = super().format(record)
        return f"{self.COLOR_CODE}{message}{self.RESET_CODE}"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger(__name__)

for handler in logging.root.handlers:
    handler.setFormatter(ColoredFormatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.getLogger('httpx').setLevel(logging.WARNING)


from xenon_core.exceptions import InterruptedException


class TokenCounter(CoreTokenCounter):
    """Backward-compatible import surface for the extracted context runtime."""


class ContextManager(CoreContextManager):
    """Backward-compatible import surface for the extracted context runtime."""
class ToolManager(CoreToolManager):
    """Backward-compatible import surface for the extracted tool runtime."""


class AIAgent:
    """AIAgent 壳（P5）：状态容器 + 兼容代理。

    - 运行期操作逻辑位于 xenon_core.agent_runtime.AgentRuntimeService，
      由 agent-loop 插件装配（bootstrap 回填 agent.agent_runtime）
    - 未装配插件（XENON_PLUGINS=off）时 bootstrap 直接构造服务（回退路径）
    - __getattr__ 把运行期方法委托给服务，webui / cli / Tools 现有调用面不变
    """

    # 委托到 AgentRuntimeService 的运行期方法（经 __getattr__ 转发）
    _RUNTIME_METHODS = frozenset({
        "_clone_message", "_record_full_history_message", "_append_conversation_message",
        "_summarize_tool_payload_for_context", "_compress_tool_messages_in_place",
        "get_full_context", "set_full_context", "_persist_full_history_snapshot",
        "_save_turn_debug_trace", "_compact_turn_for_next_context",
        "_compact_history_messages_only", "_preserved_state_messages", "_save_memory_log",
        "_save_api_request", "_handle_interrupt", "_setup_signal_handler",
        "_restore_signal_handler", "_interruptible_sleep", "_retry_request",
        "_parse_arguments", "_add_tool_message", "_handle_tool_error",
        "_get_context_token_info", "_ensure_context_size", "_auto_trim_context",
        "_save_context_summary_to_memory", "_inject_recent_memory_summary",
        "_get_project_memory_dir", "_get_project_memory_path",
        "_cleanup_old_summaries_if_healthy", "_generate_smart_summary",
        "_cache_live_context_status", "_cache_streaming_context_status",
        "_get_actual_context_status", "_get_actual_token_estimate",
        "_get_current_tools", "_get_current_tool_names", "_get_current_tool_schemas",
        "_calculate_actual_tokens", "_do_context_cleanup",
        "_build_semantic_router_catalog", "_parse_semantic_route_response",
        "_infer_semantic_route", "_prepare_orchestration_decision",
        "_record_tool_outcome", "_safe_stringify_result", "_build_tool_call_snapshot",
        "_push_recent_tool_result", "_get_recent_tool_results",
        "_reset_loaded_tools_for_new_turn", "_reset_loaded_tools_for_new_turn_without_notice",
        "_handle_load_module", "_authorize_single_tool", "_is_tool_loaded",
        "_is_single_tool_loaded", "_touch_single_tool",
        "_get_or_create_manual_context_manager_tool", "_decay_recent_tool_results",
        "_text_similarity", "_get_recent_failures", "_handle_tool_calls",
        "_handle_get_tool_description", "_handle_get_module_tools",
        "_handle_get_module_list", "_handle_context_manager_tool", "_handle_execute_tool",
        "chat", "_process_chat_with_context", "_run_single_turn", "_chat",
        "_extract_tool_call_id", "_find_pending_tool_call_ids",
        "_cleanup_reasoning_content", "_cleanup_reasoning_content_for_next_request",
        "_emergency_context_clear", "_try_inject_queued_messages",
        "_process_streaming_response", "_process_non_streaming_response",
        "_validate_and_fix_json", "_ensure_message_integrity",
        "_get_phase_memory_snapshot", "_should_resume_task", "_select_active_goal",
        "_build_internal_resume_prompt", "_build_autonomous_decision",
        "_update_autonomous_progress", "autonomous_tick", "queue_pending_user_input",
        "run_autonomous_cycle", "plan_multi_agent_subtasks", "get_multi_agent_status",
        "run_multi_agent_cycle", "_execute_multi_agent_subtask",
        "_filter_tools_for_subtask", "run",
    })

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        available_models: Optional[List[str]] = None,
        max_context_tokens_default: Optional[int] = None,
    ):
        # 允许调用方（如 webui 运行时切换供应商/密钥/模型）按实例覆盖装配参数；
        # 缺省时回退到导入时快照（CLI 等既有调用面行为不变）。
        effective_api_key = api_key if api_key is not None else API_KEY
        effective_base_url = base_url if base_url is not None else BASE_URL
        effective_provider = provider if provider is not None else PROVIDER
        core_bootstrap_agent(
            self,
            openai_client_cls=resolve_client_class(effective_provider),
            api_key=effective_api_key,
            base_url=effective_base_url,
            api_timeout=API_TIMEOUT,
            router_timeout=ROUTER_TIMEOUT,
            tool_manager_cls=ToolManager,
            task_chain_manager_cls=TaskChainToolManager,
            memory_manager_cls=SmartMemoryToolManager,
            execution_journal_cls=ExecutionJournal,
            recovery_manager_cls=RecoveryManager,
            agent_orchestrator_cls=AgentOrchestrator,
            cognitive_network_cls=CognitiveNetworkState,
            context_manager_cls=ContextManager,
            tiktoken_available=TIKTOKEN_AVAILABLE,
            max_context_tokens_test=MAX_CONTEXT_TOKENS_TEST,
            max_context_tokens_default=(
                max_context_tokens_default
                if max_context_tokens_default is not None
                else MAX_CONTEXT_TOKENS_DEFAULT
            ),
            get_cognitive_network_summary_fn=self._get_cognitive_network_summary,
            logger=logger,
            history_dir_name=str(HISTORY_DIR),
            context_dir_name=str(CONTEXT_DIR),
            print_fn=print,
        )
        # 实例级模型白名单：webui 切换供应商后按新供应商的可用模型校验
        if available_models:
            self._available_models = list(dict.fromkeys(available_models))
        else:
            self._available_models = list(AVAILABLE_MODELS)
        self.model = model or MODEL
        self._last_request_estimated_tokens = None
        self._last_api_total_tokens = None
        self._last_live_context_status = None
        self._tool_cycle_count = 0  # 单轮内工具递归循环计数器
        self._recursion_detector = RecursionDetector(threshold=3)  # 内容指纹检测器

    # ── 保留在壳上的方法 ──

    def set_model(self, model: str) -> str:
        allowed = getattr(self, "_available_models", AVAILABLE_MODELS)
        if model not in allowed:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        return self.model

    def get_model(self) -> str:
        return getattr(self, "model", MODEL)

    def _settings(self):
        """返回热配置服务（settings 插件提供，watchdog 热重载）；
        未装配（如 XENON_PLUGINS=off）时回退到导入时快照 SETTINGS。"""
        ctx = getattr(self, "ctx", None)
        if ctx is not None:
            settings = ctx.get("settings")
            if settings is not None:
                return settings
        return SETTINGS

    def _get_system_prompt(self) -> str:
        """组装系统提示词：base + 插件注册片段（prompts 插件优先，其余按激活顺序）。"""
        ctx = getattr(self, "ctx", None)
        fragments = ctx.prompts() if ctx is not None else None
        return get_system_prompt(fragments=fragments)

    def _get_available_tools_message(self, decision: Optional[ActionDecision] = None) -> str:
        """兼容接口：返回系统提示词（静态）。

        供 webui 的 token 估算等调用方使用。运行时消息组装（run_chat_turn）直接使用
        get_system_prompt()，不再注入动态上下文。
        """
        return self._get_system_prompt()

    def _get_cognitive_network_summary(
        self,
        current_query: Optional[str] = None,
        current_phase: Optional[str] = None,
        current_intent: Optional[str] = None,
        recent_failures: Optional[List[str]] = None,
    ) -> str:
        """Build a compact cognitive-state summary from the persisted network."""
        return core_get_cognitive_network_summary(
            cognitive_network=self.cognitive_network,
            cached_summary=self.cognitive_network_summary,
            logger=logger,
            set_cached_summary_fn=lambda summary: setattr(self, "cognitive_network_summary", summary),
            current_query=current_query,
            current_phase=current_phase,
            current_intent=current_intent,
            recent_failures=recent_failures,
        )

    def set_stream_callback(self, callback):
        """
        设置流式回调函数。
        callback 应接受一个参数：event_dict，包含 type 和 content 等字段。
        """
        self._stream_callback = callback

    # ── 运行期方法委托（P5）──

    def __getattr__(self, name: str):
        if name in self._RUNTIME_METHODS:
            runtime = getattr(self, "agent_runtime", None)
            if runtime is not None:
                method = getattr(runtime, name, None)
                if method is not None:
                    return lambda *args, **kwargs: method(self, *args, **kwargs)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


if __name__ == "__main__":
    # ── 启动系统级心跳（CLI 模式）──
    try:
        from xenon_core.heartbeat import start_heartbeat
        start_heartbeat(mode="cli")
    except Exception:
        pass
    
    agent = AIAgent()
    try:
        agent.run()
    finally:
        # ── 停止心跳 ──
        try:
            from xenon_core.heartbeat import stop_heartbeat
            stop_heartbeat()
        except Exception:
            pass





