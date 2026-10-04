# -*- coding: utf-8 -*-
"""AgentRuntimeService（P5）：AIAgent 的操作逻辑宿主。

AIAgent 只保留状态与少量助手方法，运行期逻辑全部在此类中，由 agent-loop 插件装配
（bootstrap 回填 agent.agent_runtime；XENON_PLUGINS=off 时由 bootstrap 直接构造）。

转换约定（从原 AIAgent 方法迁移）：
- 每个方法首参为 agent（AIAgent 实例 = 状态容器）
- 方法体内 agent 状态/方法一律经 agent.xxx 访问（委托链：agent._m → AIAgent.__getattr__ → 本类）
- 模块常量改为 agent._settings().get(key, <与 DEFAULT_CONFIG 一致的默认值>) 热读取
"""
from __future__ import annotations

import copy
import json
import logging
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from xenon_core.exceptions import InterruptedException
from xenon_core.agent_orchestrator import ActionDecision
from xenon_core.cli_runtime import run_interactive_agent_session
from xenon_core.turn_compactor import (
    TIMESTAMP_SYSTEM_PREFIXES as CORE_TIMESTAMP_SYSTEM_PREFIXES,
    compact_history_for_next_context as core_compact_history_for_next_context,
    compact_turn_for_next_context as core_compact_turn_for_next_context,
    sanitize_messages_for_api as core_sanitize_messages_for_api,
    trim_compact_history as core_trim_compact_history,
)
from xenon_core.message_flow import (
    append_conversation_message as core_append_conversation_message,
    clone_message as core_clone_message,
    ensure_message_integrity as core_ensure_message_integrity,
    extract_tool_call_id as core_extract_tool_call_id,
    find_pending_tool_call_ids as core_find_pending_tool_call_ids,
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
)
from xenon_core.tool_dispatch import handle_tool_call_batch as core_handle_tool_call_batch
from xenon_core.tool_execution import execute_tool_call as core_execute_tool_call
from xenon_core.tool_feedback import apply_tool_outcome_state, build_tool_outcome_state
from xenon_core.tool_observability import (
    append_recent_tool_result,
    build_tool_call_snapshot as build_core_tool_call_snapshot,
    collect_recent_failures,
    decay_recent_tool_results,
    get_recent_tool_results,
    safe_stringify_result,
)
from xenon_core.turn_runtime import run_chat_turn as core_run_chat_turn
from xenon_core.chat_runtime import run_chat_cycle as core_run_chat_cycle
from xenon_core.model_request import normalize_reasoning_effort
from xenon_core.chat_entry import handle_user_chat_entry as core_handle_user_chat_entry
from xenon_core.history_runtime import (
    persist_full_history_snapshot as core_persist_full_history_snapshot,
    save_api_request as core_save_api_request,
    save_memory_log as core_save_memory_log,
    save_turn_debug_trace as core_save_turn_debug_trace,
)
from xenon_core.media_capability import resolve_input_modalities
from xenon_core.media_payload import (
    append_attachment_tokens,
    apply_media_retention_policy,
    build_user_content_safely,
    degrade_media_parts_for_text_only,
    has_media_parts,
)
from xenon_core.runtime_control import (
    handle_interrupt as core_handle_interrupt,
    interruptible_sleep as core_interruptible_sleep,
    restore_signal_handler as core_restore_signal_handler,
    retry_request as core_retry_request,
    setup_signal_handler as core_setup_signal_handler,
)
from xenon_core.context_tooling import handle_context_manager_tool_call as core_handle_context_manager_tool_call
from xenon_core.context_trim import auto_trim_context as core_auto_trim_context, ensure_context_size as core_ensure_context_size
from xenon_core.orchestration_runtime import prepare_orchestration_decision as core_prepare_orchestration_decision
from xenon_core.semantic_router_runtime import (
    build_semantic_router_catalog as core_build_semantic_router_catalog,
    infer_semantic_route as core_infer_semantic_route,
    parse_semantic_route_response as core_parse_semantic_route_response,
)
from xenon_core.tool_payload_runtime import (
    compress_tool_messages_in_place as core_compress_tool_messages_in_place,
    summarize_tool_payload_for_context as core_summarize_tool_payload_for_context,
)
from xenon_core.response_runtime import (
    cleanup_reasoning_content as core_cleanup_reasoning_content,
    process_non_streaming_response as core_process_non_streaming_response,
    process_streaming_response as core_process_streaming_response,
    validate_and_fix_json as core_validate_and_fix_json,
)
from xenon_core.context_runtime import (
    calculate_actual_tokens as core_calculate_actual_tokens,
    do_context_cleanup as core_do_context_cleanup,
    format_context_token_info as core_format_context_token_info,
    get_actual_context_status as core_get_actual_context_status,
)
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
from xenon_core.multi_agent_runtime import build_subagent_prompt

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_LOG_DIR = PROJECT_ROOT / "logs" / "api_traces"


def _cfg(agent: Any, key: str, default: Any = None) -> Any:
    """热配置读取：agent._settings() 未命中时用调用点传入的默认值
    （默认值与 xenon_core.settings.DEFAULT_CONFIG 一致）。"""
    return agent._settings().get(key, default)


class AgentRuntimeService:
    """AIAgent 运行期操作逻辑（对话流 / 上下文管理 / 工具处理 / 自主执行 / 多代理）。"""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    # ═══════════════════════ 消息与历史 ═══════════════════════

    def _clone_message(self, agent: Any, message: Dict[str, Any]) -> Dict[str, Any]:
        return core_clone_message(message)

    def _record_full_history_message(self, agent: Any, message: Dict[str, Any]):
        if _cfg(agent, "context.compact_after_turn", True):
            return
        agent.full_conversation_history.append(agent._clone_message(message))
        agent._persist_full_history_snapshot()

    def _append_conversation_message(self, agent: Any, messages: List[Dict], message: Dict[str, Any], record_full_history: bool = True):
        role = message.get("role", "")
        if role in {"user", "assistant", "tool"}:
            agent.display_history.append(agent._clone_message(message))
            max_display = int(_cfg(agent, "context.max_display_history_messages", 200))
            if len(agent.display_history) > max_display:
                agent.display_history = agent.display_history[-max_display:]

        core_append_conversation_message(
            messages,
            message,
            record_full_history=record_full_history and not _cfg(agent, "context.compact_after_turn", True),
            record_full_history_fn=agent._record_full_history_message,
        )

    def _summarize_tool_payload_for_context(self, agent: Any, content: str) -> str:
        return core_summarize_tool_payload_for_context(
            content,
            max_chars=int(_cfg(agent, "tools.compressed_tool_result_max_chars", 240)),
            max_lines=int(_cfg(agent, "tools.compressed_tool_result_max_lines", 8)),
        )

    def _compress_tool_messages_in_place(
        self,
        agent: Any,
        messages: List[Dict],
        protected_indices: Optional[set] = None,
        allow_protected: bool = False,
    ) -> int:
        return core_compress_tool_messages_in_place(
            messages,
            summarize_tool_payload_fn=agent._summarize_tool_payload_for_context,
            protected_indices=protected_indices,
            allow_protected=allow_protected,
        )

    def get_full_context(self, agent: Any) -> List[Dict[str, Any]]:
        if agent.display_history:
            return copy.deepcopy(agent.display_history)
        return copy.deepcopy(agent.full_conversation_history)

    def set_full_context(self, agent: Any, messages: List[Dict[str, Any]]):
        if _cfg(agent, "context.compact_after_turn", True):
            agent.display_history = copy.deepcopy(messages or [])
            max_display = int(_cfg(agent, "context.max_display_history_messages", 200))
            if len(agent.display_history) > max_display:
                agent.display_history = agent.display_history[-max_display:]

            sanitized = core_compact_history_for_next_context(messages or [])
            next_context = core_trim_compact_history(
                sanitized,
                int(_cfg(agent, "context.max_compact_history_turns", 20)),
            )
            agent.compact_history = agent._compact_history_messages_only(next_context)
            agent.full_conversation_history = copy.deepcopy(agent.compact_history)
            agent.current_context = copy.deepcopy(next_context)
        else:
            agent.full_conversation_history = copy.deepcopy(messages or [])
            agent.display_history = copy.deepcopy(agent.full_conversation_history)
        agent._persist_full_history_snapshot()

    def _persist_full_history_snapshot(self, agent: Any):
        core_persist_full_history_snapshot(
            full_conversation_history=agent.full_conversation_history,
            history_dir=agent.history_dir,
            history_session_id=agent.history_session_id,
            logger=logger,
        )

    def _save_turn_debug_trace(self, agent: Any, turn_messages: List[Dict[str, Any]]):
        core_save_turn_debug_trace(
            enabled=bool(_cfg(agent, "logging.keep_raw_trace_log", True)),
            turn_messages=turn_messages,
            trace_dir=TRACE_LOG_DIR,
            logger=logger,
            metadata={"active_user_input": getattr(agent, "_active_user_input", "")},
            max_files=20,
        )

    def _compact_turn_for_next_context(self, agent: Any, turn_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        agent._save_turn_debug_trace(turn_messages)
        max_turns = int(_cfg(agent, "context.max_compact_history_turns", 20))

        if _cfg(agent, "context.include_tool_results_in_next_turn", False):
            next_context = core_sanitize_messages_for_api(
                turn_messages,
                preserve_current_toolchain=True,
                include_reasoning=_cfg(agent, "context.include_reasoning_in_history", False),
            )
            compact_snapshot = core_trim_compact_history(next_context, max_turns)
            next_context = self._apply_media_retention(agent, compact_snapshot)
            agent.compact_history = agent._compact_history_messages_only(next_context)
            agent.full_conversation_history = copy.deepcopy(agent.compact_history)
            agent._persist_full_history_snapshot()
            return copy.deepcopy(next_context)
        else:
            state_messages = agent._preserved_state_messages(turn_messages)
            previous_history = core_compact_history_for_next_context(agent.full_conversation_history)
            current_turn = core_compact_turn_for_next_context(turn_messages)
            next_context = state_messages + previous_history + current_turn

        next_context = core_trim_compact_history(next_context, max_turns)
        # 原生多模态（Phase 1）：最近 N 轮保留完整媒体，更早轮次降级为展示占位
        # （上下文经济性，方案 D2）；降级发生在本轮快照/落盘派生之前，保持三处一致。
        next_context = self._apply_media_retention(agent, next_context)
        agent.compact_history = agent._compact_history_messages_only(next_context)
        agent.full_conversation_history = copy.deepcopy(agent.compact_history)
        agent._persist_full_history_snapshot()
        return copy.deepcopy(next_context)

    def _apply_media_retention(self, agent: Any, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """媒体保留策略：最近 N 轮完整保留，更早轮次 → display 占位（无媒体零开销）。"""
        if not has_media_parts(messages):
            return messages
        return apply_media_retention_policy(
            messages,
            keep_recent_rounds=int(_cfg(agent, "context.media_retention_rounds", 3)),
        )

    def _model_supports_image(self, agent: Any) -> bool:
        """当前模型是否支持原生图像输入（显式配置优先，模型名启发式兜底）。"""
        modalities = resolve_input_modalities(
            explicit=_cfg(agent, "llm.input_modalities", None),
            model=str(agent.get_model() or ""),
        )
        return "image" in modalities

    def _apply_media_policy(self, agent: Any, api_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """发送前媒体策略（方案 D2 保留策略 + L4 能力降级）。

        - 最近 N 轮保留完整媒体、更早轮次 → 展示占位（对 compact 链已降级过的
          列表是幂等操作，同时覆盖未经过 compact 的旁路入口）；
        - 当前模型不支持图像输入时，全部媒体 → 文字占位 + 工具提示（防 400）。
        """
        if not has_media_parts(api_messages):
            return api_messages
        messages = apply_media_retention_policy(
            api_messages,
            keep_recent_rounds=int(_cfg(agent, "context.media_retention_rounds", 3)),
        )
        if not self._model_supports_image(agent):
            logger.info(
                "当前模型 %s 不支持原生图像输入，媒体已降级为文字占位 + 工具提示",
                agent.get_model(),
            )
            messages = degrade_media_parts_for_text_only(messages)
        return messages

    def _compact_history_messages_only(self, agent: Any, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return copy.deepcopy(
            [
                message
                for message in messages
                if message.get("role") in {"user", "assistant"}
                or (
                    message.get("role") == "system"
                    and str(message.get("content", "")).startswith(CORE_TIMESTAMP_SYSTEM_PREFIXES)
                )
            ]
        )

    def _preserved_state_messages(self, agent: Any, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized = core_sanitize_messages_for_api(messages or [])
        return [
            copy.deepcopy(message)
            for message in sanitized
            if message.get("role") == "system"
            and str(message.get("content", "")).startswith("【任务状态检查点】")
        ]

    def _save_memory_log(self, agent: Any, role: str, content: str = "", reasoning_content: str = ""):
        core_save_memory_log(
            enabled=bool(_cfg(agent, "logging.memory_logging", False)),
            role=role,
            content=content,
            reasoning_content=reasoning_content,
            logger=logger,
        )

    # ═══════════════════════ 请求与重试 ═══════════════════════

    def _save_api_request(self, agent: Any, model: str, messages: List[Dict], tools: Optional[List[Dict]] = None):
        core_save_api_request(
            enabled=bool(_cfg(agent, "logging.api_request_logging", True))
            and bool(_cfg(agent, "logging.keep_raw_trace_log", True)),
            model=model,
            messages=messages,
            tools=tools,
            context_dir=TRACE_LOG_DIR,
            logger=logger,
            max_files=20,
            use_dated_subdir=True,
        )
        try:
            if agent.context_manager and agent.context_manager.token_counter:
                agent._last_request_estimated_tokens = agent.context_manager.token_counter.estimate_total_tokens(
                    messages, tools
                )
        except Exception:
            pass

    def _handle_interrupt(self, agent: Any, signum, frame):
        core_handle_interrupt(agent, interrupted_exception_cls=InterruptedException)

    def _setup_signal_handler(self, agent: Any):
        core_setup_signal_handler(agent, signal_handler=agent._handle_interrupt)

    def _restore_signal_handler(self, agent: Any):
        core_restore_signal_handler(agent)

    def _interruptible_sleep(self, agent: Any, seconds: float):
        core_interruptible_sleep(
            is_interrupted=lambda: agent.interrupted,
            seconds=seconds,
            interrupted_exception_cls=InterruptedException,
        )

    def _retry_request(self, agent: Any, func, *args, **kwargs):
        return core_retry_request(
            func,
            *args,
            max_attempts=int(_cfg(agent, "llm.retry_attempts", 3)),
            retry_delay=int(_cfg(agent, "llm.network_retry_delay", 2)),
            interrupted_exception_cls=InterruptedException,
            logger=logger,
            is_interrupted=lambda: agent.interrupted,
            **kwargs,
        )

    def _parse_arguments(self, agent: Any, arguments_str: str) -> Dict:
        if not arguments_str or arguments_str.strip() == "":
            return {}
        fixed_json = agent._validate_and_fix_json(arguments_str)
        if fixed_json is None:
            raise json.JSONDecodeError("无法修复不完整的JSON", arguments_str, 0)
        return json.loads(fixed_json)

    def _add_tool_message(self, agent: Any, messages: List[Dict], tool_call_id: str, content: str):
        tool_message = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        agent._append_conversation_message(messages, tool_message)

    def _handle_tool_error(self, agent: Any, messages: List[Dict], tool_call_id: str, error: Exception, error_type: str):
        logger.error(f"{error_type}失败: {error}")
        print(f"错误: {error}")
        agent._add_tool_message(messages, tool_call_id, f"{error_type}错误: {error}")

    # ═══════════════════════ 上下文管理 ═══════════════════════

    def _get_context_token_info(
        self,
        agent: Any,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_message: Optional[str] = None,
    ) -> str:
        resolved_messages = messages if messages is not None else agent.current_context.copy()
        resolved_tools = tools if tools is not None else agent._get_current_tools()
        if system_message is None:
            system_message = agent._get_available_tools_message() if messages is None else ""

        return core_format_context_token_info(
            messages=resolved_messages,
            system_message=system_message,
            current_tools=resolved_tools,
            context_manager=agent.context_manager,
            logger=logger,
        )

    def _ensure_context_size(self, agent: Any, messages: List[Dict], tools: List[Dict]) -> bool:
        agent._cache_live_context_status(messages, tools)
        trimmed = core_ensure_context_size(
            messages=messages,
            tools=tools,
            context_manager=agent.context_manager,
            auto_trim_context_fn=agent._auto_trim_context,
            logger=logger,
        )
        agent._cache_live_context_status(messages, tools)
        return trimmed

    def _auto_trim_context(self, agent: Any, messages: List[Dict], tools: List[Dict]):
        core_auto_trim_context(
            messages=messages,
            tools=tools,
            context_manager=agent.context_manager,
            protected_conversation_rounds=int(_cfg(agent, "context.protected_conversation_rounds", 3)),
            extract_tool_call_id_fn=agent._extract_tool_call_id,
            save_context_summary_fn=agent._save_context_summary_to_memory,
            compress_tool_messages_fn=agent._compress_tool_messages_in_place,
            inject_recent_memory_summary_fn=agent._inject_recent_memory_summary,
            emergency_context_clear_fn=agent._emergency_context_clear,
            logger=logger,
        )
        agent._cache_live_context_status(messages, tools)

    def _save_context_summary_to_memory(self, agent: Any, messages: List[Dict]):
        try:
            dialog_messages, last_user_msg = core_collect_summary_source_messages(
                messages,
                recent_limit=int(_cfg(agent, "summary.recent_message_limit", 60)),
            )
            if not dialog_messages:
                return

            latest_path = agent._get_project_memory_path()
            try:
                latest_path.unlink(missing_ok=True)
            except Exception as cleanup_error:
                logger.warning("清理旧任务状态检查点失败: %s", cleanup_error)

            previous_checkpoint = core_extract_current_checkpoint(messages)
            logger.info("正在生成任务状态检查点...")
            smart_summary = agent._generate_smart_summary(
                dialog_messages,
                previous_checkpoint=previous_checkpoint,
            )
            if not smart_summary:
                return

            summary_text = core_build_project_memory_text(
                smart_summary=smart_summary,
                last_user_message=last_user_msg,
            )
            latest_path, snapshot_path = core_write_project_memory_files(
                summary_text=summary_text,
                project_memory_filename=str(_cfg(agent, "memory.project_memory_filename", "project_context_latest.txt")),
                summary_dir=agent._get_project_memory_dir(),
            )
            logger.info("任务状态检查点已保存: %s, %s", latest_path, snapshot_path)
        except Exception as e:
            logger.error("保存任务状态检查点失败: %s", e)

    def _inject_recent_memory_summary(self, agent: Any, messages: List[Dict], include_user_question: bool = True):
        core_inject_recent_memory_summary(
            messages=messages,
            project_memory_filename=str(_cfg(agent, "memory.project_memory_filename", "project_context_latest.txt")),
            summary_injection_max_chars=int(_cfg(agent, "summary.injection_max_chars", 4000)),
            include_user_question=include_user_question,
            summary_dir=agent._get_project_memory_dir(),
            logger=logger,
        )

    def _get_project_memory_dir(self, agent: Any) -> Path:
        return core_get_project_memory_dir()

    def _get_project_memory_path(self, agent: Any) -> Path:
        return core_get_project_memory_path(
            project_memory_filename=str(_cfg(agent, "memory.project_memory_filename", "project_context_latest.txt")),
            summary_dir=agent._get_project_memory_dir(),
        )

    def _cleanup_old_summaries_if_healthy(self, agent: Any, messages: List[Dict], tools: List[Dict]):
        core_cleanup_old_summaries_if_healthy(
            messages=messages,
            tools=tools,
            context_manager=agent.context_manager,
            project_memory_snapshot_prefix=str(_cfg(agent, "memory.project_memory_snapshot_prefix", "project_context_snapshot_")),
            summary_dir=agent._get_project_memory_dir(),
            logger=logger,
        )

    def _generate_smart_summary(self, agent: Any, messages: List[Dict], previous_checkpoint: str = "") -> str:
        from xenon_core.providers import resolve_client_class

        s = agent._settings()
        summary_thinking = bool(s.get("summary.thinking_enabled", False))
        return core_generate_smart_summary(
            messages=messages,
            project_memory_filename=str(s.get("memory.project_memory_filename", "project_context_latest.txt")),
            summary_dir=agent._get_project_memory_dir(),
            recent_message_limit=int(s.get("summary.recent_message_limit", 60)),
            max_message_snippet=int(s.get("summary.max_message_snippet", 1200)),
            max_tool_snippet=int(s.get("summary.max_tool_snippet", 800)),
            summarize_tool_payload_fn=agent._summarize_tool_payload_for_context,
            openai_client_cls=resolve_client_class(str(s.get("llm.provider", "openai_compat"))),
            api_key=s.get("llm.api_key", ""),
            base_url=s.get("llm.base_url", "https://api.deepseek.com"),
            summary_timeout=int(s.get("summary.timeout", 45)),
            summary_model=s.get("summary.model") or s.get("llm.model", "deepseek-v4-flash"),
            summary_max_tokens=int(s.get("summary.max_tokens", 1200)),
            summary_thinking_enabled=summary_thinking,
            summary_reasoning_effort=(
                normalize_reasoning_effort(
                    s.get("llm.reasoning_effort", "max"),
                    str(s.get("llm.base_url", "")),
                ) if summary_thinking else None
            ),
            include_previous_summary=False,
            previous_summary_override=previous_checkpoint,
            logger=logger,
        )

    def _cache_live_context_status(self, agent: Any, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        status = core_get_actual_context_status(
            messages=messages,
            system_message="",
            current_tools=tools,
            context_manager=agent.context_manager,
        )
        agent._last_live_context_status = status
        return status

    def _cache_streaming_context_status(
        self,
        agent: Any,
        messages: List[Dict],
        tools: List[Dict],
        content: str,
        reasoning_content: str = "",
    ) -> Optional[Dict[str, Any]]:
        if not content and not reasoning_content:
            return None
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        return agent._cache_live_context_status(messages + [assistant_message], tools)

    def _get_actual_context_status(
        self,
        agent: Any,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_messages = messages if messages is not None else agent.current_context
        resolved_tools = tools if tools is not None else agent._get_current_tools()
        if system_message is None:
            system_message = agent._get_available_tools_message() if messages is None else ""
        return core_get_actual_context_status(
            messages=resolved_messages,
            system_message=system_message,
            current_tools=resolved_tools,
            context_manager=agent.context_manager,
        )

    def _get_actual_token_estimate(self, agent: Any) -> Dict[str, Any]:
        return agent._get_actual_context_status()

    def _get_current_tools(self, agent: Any) -> List[Dict]:
        return [
            tool
            for tool in core_build_current_tools(
                load_module_tool=agent.load_module_tool,
                tool_description_tool=agent.tool_description_tool,
                get_module_tools_tool=agent.get_module_tools_tool,
                get_module_list_tool=agent.get_module_list_tool,
                loaded_modules=agent.loaded_modules,
                loaded_single_tools=agent.loaded_single_tools,
            )
            if tool is not None
        ]

    def _get_current_tool_names(self, agent: Any) -> List[str]:
        names = []
        for tool in agent._get_current_tools():
            func = tool.get("function", {})
            name = func.get("name")
            if name:
                names.append(name)
        return names

    def _get_current_tool_schemas(self, agent: Any) -> List[Dict]:
        return agent._get_current_tools()

    def _calculate_actual_tokens(self, agent: Any) -> int:
        return core_calculate_actual_tokens(
            messages=agent.current_context,
            current_tools=agent._get_current_tools(),
            context_manager=agent.context_manager,
        )

    def _do_context_cleanup(self, agent: Any, arguments: Dict, messages: List[Dict]) -> Dict[str, Any]:
        return core_do_context_cleanup(
            arguments=arguments,
            messages=messages,
            current_context=agent.current_context,
            context_manager=agent.context_manager,
        )

    # ═══════════════════════ 语义路由与编排 ═══════════════════════

    def _build_semantic_router_catalog(self, agent: Any, tool_schemas: List[Dict[str, Any]]) -> str:
        return core_build_semantic_router_catalog(
            tool_schemas=tool_schemas,
            module_names=agent.tool_manager.get_module_list(),
        )

    def _parse_semantic_route_response(self, agent: Any, content: str) -> Optional[Dict[str, Any]]:
        return core_parse_semantic_route_response(content)

    def _infer_semantic_route(
        self,
        agent: Any,
        user_input: str,
        tool_schemas: List[Dict[str, Any]],
        current_task: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        s = agent._settings()
        router_model = s.get("router.model") or s.get("llm.model", "deepseek-v4-flash")
        router_thinking = bool(s.get("router.thinking_enabled", False))
        return core_infer_semantic_route(
            user_input=user_input,
            tool_schemas=tool_schemas,
            current_task=current_task,
            get_module_list_fn=agent.tool_manager.get_module_list,
            routing_client=agent.routing_client,
            router_model=router_model,
            router_max_tokens=int(s.get("router.max_tokens", 450)),
            router_thinking_enabled=router_thinking,
            router_reasoning_effort=(
                normalize_reasoning_effort(
                    s.get("llm.reasoning_effort", "max"),
                    str(s.get("llm.base_url", "")),
                ) if router_thinking else None
            ),
            logger=logger,
        )

    def _prepare_orchestration_decision(
        self,
        agent: Any,
        user_input: str,
        internal_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ActionDecision]:
        return core_prepare_orchestration_decision(
            user_input=user_input,
            internal_context=internal_context,
            context_manager=agent.context_manager,
            last_tool_result=agent.last_tool_result,
            get_actual_context_status_fn=agent._get_actual_context_status,
            get_current_tool_schemas_fn=agent._get_current_tool_schemas,
            get_cognitive_network_summary_fn=agent._get_cognitive_network_summary,
            get_recent_failures_fn=agent._get_recent_failures,
            agent_orchestrator=agent.agent_orchestrator,
            task_chain_manager=agent.task_chain_manager,
            execution_journal=agent.execution_journal,
            set_orchestration_decision_fn=lambda decision: setattr(agent, "orchestration_decision", decision),
            logger=logger,
            get_semantic_route_hint_fn=lambda user_input, tool_schemas, current_task: (
                agent._infer_semantic_route(
                    user_input=user_input,
                    tool_schemas=tool_schemas,
                    current_task=current_task,
                )
            ),
        )

    def _record_tool_outcome(
        self,
        agent: Any,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any = None,
        success: bool = True,
        recovery_plan: Optional[Dict[str, Any]] = None,
    ):
        decision = agent.orchestration_decision
        outcome_state = build_tool_outcome_state(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=success,
            recovery_plan=recovery_plan,
            phase=decision.phase if decision else "analyze",
            goal=decision.goal if decision else "",
            orchestration_mode=decision.mode if decision else None,
            orchestration_next_actions=decision.next_actions if decision else None,
            orchestration_reasoning_summary=decision.reasoning_summary if decision else "",
            stringify_result_fn=agent._safe_stringify_result,
            summarize_payload_fn=agent._summarize_tool_payload_for_context,
        )
        agent.last_tool_result = outcome_state["last_tool_result"]
        agent.last_recovery_plan = outcome_state["last_recovery_plan"]
        apply_tool_outcome_state(
            outcome_state,
            execution_journal=agent.execution_journal,
            task_chain_manager=agent.task_chain_manager,
            memory_manager=agent.memory_manager,
            logger=logger,
        )

    def _safe_stringify_result(self, agent: Any, value: Any, max_chars: int = 1500) -> str:
        return safe_stringify_result(value, max_chars=max_chars)

    def _build_tool_call_snapshot(
        self,
        agent: Any,
        tool_name: str,
        arguments: Dict[str, Any],
        success: bool,
        result: Any,
        error: str = "",
        recovery_summary: str = "",
    ) -> Dict[str, Any]:
        return build_core_tool_call_snapshot(
            tool_name=tool_name,
            arguments=arguments,
            success=success,
            result=result,
            phase=agent.orchestration_decision.phase if agent.orchestration_decision else "analyze",
            summarize_payload_fn=agent._summarize_tool_payload_for_context,
            error=error,
            recovery_summary=recovery_summary,
        )

    def _push_recent_tool_result(self, agent: Any, snapshot: Dict[str, Any]):
        agent._recent_tool_results = append_recent_tool_result(
            agent._recent_tool_results,
            snapshot,
            limit=agent._recent_tool_result_limit,
        )

    def _get_recent_tool_results(self, agent: Any, limit: int = 5) -> List[Dict[str, Any]]:
        return get_recent_tool_results(agent._recent_tool_results, limit=limit)

    # ═══════════════════════ 工具处理 ═══════════════════════

    def _reset_loaded_tools_for_new_turn(self, agent: Any):
        return core_reset_loaded_tools_for_new_turn(
            loaded_modules=agent.loaded_modules,
            loaded_single_tools=agent.loaded_single_tools,
            approved_tools=agent.approved_tools,
        )

    def _reset_loaded_tools_for_new_turn_without_notice(self, agent: Any) -> Optional[str]:
        agent._reset_loaded_tools_for_new_turn()
        return None

    def _handle_load_module(self, agent: Any, tool_call_id: str, arguments_str: str, messages: list) -> None:
        core_handle_load_module_call(
            tool_call_id=tool_call_id,
            arguments_str=arguments_str,
            messages=messages,
            parse_arguments_fn=agent._parse_arguments,
            get_tool_list_fn=agent.tool_manager.get_tool_list,
            get_tool_schema_by_name_fn=agent.tool_manager.get_tool_schema_by_name,
            add_tool_message_fn=agent._add_tool_message,
            handle_tool_error_fn=agent._handle_tool_error,
            loaded_modules=agent.loaded_modules,
            loaded_single_tools=agent.loaded_single_tools,
            approved_tools=agent.approved_tools,
            max_loaded_modules=agent._max_loaded_modules,
            logger=logger,
            print_fn=print,
            alias_map=agent.tool_manager.module_aliases,
        )

    def _authorize_single_tool(self, agent: Any, tool_name: str, schema: dict) -> None:
        core_authorize_single_tool(
            loaded_single_tools=agent.loaded_single_tools,
            approved_tools=agent.approved_tools,
            tool_name=tool_name,
            schema=schema,
        )

    def _is_tool_loaded(self, agent: Any, tool_name: str) -> bool:
        return core_is_tool_loaded(agent.loaded_modules, tool_name)

    def _is_single_tool_loaded(self, agent: Any, tool_name: str) -> bool:
        return core_is_single_tool_loaded(agent.loaded_single_tools, tool_name)

    def _touch_single_tool(self, agent: Any, tool_name: str) -> None:
        core_touch_single_tool(agent.loaded_single_tools, tool_name)

    def _get_or_create_manual_context_manager_tool(self, agent: Any):
        return agent.context_manager

    def _decay_recent_tool_results(self, agent: Any, user_input: str):
        if not agent._recent_tool_results:
            return
        previous_user_input = ""
        for message in reversed(agent.current_context):
            if message.get("role") == "user":
                previous_user_input = str(message.get("content", "")).strip()
                break
        agent._recent_tool_results = decay_recent_tool_results(
            agent._recent_tool_results,
            previous_user_input=previous_user_input,
            user_input=user_input,
            similarity_fn=agent._text_similarity,
            same_topic_keep=agent._recent_tool_result_same_topic_keep,
            topic_shift_keep=agent._recent_tool_result_topic_shift_keep,
            limit=agent._recent_tool_result_limit,
        )

    def _text_similarity(self, agent: Any, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        return SequenceMatcher(None, left, right).ratio()

    def _get_recent_failures(self, agent: Any, limit: int = 3) -> List[str]:
        current_task_wrapper = agent.task_chain_manager.get_current_task()
        current_task = current_task_wrapper["task"] if current_task_wrapper else None
        execution_state = (current_task or {}).get("execution_state", {}) or {}
        blockage_reason = execution_state.get("blockage_reason")
        return collect_recent_failures(
            agent._recent_tool_results,
            blockage_reason=blockage_reason,
            limit=limit,
        )

    def _handle_tool_calls(self, agent: Any, tool_calls, messages: List[Dict], tools: List[Dict] = None):
        core_handle_tool_call_batch(
            tool_calls=tool_calls,
            messages=messages,
            tools=tools,
            interrupted_exception_cls=InterruptedException,
            is_interrupted_fn=lambda: agent.interrupted,
            handle_load_module_fn=agent._handle_load_module,
            handle_get_tool_description_fn=agent._handle_get_tool_description,
            handle_get_module_tools_fn=agent._handle_get_module_tools,
            handle_get_module_list_fn=agent._handle_get_module_list,
            handle_context_manager_tool_fn=agent._handle_context_manager_tool,
            handle_execute_tool_fn=agent._handle_execute_tool,
            add_tool_message_fn=agent._add_tool_message,
            is_tool_loaded_fn=agent._is_tool_loaded,
            is_single_tool_loaded_fn=agent._is_single_tool_loaded,
            logger=logger,
        )

    def _handle_get_tool_description(self, agent: Any, tool_call_id: str, arguments_str: str, messages: List[Dict]):
        core_handle_get_tool_description_call(
            tool_call_id=tool_call_id,
            arguments_str=arguments_str,
            messages=messages,
            parse_arguments_fn=agent._parse_arguments,
            get_tool_schema_by_name_fn=agent.tool_manager.get_tool_schema_by_name,
            authorize_single_tool_fn=agent._authorize_single_tool,
            add_tool_message_fn=agent._add_tool_message,
            handle_tool_error_fn=agent._handle_tool_error,
            logger=logger,
        )

    def _handle_get_module_tools(self, agent: Any, tool_call_id: str, arguments_str: str, messages: List[Dict]):
        core_handle_get_module_tools_call(
            tool_call_id=tool_call_id,
            arguments_str=arguments_str,
            messages=messages,
            parse_arguments_fn=agent._parse_arguments,
            get_tool_list_fn=agent.tool_manager.get_tool_list,
            add_tool_message_fn=agent._add_tool_message,
            handle_tool_error_fn=agent._handle_tool_error,
            logger=logger,
        )

    def _handle_get_module_list(self, agent: Any, tool_call_id: str, arguments_str: str, messages: List[Dict]):
        core_handle_get_module_list_call(
            tool_call_id=tool_call_id,
            arguments_str=arguments_str,
            messages=messages,
            parse_arguments_fn=agent._parse_arguments,
            get_module_list_fn=agent.tool_manager.get_module_list,
            add_tool_message_fn=agent._add_tool_message,
            handle_tool_error_fn=agent._handle_tool_error,
            logger=logger,
        )

    def _handle_context_manager_tool(self, agent: Any, tool_call_id: str, tool_name: str, arguments_str: str, messages: List[Dict], tools: List[Dict] = None):
        core_handle_context_manager_tool_call(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments_str=arguments_str,
            messages=messages,
            tools=tools,
            parse_arguments_fn=agent._parse_arguments,
            find_pending_tool_call_ids_fn=agent._find_pending_tool_call_ids,
            add_tool_message_fn=agent._add_tool_message,
            get_actual_context_status_fn=agent._get_actual_context_status,
            get_actual_token_estimate_fn=agent._get_actual_token_estimate,
            get_current_tools_fn=agent._get_current_tools,
            auto_trim_context_fn=agent._auto_trim_context,
            extract_tool_call_id_fn=agent._extract_tool_call_id,
            append_conversation_message_fn=agent._append_conversation_message,
            get_or_create_context_manager_tool_fn=agent._get_or_create_manual_context_manager_tool,
            handle_tool_error_fn=agent._handle_tool_error,
            logger=logger,
        )

    def _handle_execute_tool(self, agent: Any, tool_call_id: str, tool_name: str, arguments_str: str, messages: List[Dict]):
        core_execute_tool_call(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments_str=arguments_str,
            messages=messages,
            parse_arguments_fn=agent._parse_arguments,
            execute_tool_fn=agent.tool_manager.execute_tool,
            record_tool_outcome_fn=agent._record_tool_outcome,
            build_tool_call_snapshot_fn=agent._build_tool_call_snapshot,
            push_recent_tool_result_fn=agent._push_recent_tool_result,
            monitor_tool_result_snapshot_fn=lambda snapshot: None,
            add_tool_message_fn=agent._add_tool_message,
            handle_tool_error_fn=agent._handle_tool_error,
            build_recovery_plan_fn=agent.recovery_manager.build_recovery_plan,
            touch_single_tool_fn=agent._touch_single_tool,
            loaded_modules=agent.loaded_modules,
            set_tool_executing_fn=lambda value: setattr(agent, "_tool_executing", value),
            stream_callback=agent._stream_callback,
            current_phase=agent.orchestration_decision.phase if agent.orchestration_decision else None,
            logger=logger,
        )

    # ═══════════════════════ 对话流 ═══════════════════════

    def chat(self, agent: Any, user_input: str, attachments: Optional[List[str]] = None):
        # 原生多模态（Phase 1）：排队路径把附件转为 @token 随文本入队，
        # 复用 chat_entry 的统一解析；主路径直接透传附件列表。
        queued_input = (
            append_attachment_tokens(user_input, attachments)
            if attachments
            else user_input
        )
        if getattr(agent, "_autonomous_running", False):
            agent.interrupted = True
            return agent.queue_pending_user_input(queued_input)

        if getattr(agent, "_turn_running", False):
            result = agent.queue_pending_user_input(queued_input)
            if agent._stream_callback:
                agent._stream_callback({
                    "type": "user_queued",
                    "content": user_input,
                    "queue_position": result.get("pending_count", 0),
                })
            return result

        reset_loaded_tools_fn = agent._reset_loaded_tools_for_new_turn
        if _cfg(agent, "context.compact_after_turn", True):
            reset_loaded_tools_fn = agent._reset_loaded_tools_for_new_turn_without_notice

        core_handle_user_chat_entry(
            user_input=user_input,
            attachments=attachments,
            current_context=agent.current_context,
            decay_recent_tool_results_fn=agent._decay_recent_tool_results,
            set_interrupted_fn=lambda value: setattr(agent, "interrupted", value),
            set_active_user_input_fn=lambda value: setattr(agent, "_active_user_input", value),
            reset_loaded_tools_for_new_turn_fn=reset_loaded_tools_fn,
            find_pending_tool_call_ids_fn=agent._find_pending_tool_call_ids,
            append_conversation_message_fn=agent._append_conversation_message,
            cleanup_reasoning_content_fn=agent._cleanup_reasoning_content_for_next_request,
            process_chat_with_context_fn=agent._process_chat_with_context,
            logger=logger,
        )

    def _process_chat_with_context(self, agent: Any, user_input: str, internal_context: Optional[Dict[str, Any]] = None):
        agent._active_user_input = user_input
        agent._turn_running = True
        agent._tool_cycle_count = 0
        agent._recursion_detector.reset()
        try:
            agent._run_single_turn(user_input, internal_context)
        finally:
            agent._turn_running = False

        while agent._pending_user_inputs:
            pending = agent._pending_user_inputs.pop(0)
            agent._turn_running = True
            try:
                if agent._stream_callback:
                    agent._stream_callback({
                        "type": "queue_processing",
                        "content": pending,
                        "queue_remaining": len(agent._pending_user_inputs),
                    })
                agent._decay_recent_tool_results(pending)
                agent._active_user_input = pending
                reset_loaded_tools_fn = agent._reset_loaded_tools_for_new_turn
                if _cfg(agent, "context.compact_after_turn", True):
                    reset_loaded_tools_fn = agent._reset_loaded_tools_for_new_turn_without_notice

                core_handle_user_chat_entry(
                    user_input=pending,
                    current_context=agent.current_context,
                    decay_recent_tool_results_fn=lambda _: None,
                    set_interrupted_fn=lambda value: setattr(agent, "interrupted", value),
                    set_active_user_input_fn=lambda value: setattr(agent, "_active_user_input", value),
                    reset_loaded_tools_for_new_turn_fn=reset_loaded_tools_fn,
                    find_pending_tool_call_ids_fn=agent._find_pending_tool_call_ids,
                    append_conversation_message_fn=agent._append_conversation_message,
                    cleanup_reasoning_content_fn=agent._cleanup_reasoning_content_for_next_request,
                    process_chat_with_context_fn=agent._run_single_turn,
                    logger=logger,
                )
            finally:
                agent._turn_running = False

    def _run_single_turn(self, agent: Any, user_input: str, internal_context: Optional[Dict[str, Any]] = None):
        core_run_chat_turn(
            user_input=user_input,
            internal_context=internal_context,
            current_context=agent.current_context,
            context_manager=agent.context_manager,
            prepare_orchestration_decision_fn=agent._prepare_orchestration_decision,
            get_actual_context_status_fn=agent._get_actual_context_status,
            get_current_tool_names_fn=agent._get_current_tool_names,
            get_static_system_prompt_fn=agent._get_system_prompt,
            get_context_token_info_fn=agent._get_context_token_info,
            get_current_tools_fn=agent._get_current_tools,
            ensure_context_size_fn=agent._ensure_context_size,
            chat_fn=agent._chat,
            cleanup_old_summaries_if_healthy_fn=agent._cleanup_old_summaries_if_healthy,
            save_memory_log_fn=agent._save_memory_log,
            set_current_context_fn=lambda context: setattr(agent, "current_context", context),
            interrupted_exception_cls=InterruptedException,
            model_for_chat=agent.get_model(),
            logger=logger,
            print_fn=print,
            reset_loaded_tools_for_next_turn_fn=agent._reset_loaded_tools_for_new_turn,
            compact_turn_after_commit_fn=(
                agent._compact_turn_for_next_context if _cfg(agent, "context.compact_after_turn", True) else None
            ),
            stream_callback_fn=agent._stream_callback,
        )

    def _chat(self, agent: Any, messages: List[Dict], tools: List[Dict], model: str, _retry_count: int = 0):
        if _retry_count == 0 and getattr(agent, "_turn_running", False) and agent._pending_user_inputs:
            pending = agent._pending_user_inputs.pop(0)
            # 原生多模态（Phase 1）：排队消息中的 @附件引用在此统一解析为 content parts
            agent._append_conversation_message(
                messages,
                {"role": "user", "content": build_user_content_safely(pending)},
            )
            agent._active_user_input = pending
            if agent._stream_callback:
                agent._stream_callback({
                    "type": "queue_processing",
                    "content": pending,
                    "queue_remaining": len(agent._pending_user_inputs),
                })

        core_run_chat_cycle(
            messages=messages,
            tools=tools,
            model=model,
            retry_count=_retry_count,
            max_trim_retries=2,
            context_manager=agent.context_manager,
            enable_streaming=bool(agent._settings().get("llm.streaming_enabled", True)),
            ensure_message_integrity_fn=agent._ensure_message_integrity,
            auto_trim_context_fn=agent._auto_trim_context,
            emergency_context_clear_fn=agent._emergency_context_clear,
            append_conversation_message_fn=agent._append_conversation_message,
            save_api_request_fn=agent._save_api_request,
            save_api_usage_fn=lambda tokens: setattr(agent, '_last_api_total_tokens', tokens),
            retry_request_fn=agent._retry_request,
            create_completion_fn=agent.client.chat.completions.create,
            process_streaming_response_fn=agent._process_streaming_response,
            process_non_streaming_response_fn=agent._process_non_streaming_response,
            recursive_chat_fn=lambda next_messages, next_tools, next_model, next_retry_count: agent._chat(
                next_messages,
                next_tools,
                next_model,
                next_retry_count,
            ),
            set_in_api_call_fn=lambda value: setattr(agent, "_in_api_call", value),
            is_interrupted_fn=lambda: agent.interrupted,
            interrupted_exception_cls=InterruptedException,
            logger=logger,
            thinking_enabled=bool(agent._settings().get("llm.thinking_enabled", True)),
            reasoning_effort=normalize_reasoning_effort(
                agent._settings().get("llm.reasoning_effort", "max"),
                str(getattr(agent.client, "base_url", "")),
            ),
            base_url=str(getattr(agent.client, "base_url", "")),
            thinking_mode=str(agent._settings().get("llm.thinking_mode", "auto") or "auto"),
            print_fn=print,
            stream_error_fn=lambda message: (
                agent._stream_callback({"type": "error", "content": message})
                if getattr(agent, "_stream_callback", None)
                else None
            ),
            # 严格模板供应商学习机制：首次折叠重试成功后标记，后续轮次预折叠，
            # 避免每轮都先在供应商侧留下一条 500 错误日志。
            pre_fold_system_messages=bool(getattr(agent, "_strict_template_provider", False)),
            on_strict_template_retry_fn=lambda: (
                setattr(agent, "_strict_template_provider", True),
                logger.info("已标记当前供应商为严格模板，后续轮次将预折叠 system 消息"),
            ),
            apply_media_policy_fn=lambda api_messages: self._apply_media_policy(agent, api_messages),
        )

    def _extract_tool_call_id(self, agent: Any, tc) -> Optional[str]:
        return core_extract_tool_call_id(tc)

    def _find_pending_tool_call_ids(self, agent: Any, messages: List[Dict], exclude_id: str = None) -> set:
        return core_find_pending_tool_call_ids(messages, exclude_id=exclude_id)

    def _cleanup_reasoning_content(self, agent: Any, messages: List[Dict]):
        core_cleanup_reasoning_content(messages)

    def _cleanup_reasoning_content_for_next_request(self, agent: Any, messages: List[Dict]):
        if bool(agent._settings().get("llm.thinking_enabled", True)):
            return None
        return core_cleanup_reasoning_content(messages)

    def _emergency_context_clear(self, agent: Any, messages: List[Dict], include_user_question: bool = False):
        core_emergency_context_clear(
            messages=messages,
            include_user_question=include_user_question,
            save_context_summary_fn=agent._save_context_summary_to_memory,
            inject_recent_memory_summary_fn=agent._inject_recent_memory_summary,
            extract_tool_call_id_fn=agent._extract_tool_call_id,
            logger=logger,
        )

    def _try_inject_queued_messages(self, agent: Any, messages: List[Dict]):
        if not getattr(agent, "_turn_running", False):
            return False
        import time as _time

        for _ in range(20):
            if agent._pending_user_inputs:
                break
            _time.sleep(0.01)
        if not agent._pending_user_inputs:
            return False
        pending = agent._pending_user_inputs.pop(0)
        # 原生多模态（Phase 1）：排队消息中的 @附件引用在此统一解析为 content parts
        agent._append_conversation_message(
            messages,
            {"role": "user", "content": build_user_content_safely(pending)},
        )
        agent._active_user_input = pending
        print(f"\n[排队注入] 中途将排队消息注入到对话中: {pending[:80]}...", file=sys.stderr)
        if agent._stream_callback:
            agent._stream_callback({
                "type": "queue_processing",
                "content": pending,
                "queue_remaining": len(agent._pending_user_inputs),
            })
        return True

    def _process_streaming_response(self, agent: Any, response, messages: List[Dict], tools: List[Dict]):
        max_depth = int(_cfg(agent, "tools.max_tool_recursion_depth", 100))

        def continue_with_injection(next_messages, next_tools, next_model):
            agent._tool_cycle_count += 1
            if agent._tool_cycle_count > max_depth:
                logger.warning("[loop_guard] 工具递归达到上限 (%s)，注入提醒继续执行", max_depth)
                agent._append_conversation_message(next_messages, {
                    # Gemini 等供应商要求请求最后一条消息不能是 assistant(model) 轮次，必须以 user 身份注入
                    "role": "user",
                    "content": f"Xenon{max_depth}次调用了，工作做完了吗？",
                })
            if agent._recursion_detector.check_and_inject(
                next_messages,
                append_message_fn=next_messages.append,
            ):
                logger.warning("[recursion_detector] 检测到递归死循环，强制中断工具链")
                return
            agent._try_inject_queued_messages(next_messages)
            agent._chat(next_messages, next_tools, next_model)

        streaming_content_parts: List[str] = []
        streaming_reasoning_parts: List[str] = []
        streaming_chars_since_update = 0
        streaming_last_update = 0.0

        def stream_callback_with_usage(event: Dict[str, Any]):
            nonlocal streaming_chars_since_update, streaming_last_update
            if agent._stream_callback:
                agent._stream_callback(event)

            event_type = event.get("type")
            chunk = str(event.get("content", "") or "")
            if event_type == "content" and chunk:
                streaming_content_parts.append(chunk)
            elif event_type == "thinking" and chunk:
                streaming_reasoning_parts.append(chunk)
            else:
                return

            import time as _time

            streaming_chars_since_update += len(chunk)
            now = _time.monotonic()
            if streaming_chars_since_update < 256 and now - streaming_last_update < 0.75:
                return

            agent._cache_streaming_context_status(
                messages,
                tools,
                "".join(streaming_content_parts),
                "".join(streaming_reasoning_parts),
            )
            streaming_chars_since_update = 0
            streaming_last_update = now

        core_process_streaming_response(
            response=response,
            messages=messages,
            tools=tools,
            interrupted_exception_cls=InterruptedException,
            is_interrupted_fn=lambda: agent.interrupted,
            stream_callback=stream_callback_with_usage,
            validate_and_fix_json_fn=agent._validate_and_fix_json,
            append_conversation_message_fn=agent._append_conversation_message,
            save_memory_log_fn=agent._save_memory_log,
            handle_tool_calls_fn=agent._handle_tool_calls,
            get_current_tools_fn=agent._get_current_tools,
            continue_chat_fn=continue_with_injection,
            model_for_recursive_chat=agent.get_model(),
            logger=logger,
            print_fn=print,
        )
        agent._cache_live_context_status(messages, tools)

    def _process_non_streaming_response(self, agent: Any, response, messages: List[Dict], tools: List[Dict]):
        max_depth = int(_cfg(agent, "tools.max_tool_recursion_depth", 100))

        def continue_with_injection(next_messages, next_tools, next_model):
            agent._tool_cycle_count += 1
            if agent._tool_cycle_count > max_depth:
                logger.warning("[loop_guard] 工具递归达到上限 (%s)，注入提醒继续执行", max_depth)
                agent._append_conversation_message(next_messages, {
                    # Gemini 等供应商要求请求最后一条消息不能是 assistant(model) 轮次，必须以 user 身份注入
                    "role": "user",
                    "content": f"Xenon{max_depth}次调用了，工作做完了吗？",
                })
            if agent._recursion_detector.check_and_inject(
                next_messages,
                append_message_fn=next_messages.append,
            ):
                logger.warning("[recursion_detector] 检测到递归死循环，强制中断工具链")
                return
            agent._try_inject_queued_messages(next_messages)
            agent._chat(next_messages, next_tools, next_model)

        core_process_non_streaming_response(
            response=response,
            messages=messages,
            tools=tools,
            interrupted_exception_cls=InterruptedException,
            is_interrupted_fn=lambda: agent.interrupted,
            append_conversation_message_fn=agent._append_conversation_message,
            save_memory_log_fn=agent._save_memory_log,
            handle_tool_calls_fn=agent._handle_tool_calls,
            get_current_tools_fn=agent._get_current_tools,
            continue_chat_fn=continue_with_injection,
            model_for_recursive_chat=agent.get_model(),
            logger=logger,
            print_fn=print,
        )
        agent._cache_live_context_status(messages, tools)

    def _validate_and_fix_json(self, agent: Any, json_str: str) -> Optional[str]:
        return core_validate_and_fix_json(json_str, logger=logger)

    def _ensure_message_integrity(self, agent: Any, messages: List[Dict]) -> List[Dict]:
        return core_ensure_message_integrity(messages, logger=logger)

    # ═══════════════════════ 自主执行与多代理 ═══════════════════════

    def _get_phase_memory_snapshot(
        self,
        agent: Any,
        goal: str,
        phase: str,
        intent: str,
        recent_failures: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return core_get_phase_memory_snapshot(
            goal=goal,
            phase=phase,
            intent=intent,
            recent_failures=recent_failures,
            memory_manager=agent.memory_manager,
            get_cognitive_network_summary_fn=agent._get_cognitive_network_summary,
        )

    def _should_resume_task(
        self,
        agent: Any,
        current_task: Optional[Dict[str, Any]],
        replan_suggestion: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return core_should_resume_task(
            current_task=current_task,
            replan_suggestion=replan_suggestion,
            max_tool_failures=agent._autonomous_max_tool_failures,
        )

    def _select_active_goal(
        self,
        agent: Any,
        current_task: Optional[Dict[str, Any]],
        replan_suggestion: Optional[Dict[str, Any]],
        recent_failures: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        return core_select_active_goal(
            current_task=current_task,
            replan_suggestion=replan_suggestion,
            recent_failures=recent_failures,
            pending_user_inputs=agent._pending_user_inputs,
            max_tool_failures=agent._autonomous_max_tool_failures,
        )

    def _build_internal_resume_prompt(
        self,
        agent: Any,
        goal_payload: Dict[str, Any],
        memory_summary: str,
        activation_set: List[Dict[str, Any]],
        self_model: Dict[str, Any],
        recent_failures: Optional[List[str]] = None,
        replan_suggestion: Optional[Dict[str, Any]] = None,
    ) -> str:
        return core_build_internal_resume_prompt(
            goal_payload=goal_payload,
            memory_summary=memory_summary,
            activation_set=activation_set,
            self_model=self_model,
            recent_failures=recent_failures,
            replan_suggestion=replan_suggestion,
        )

    def _build_autonomous_decision(
        self,
        agent: Any,
        current_task: Dict[str, Any],
        goal_payload: Dict[str, Any],
        memory_snapshot: Dict[str, Any],
        self_model: Dict[str, Any],
        replan_suggestion: Optional[Dict[str, Any]],
        recent_failures: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return core_build_autonomous_decision(
            current_task=current_task,
            goal_payload=goal_payload,
            memory_snapshot=memory_snapshot,
            self_model=self_model,
            replan_suggestion=replan_suggestion,
            recent_failures=recent_failures,
        )

    def _update_autonomous_progress(
        self,
        agent: Any,
        previous_task: Dict[str, Any],
        goal_payload: Dict[str, Any],
        internal_prompt: str,
    ) -> Dict[str, Any]:
        return core_update_autonomous_progress(
            previous_task=previous_task,
            goal_payload=goal_payload,
            internal_prompt=internal_prompt,
            task_chain_manager=agent.task_chain_manager,
            recent_tool_results=agent._recent_tool_results,
            max_phase_stagnation=agent._autonomous_max_phase_stagnation,
            max_repeated_actions=agent._autonomous_max_repeated_actions,
            max_tool_failures=agent._autonomous_max_tool_failures,
        )

    def autonomous_tick(self, agent: Any) -> Dict[str, Any]:
        return core_run_autonomous_tick(
            task_chain_manager=agent.task_chain_manager,
            memory_manager=agent.memory_manager,
            current_context=agent.current_context,
            pending_user_inputs=agent._pending_user_inputs,
            get_cognitive_network_summary_fn=agent._get_cognitive_network_summary,
            get_recent_failures_fn=agent._get_recent_failures,
            get_recent_tool_results_fn=agent._get_recent_tool_results,
            cleanup_reasoning_content_fn=agent._cleanup_reasoning_content_for_next_request,
            append_conversation_message_fn=agent._append_conversation_message,
            process_chat_with_context_fn=agent._process_chat_with_context,
            max_phase_stagnation=agent._autonomous_max_phase_stagnation,
            max_repeated_actions=agent._autonomous_max_repeated_actions,
            max_tool_failures=agent._autonomous_max_tool_failures,
            log_autonomous_tick_fn=agent.execution_journal.log_autonomous_tick,
        )

    def queue_pending_user_input(self, agent: Any, user_input: str) -> Dict[str, Any]:
        return core_enqueue_pending_user_input(
            agent._pending_user_inputs,
            user_input,
            limit=agent._pending_user_input_limit,
        )

    def run_autonomous_cycle(self, agent: Any, max_steps: int = 1) -> Dict[str, Any]:
        agent._autonomous_running = True
        try:
            return core_run_autonomous_cycle(
                max_steps=max_steps,
                autonomous_tick_fn=agent.autonomous_tick,
            )
        finally:
            agent._autonomous_running = False

    def plan_multi_agent_subtasks(self, agent: Any, max_subtasks: int = 2) -> Dict[str, Any]:
        current_task_wrapper = agent.task_chain_manager.get_current_task()
        if not current_task_wrapper:
            return {"success": False, "status": "idle", "reason": "no_active_task"}
        run = agent.multi_agent_runtime.create_run(
            current_task_wrapper["task"],
            max_subtasks=max_subtasks,
        )
        agent.multi_agent_runtime.log_plan(agent.execution_journal, run)
        return {"success": True, "status": run.get("status"), "run": run}

    def get_multi_agent_status(self, agent: Any, run_id: Optional[str] = None) -> Dict[str, Any]:
        return agent.multi_agent_runtime.get_status(run_id=run_id)

    def run_multi_agent_cycle(
        self,
        agent: Any,
        max_subtasks: int = 2,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        status = agent.multi_agent_runtime.get_status(run_id=run_id)
        active_run = status.get("active_run")
        if not active_run:
            planned = agent.plan_multi_agent_subtasks(max_subtasks=max_subtasks)
            if not planned.get("success"):
                return planned
            run_id = planned["run"]["run_id"]
        elif run_id is None:
            run_id = active_run.get("run_id")

        execution = agent.multi_agent_runtime.run_pending(
            run_id=run_id,
            max_subtasks=max_subtasks,
            executor_fn=agent._execute_multi_agent_subtask,
        )
        integration = None
        if execution.get("status") in {"completed", "completed_with_failures"}:
            integration = agent.multi_agent_runtime.integrate_run(
                run_id=execution.get("run_id"),
                task_chain_manager=agent.task_chain_manager,
                execution_journal=agent.execution_journal,
            )
        return {
            "success": execution.get("success", False),
            "status": integration.get("status") if integration else execution.get("status"),
            "run_id": execution.get("run_id"),
            "execution": execution,
            "integration": integration,
        }

    def _execute_multi_agent_subtask(self, agent: Any, subtask: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_subagent_prompt(subtask)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an isolated Xenon sub-agent. Keep the result concise, "
                    "structured, and limited to the assigned subtask."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        tools = agent._filter_tools_for_subtask(subtask)
        agent._chat(messages, tools, agent.get_model())
        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        summary = assistant_messages[-1].get("content", "") if assistant_messages else ""
        return {
            "success": bool(summary.strip()),
            "summary": summary.strip() or "Sub-agent finished without an assistant summary.",
            "artifacts": [],
            "conflicts": [],
            "metadata": {
                "isolated_messages": len(messages),
                "allowed_tools": subtask.get("allowed_tools") or [],
            },
        }

    def _filter_tools_for_subtask(self, agent: Any, subtask: Dict[str, Any]) -> List[Dict[str, Any]]:
        allowed = [str(item).lower() for item in (subtask.get("allowed_tools") or [])]
        if not allowed:
            return []
        filtered = []
        for tool in agent._get_current_tools():
            name = str((tool.get("function") or {}).get("name") or "").lower()
            if any(token in name for token in allowed):
                filtered.append(tool)
        return filtered

    # ═══════════════════════ CLI 入口 ═══════════════════════

    def run(self, agent: Any):
        # 惰性取外壳版本号（运行期调用，无循环导入风险）
        from Xenon import APP_VERSION

        run_interactive_agent_session(
            agent,
            project_root=PROJECT_ROOT,
            model=agent.get_model(),
            thinking_enabled=bool(agent._settings().get("llm.thinking_enabled", True)),
            streaming_enabled=bool(agent._settings().get("llm.streaming_enabled", True)),
            interrupted_exception_cls=InterruptedException,
            app_version=APP_VERSION,
        )
