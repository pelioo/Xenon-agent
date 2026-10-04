from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from xenon_core.execution_context import (
    SandboxContext,
    ToolHealthChecker,
    create_default_context,
)
from xenon_core.context import XenonContext
from xenon_core.delivery_closure import GitStatusProbe, DeliveryReport
from xenon_core.multi_agent_runtime import MultiAgentRuntime


# build_core_management_tools 已迁至 xenon_core.tool_catalog（P4：由 tools 插件构建）；
# 此处保留导入以便旧引用与回退路径使用。
from xenon_core.tool_catalog import build_core_management_tools


def initialize_context_manager(
    *,
    context_manager_cls: Any,
    tiktoken_available: bool,
    max_context_tokens_test: Optional[int],
    max_context_tokens_default: int,
    logger: Any,
    print_fn: Callable[..., Any] = print,
) -> Any:
    context_manager = None
    if tiktoken_available:
        try:
            max_tokens = (
                max_context_tokens_test
                if max_context_tokens_test is not None
                else max_context_tokens_default
            )
            context_manager = context_manager_cls(max_context_tokens=max_tokens)
            print_fn(
                f"\033[38;2;111;208;104m[OK] Context manager initialized (limit: {max_tokens} tokens)\033[0m"
            )
        except Exception as error:
            logger.error("上下文管理器初始化失败: %s", error)
            context_manager = None
    else:
        print_fn("\033[93m[WARN] Context manager unavailable because tiktoken is not installed.\033[0m")
    return context_manager


def bootstrap_agent(
    agent: Any,
    *,
    openai_client_cls: Any,
    api_key: str,
    base_url: str,
    api_timeout: int,
    router_timeout: int,
    tool_manager_cls: Any,
    task_chain_manager_cls: Any,
    memory_manager_cls: Any,
    execution_journal_cls: Any,
    recovery_manager_cls: Any,
    agent_orchestrator_cls: Any,
    cognitive_network_cls: Any,
    context_manager_cls: Any,
    tiktoken_available: bool,
    max_context_tokens_test: Optional[int],
    max_context_tokens_default: int,
    get_cognitive_network_summary_fn: Callable[[], str],
    logger: Any,
    history_dir_name: str = ".agent_history",
    context_dir_name: str = "Context",
    memory_dir: str = "Memory/memory_Write",
    print_fn: Callable[..., Any] = print,
) -> None:
    agent.client = openai_client_cls(api_key=api_key, base_url=base_url, timeout=api_timeout)
    agent.routing_client = openai_client_cls(api_key=api_key, base_url=base_url, timeout=router_timeout)

    agent.current_context = []
    agent.full_conversation_history = []
    agent.compact_history = []
    agent.display_history = []  # 完整对话历史（含工具调用、思考过程），专用于 WebUI 显示，与API提交解耦

    agent.history_dir = Path(history_dir_name)
    agent.history_dir.mkdir(parents=True, exist_ok=True)
    agent.history_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # P4：tool_manager 不再在此创建——由 tools 插件在装配阶段构建，
    # 装配完成后回填（见文件末尾 _wire_tools_after_plugins）。
    agent.task_chain_manager = task_chain_manager_cls()
    agent.memory_manager = memory_manager_cls(memory_dir=memory_dir, enable_network=False)
    agent.execution_journal = execution_journal_cls()
    agent.recovery_manager = recovery_manager_cls()
    agent.agent_orchestrator = agent_orchestrator_cls()

    agent.interrupted = False
    agent.approved_tools = set()
    agent.loaded_modules = {}
    agent._max_loaded_modules = 5
    agent.loaded_single_tools = {}

    agent.context_dir = Path(context_dir_name)
    agent.context_dir.mkdir(exist_ok=True)
    agent._original_sigint_handler = None
    agent._in_api_call = False
    agent._tool_executing = False
    agent._tool_call_depth = 0  # 单轮内工具调用递归深度计数器
    agent._pending_user_inputs = []
    agent._pending_user_input_limit = 8
    agent._autonomous_running = False
    agent._turn_running = False
    agent.orchestration_decision = None
    agent.last_tool_result = None
    agent.last_recovery_plan = None

    agent.cognitive_network = cognitive_network_cls()
    agent.cognitive_network_summary = ""
    agent._active_user_input = ""
    agent._recent_tool_results = []
    agent._recent_tool_result_limit = 8
    agent._recent_tool_result_same_topic_keep = 6
    agent._recent_tool_result_topic_shift_keep = 3
    agent._autonomous_max_phase_stagnation = 3
    agent._autonomous_max_repeated_actions = 3
    agent._autonomous_max_tool_failures = 3
    agent.multi_agent_runtime = MultiAgentRuntime()
    agent._multi_agent_default_subtasks = 2

    agent.context_manager = initialize_context_manager(
        context_manager_cls=context_manager_cls,
        tiktoken_available=tiktoken_available,
        max_context_tokens_test=max_context_tokens_test,
        max_context_tokens_default=max_context_tokens_default,
        logger=logger,
        print_fn=print_fn,
    )

    agent.cognitive_network_summary = get_cognitive_network_summary_fn() or ""
    if agent.cognitive_network_summary:
        print_fn("\033[38;2;111;208;104m[OK] Cognitive network initialized from memory sources\033[0m")

    # Phase 4: 统一执行上下文 & 沙箱隔离
    agent.sandbox_context = create_default_context(workspace_root=str(Path.cwd()))
    print_fn("\033[38;2;111;208;104m[OK] Sandbox context initialized (isolation: off)\033[0m")

    # Phase 4: 工具健康检查
    agent.health_checker = ToolHealthChecker()
    health_report = agent.health_checker.check_all()
    print_fn(agent.health_checker.format_report_console())

    # Phase 4: 工具健康检查
    agent.health_checker = ToolHealthChecker()
    health_report = agent.health_checker.check_all()
    print_fn(agent.health_checker.format_report_console())

    # Phase 5: Git 状态感知
    agent.git_probe = GitStatusProbe()
    git_state = agent.git_probe.probe()
    print_fn(agent.git_probe.format_status_console())
    agent.delivery_report = DeliveryReport(agent.git_probe)

    agent._stream_callback = None

    # ── P0：容器服务登记（进程态）──────────────────────────────────
    # 仅镜像现有 agent 属性（同一对象引用），行为零变化；
    # AIAgent 上的同名属性保留为兼容代理，现有调用路径不受影响。
    # 会话态/轮次态服务（session / task-chain / journal / recovery /
    # orchestration / multi-agent）在后续阶段随插件迁移，此处不登记。
    # tools / tools.meta 由 tools 插件在装配时注册（P4）。
    ctx = XenonContext()
    agent.ctx = ctx
    ctx.provide("agent", agent)  # P5：agent-loop 插件据此构造运行期服务
    ctx.provide("llm", agent.client)
    ctx.provide("llm.routing", agent.routing_client)
    ctx.provide("memory", agent.memory_manager)
    ctx.provide("context", agent.context_manager)
    ctx.provide("cognitive", agent.cognitive_network)
    ctx.provide("sandbox", agent.sandbox_context)
    ctx.provide("health", agent.health_checker)
    ctx.provide("git", agent.git_probe)
    ctx.provide("delivery", agent.delivery_report)

    # ── P1：清单驱动插件装配 ──────────────────────────────────────
    # 默认装配 = 现状行为：插件激活只注册服务/幂等启动，不改变现有调用路径。
    # 回滚：XENON_PLUGINS=off 跳过装配；或清空 xenon.profile.yml。
    from xenon_core.loader import load_profile, ProfileWatcher

    try:
        plugin_report = load_profile(ctx)
    except Exception as error:
        logger.error("插件装配失败: %s", error)
        raise
    for line in plugin_report.render():
        print_fn(line)

    # ── P3：清单热重载（按 id 差异；XENON_PROFILE_WATCH=off 关闭）──
    if os.environ.get("XENON_PROFILE_WATCH", "").strip().lower() not in {"0", "false", "off", "no"}:
        try:
            watcher = ProfileWatcher(ctx)
            if watcher.start():
                watcher.on_reload(
                    lambda report: print_fn("\n".join(report.render()))
                )
                agent.profile_watcher = watcher
        except Exception as error:
            logger.warning("插件清单热重载启动失败: %s", error)

    # ── P4：工具层回填（tools 插件装配后）──────────────────────────
    # tools 插件已注册 tools / tools.meta 服务；此处回填 agent 兼容代理，
    # 并执行沙箱/宿主注入（与原 bootstrap 时序等效）。
    # 回退：插件未装配（XENON_PLUGINS=off 或清单禁用 tools）时直接创建 ToolManager。
    agent.tool_manager = ctx.get("tools")
    meta_tools = ctx.get("tools.meta")
    if agent.tool_manager is None:
        agent.tool_manager = tool_manager_cls()
        meta_tools = meta_tools or build_core_management_tools()

    meta_tools = meta_tools or build_core_management_tools()
    agent.load_module_tool = meta_tools["load_module_tool"]
    agent.tool_description_tool = meta_tools["tool_description_tool"]
    agent.get_module_tools_tool = meta_tools["get_module_tools_tool"]
    agent.get_module_list_tool = meta_tools["get_module_list_tool"]

    inject_host_agent = getattr(agent.tool_manager, "inject_host_agent", None)
    if callable(inject_host_agent):
        inject_host_agent(agent)

    inject_sandbox_context = getattr(agent.tool_manager, "inject_sandbox_context", None)
    if callable(inject_sandbox_context):
        injected = inject_sandbox_context(agent.sandbox_context)
        print_fn(f"\033[38;2;111;208;104m[OK] Sandbox context injected to {injected} tool(s)\033[0m")
    else:
        print_fn("\033[93m[WARN] Tool manager does not support sandbox context injection.\033[0m")

    # ── P5：运行期服务回填（agent-loop 插件装配后）──────────────────
    # AIAgent.__getattr__ 把所有运行期方法委托给 AgentRuntimeService。
    # 回退：插件未装配（XENON_PLUGINS=off 或清单禁用 agent-loop）时直接构造。
    from xenon_core.agent_runtime import AgentRuntimeService

    agent.agent_runtime = ctx.get("agent-loop") or AgentRuntimeService(agent)

