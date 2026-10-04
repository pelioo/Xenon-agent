# -*- coding: utf-8 -*-
"""插件加载器（P1）：读清单 → 补丁层组合 → 导入模块 → 校验声明 → 拓扑排序 → 激活。

装配语义（与 DSH 的 cordis.patch.yml 同构）：
- 插件行 = {id, name, enabled, optional, config, dependencies}
- 层序：内置清单（base）→ 项目清单（profile）→ 机器层补丁（home）；后写覆盖先写
- 同 id 覆盖：整行替换 config（不深合并）；缺省字段（name/enabled/optional）沿用旧行
- `disabled: true` 移除该行（幂等：不存在则忽略）

P1 约定：
- 默认装配 = 现状行为：插件激活只做"注册服务/幂等启动"，不改变现有调用路径
- 回滚开关：环境变量 XENON_PLUGINS=off|0|false 跳过装配；或清空 xenon.profile.yml
- `python -m xenon_core.loader --dump-profile` 打印组合后的插件树（不激活），用于排障

后续阶段：
- P3：热重载（watchdog 监听清单，按 id 差异重载）；plugins/ 用户目录扫描兜底
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from xenon_core.context import XenonContext

logger = logging.getLogger(__name__)

# 层标签
LAYER_BASE = "base"
LAYER_PROFILE = "profile"
LAYER_HOME = "home"

# 装配开关：XENON_PLUGINS=off|0|false 跳过装配
PLUGINS_DISABLED_VALUES = {"0", "false", "off", "no", "disable", "disabled"}
PLUGINS_ENV_KEY = "XENON_PLUGINS"

# 默认路径（相对本文件定位项目根）
_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_DIR.parent
DEFAULT_BASE_MANIFEST = _PACKAGE_DIR / "plugins" / "manifest.yml"
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "xenon.profile.yml"
DEFAULT_HOME_PATCH = Path.home() / ".xenon" / "patch.yml"

# 插件行字段
_ROW_FIELDS = ("id", "name", "enabled", "optional", "config", "dependencies", "disabled")


class LoadError(Exception):
    """插件装配失败（fail-loud）。"""


@dataclass
class PluginSpec:
    """组合后的一个插件行（跨层覆盖后的最终形态）。"""

    id: str
    name: str
    enabled: bool = True
    optional: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    source: str = LAYER_BASE  # 最后一次写入该行的层标签


@dataclass
class LoadReport:
    """一次装配的报告：分层来源、组合结果、激活/禁用/失败/跳过清单。"""

    layers: List[Tuple[str, str]] = field(default_factory=list)  # (label, path)
    specs: List[PluginSpec] = field(default_factory=list)
    activated: List[str] = field(default_factory=list)
    disabled: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)  # (id, error)
    errors: List[str] = field(default_factory=list)

    def render(self) -> List[str]:
        """渲染为终端友好的多行文本（供启动报告/dump 使用）。"""
        lines = ["\n── 插件装配报告 ──"]
        for label, path in self.layers:
            lines.append(f"  [{label}] {path}")
        if not self.specs:
            lines.append("  （无插件行：装配被禁用或清单为空）")
        for spec in self.specs:
            state = (
                "disabled"
                if not spec.enabled
                else "active"
                if spec.id in self.activated
                else "skipped"
                if spec.id in self.skipped
                else "failed"
                if any(fid == spec.id for fid, _ in self.failed)
                else "loaded"
            )
            flags = "optional" if spec.optional else ""
            lines.append(
                f"  {state:8s} {spec.id:16s} {spec.name}  {flags}".rstrip()
            )
        for pid, error in self.failed:
            lines.append(f"  FAILED {pid}: {error}")
        for error in self.errors:
            lines.append(f"  ERROR {error}")
        return lines


# ────────────────────────── 清单读取 ──────────────────────────


def load_yaml_document(path: Path) -> Any:
    """读取 YAML 文档；文件不存在返回 None。"""
    if not path.exists():
        return None
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_plugin_rows(data: Any) -> List[Dict[str, Any]]:
    """把清单文档归一化为插件行列表。

    支持两种形状：
      - 顶层列表：[{id, name, ...}]
      - {plugins: [...]}（用户层常用）
    """
    if data is None:
        return []
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        rows = data.get("plugins")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    raise LoadError(f"清单格式无效：应为插件行列表或 {{plugins: [...]}}，实际为 {type(data).__name__}")


# ────────────────────────── 补丁层组合 ──────────────────────────


def compose_layers(*layers: Tuple[str, List[Dict[str, Any]]]) -> List[PluginSpec]:
    """按层序组合插件行：同 id 整行覆盖 config，disabled 移除，保持首次出现顺序。"""
    merged: Dict[str, PluginSpec] = {}
    order: List[str] = []

    for label, rows in layers:
        for row in rows:
            plugin_id = row.get("id")
            if not isinstance(plugin_id, str) or not plugin_id:
                raise LoadError(f"[{label}] 插件行缺少有效的 id：{row!r}")
            if row.get("disabled"):
                if plugin_id in merged:
                    del merged[plugin_id]
                    order.remove(plugin_id)
                continue
            previous = merged.get(plugin_id)
            spec = PluginSpec(
                id=plugin_id,
                name=str(row.get("name") or (previous.name if previous else "")),
                enabled=bool(row.get("enabled", previous.enabled if previous else True)),
                optional=bool(row.get("optional", previous.optional if previous else False)),
                config=dict(row.get("config") or {}),
                dependencies=[
                    str(dep) for dep in (row.get("dependencies") or []) if str(dep)
                ],
                source=label,
            )
            if previous is None:
                merged[plugin_id] = spec
                order.append(plugin_id)
            else:
                merged[plugin_id] = spec  # 原位替换，保持顺序
    return [merged[plugin_id] for plugin_id in order]


# ────────────────────────── 依赖拓扑排序 ──────────────────────────


def topo_sort(specs: Sequence[PluginSpec]) -> List[PluginSpec]:
    """按 dependencies（插件 id）拓扑排序；环或缺失依赖抛 LoadError。"""
    by_id = {spec.id: spec for spec in specs}
    order: List[PluginSpec] = []
    visiting: set = set()
    visited: set = set()

    def visit(plugin_id: str) -> None:
        if plugin_id in visited:
            return
        if plugin_id in visiting:
            raise LoadError(f"插件依赖成环：{plugin_id} 出现在依赖链中")
        spec = by_id.get(plugin_id)
        if spec is None:
            raise LoadError(f"插件依赖缺失：{plugin_id} 未在清单中声明")
        visiting.add(plugin_id)
        for dep in spec.dependencies:
            visit(dep)
        visiting.discard(plugin_id)
        visited.add(plugin_id)
        order.append(spec)

    for spec in specs:
        visit(spec.id)
    return order


# ────────────────────────── 插件模块导入与声明校验 ──────────────────────────


def import_plugin_module(name: str) -> Any:
    """按模块路径导入插件模块（如 xenon_core.plugins.heartbeat）。"""
    try:
        return importlib.import_module(name)
    except Exception as error:
        raise LoadError(f"导入插件模块失败 {name}: {error}") from error


def validate_plugin_declaration(module: Any, spec: PluginSpec) -> Dict[str, Any]:
    """校验插件模块的 PLUGIN 声明与清单行一致；返回声明 dict。"""
    declaration = getattr(module, "PLUGIN", None)
    if not isinstance(declaration, dict):
        raise LoadError(f"插件模块 {spec.name} 缺少 PLUGIN 声明（dict）")
    declared_id = declaration.get("id")
    if declared_id != spec.id:
        raise LoadError(
            f"插件声明 id 与清单不一致：清单={spec.id!r}，模块={declared_id!r}（{spec.name}）"
        )
    declared_name = declaration.get("name")
    if declared_name and declared_name != spec.name:
        raise LoadError(
            f"插件声明 name 与清单不一致：清单={spec.name!r}，模块={declared_name!r}"
        )
    return declaration


# ────────────────────────── 装配主流程 ──────────────────────────


def plugins_enabled() -> bool:
    """XENON_PLUGINS 装配开关：off|0|false 等值禁用。"""
    return os.environ.get(PLUGINS_ENV_KEY, "").strip().lower() not in PLUGINS_DISABLED_VALUES


def load_profile(
    ctx: XenonContext,
    *,
    base_manifest: Optional[Path] = None,
    profile_path: Optional[Path] = None,
    home_patch_path: Optional[Path] = None,
    fail_loud: bool = True,
    print_fn: Any = None,
) -> LoadReport:
    """装配一次插件树（P1：全量装配；P3 升级为按 id 差异重载）。

    - 层序：base → profile → home；文件缺失的层跳过
    - fail_loud=True 且存在非 optional 插件失败时抛 LoadError（启动即暴露）
    - 返回 LoadReport（激活/禁用/失败/跳过 + 渲染文本）
    """
    report = LoadReport()
    if not plugins_enabled():
        report.errors.append(f"装配已禁用（{PLUGINS_ENV_KEY}=off）")
        return report

    base_path = Path(base_manifest or DEFAULT_BASE_MANIFEST)
    profile_path = Path(profile_path or DEFAULT_PROFILE_PATH)
    home_path = Path(home_patch_path or DEFAULT_HOME_PATCH)

    layers: List[Tuple[str, Path]] = []
    if base_path.exists():
        layers.append((LAYER_BASE, base_path))
    else:
        raise LoadError(f"内置插件清单缺失：{base_path}")
    if profile_path.exists():
        layers.append((LAYER_PROFILE, profile_path))
    if home_path.exists():
        layers.append((LAYER_HOME, home_path))

    report.layers = [(label, str(path)) for label, path in layers]
    layer_rows: List[Tuple[str, List[Dict[str, Any]]]] = []
    for label, path in layers:
        data = load_yaml_document(path)
        try:
            rows = parse_plugin_rows(data)
        except LoadError as error:
            report.errors.append(f"[{label}] {error}")
            if fail_loud:
                raise
            rows = []
        layer_rows.append((label, rows))

    specs = compose_layers(*layer_rows)
    report.specs = specs
    report.disabled = [spec.id for spec in specs if not spec.enabled]

    # 全部行登记进容器（含 disabled：以 inactive 状态存在，便于检视"存在但被禁用"）
    for spec in specs:
        ctx.register_plugin(
            spec.id,
            name=spec.name,
            config=spec.config,
            dependencies=spec.dependencies,
        )

    enabled_specs = [spec for spec in specs if spec.enabled]

    # 预检：enabled 插件的依赖不能是 disabled 插件
    disabled_ids = {spec.id for spec in specs if not spec.enabled}
    for spec in enabled_specs:
        for dep in spec.dependencies:
            if dep in disabled_ids:
                error = f"插件 {spec.id} 依赖的插件 {dep} 处于禁用状态"
                report.errors.append(error)
                if fail_loud:
                    raise LoadError(error)

    try:
        ordered = topo_sort(enabled_specs)
    except LoadError as error:
        report.errors.append(str(error))
        if fail_loud:
            raise
        return report

    for spec in ordered:
        error = _activate_single_spec(ctx, spec)
        if error is None:
            report.activated.append(spec.id)
        else:
            report.failed.append((spec.id, error))

    _finalize_report(report, specs, fail_loud)
    return report


def _activate_single_spec(ctx: XenonContext, spec: PluginSpec) -> Optional[str]:
    """导入、校验、注册并激活单个插件；成功返回 None，失败返回错误信息。"""
    try:
        module = import_plugin_module(spec.name)
        declaration = validate_plugin_declaration(module, spec)

        # 服务依赖预检（inject 声明的服务 key 必须已存在）
        missing_services = [
            key for key in (declaration.get("inject") or []) if not ctx.has(key)
        ]
        if missing_services:
            return f"插件 {spec.id} 依赖的服务缺失：{', '.join(missing_services)}"

        ctx.register_plugin(
            spec.id,
            name=spec.name,
            config=spec.config,
            dependencies=spec.dependencies,
            activate_fn=getattr(module, "activate", None),
            deactivate_fn=getattr(module, "deactivate", None),
        )
        if ctx.activate(spec.id):
            return None
        record = ctx.plugins().get(spec.id)
        return record.error if record else "激活失败"
    except LoadError as error:
        return str(error)
    except Exception as error:
        return str(error)


def _finalize_report(report: LoadReport, specs: Sequence[PluginSpec], fail_loud: bool) -> None:
    """非 optional 失败时按 fail_loud 抛错。"""
    non_optional_failures = [
        (pid, err)
        for pid, err in report.failed
        if not any(spec.id == pid and spec.optional for spec in specs)
    ]
    if non_optional_failures:
        summary = "; ".join(f"{pid}: {err}" for pid, err in non_optional_failures)
        if fail_loud:
            raise LoadError(f"插件装配失败（{len(non_optional_failures)} 个）：{summary}")


# ────────────────────────── 热重载（P3：按 id 差异）──────────────────────────


def apply_profile_diff(
    ctx: XenonContext,
    old_specs: Sequence[PluginSpec],
    new_specs: Sequence[PluginSpec],
    *,
    fail_loud: bool = True,
) -> LoadReport:
    """对比新旧组合，按 id 差异应用变更（热重载核心）。

    变更规则：
    - 移除：deactivate 该插件（若 active）
    - 新增：register + activate（依赖已存在才可激活）
    - 变更（enabled/config/name/dependencies 任一不同）：deactivate + 重新 activate
    - 级联：依赖被变更/移除插件的已启用插件，一并重启（deactivate + activate）

    热重载是尽力而为：失败记录进报告，不抛错（fail_loud 仅用于说明）。
    """
    report = LoadReport()
    old_by_id = {spec.id: spec for spec in old_specs}
    new_by_id = {spec.id: spec for spec in new_specs}
    all_ids = sorted(set(old_by_id) | set(new_by_id))

    def _spec_changed(old: PluginSpec, new: PluginSpec) -> bool:
        return (
            old.name != new.name
            or old.enabled != new.enabled
            or old.optional != new.optional
            or old.config != new.config
            or old.dependencies != new.dependencies
        )

    changed_ids: List[str] = []
    removed_ids: List[str] = []

    for plugin_id in all_ids:
        old = old_by_id.get(plugin_id)
        new = new_by_id.get(plugin_id)
        if old is None:  # 新增
            if new.enabled:
                error = _activate_single_spec(ctx, new)
                if error is None:
                    report.activated.append(plugin_id)
                else:
                    report.failed.append((plugin_id, error))
            continue
        if new is None:  # 移除
            if old.enabled:
                removed_ids.append(plugin_id)
            continue
        if not new.enabled and old.enabled:  # 禁用
            removed_ids.append(plugin_id)
            continue
        if new.enabled and not old.enabled:  # 启用
            error = _activate_single_spec(ctx, new)
            if error is None:
                report.activated.append(plugin_id)
            else:
                report.failed.append((plugin_id, error))
            continue
        if _spec_changed(old, new):  # 变更
            changed_ids.append(plugin_id)

    # 级联：依赖被变更/移除插件的已启用插件（传递闭包）
    affected = set(changed_ids) | set(removed_ids)
    dependents: set = set()
    new_by_id_enabled = {pid: spec for pid, spec in new_by_id.items() if spec.enabled}
    for plugin_id, spec in new_by_id_enabled.items():
        if plugin_id in affected:
            continue
        stack = list(spec.dependencies)
        seen = set()
        while stack:
            dep = stack.pop()
            if dep in seen:
                continue
            seen.add(dep)
            if dep in affected:
                dependents.add(plugin_id)
                break
            dep_spec = new_by_id.get(dep)
            if dep_spec:
                stack.extend(dep_spec.dependencies)
    affected |= dependents

    # 停用（逆激活序，先停用级联依赖者）
    records = ctx.plugins()
    active_order = [pid for pid in reversed(ctx._active_order) if pid in affected]
    for plugin_id in active_order:
        if records.get(plugin_id) is not None:
            ctx.deactivate(plugin_id)
            report.skipped.append(plugin_id)

    # 重新激活（拓扑序）
    restart_ids = [pid for pid in affected if pid in new_by_id and new_by_id[pid].enabled]
    restart_specs = [new_by_id[pid] for pid in restart_ids]
    try:
        ordered = topo_sort(restart_specs)
    except LoadError as error:
        report.errors.append(str(error))
        ordered = restart_specs
    for spec in ordered:
        error = _activate_single_spec(ctx, spec)
        if error is None:
            report.activated.append(spec.id)
        else:
            report.failed.append((spec.id, error))

    report.disabled = [spec.id for spec in new_specs if not spec.enabled]
    return report


class ProfileWatcher:
    """监听清单文件（xenon.profile.yml / ~/.xenon/patch.yml），变化后按 id 差异热重载。"""

    def __init__(
        self,
        ctx: XenonContext,
        *,
        base_manifest: Optional[Path] = None,
        profile_path: Optional[Path] = None,
        home_patch_path: Optional[Path] = None,
        debounce: float = 1.0,
    ) -> None:
        self._ctx = ctx
        self._base_manifest = Path(base_manifest or DEFAULT_BASE_MANIFEST)
        self._profile_path = Path(profile_path or DEFAULT_PROFILE_PATH)
        self._home_path = Path(home_patch_path or DEFAULT_HOME_PATCH)
        self._debounce = debounce
        self._observer: Any = None
        self._debounce_timer: Any = None
        self._debounce_lock = threading.Lock()
        self._listeners: List[Callable[[LoadReport], None]] = []
        self._current_specs: List[PluginSpec] = []

    def on_reload(self, fn: Callable[[LoadReport], None]) -> Callable[[LoadReport], None]:
        self._listeners.append(fn)
        return fn

    def start(self) -> bool:
        """启动清单监听；返回是否成功。"""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning("watchdog 不可用，插件清单热重载已禁用")
            return False

        watched = []
        for path in (self._profile_path, self._home_path):
            if path.exists() and path.parent not in watched:
                watched.append(path.parent)
        if not watched:
            return False

        self._current_specs = self._compose()

        watcher = self

        class _ManifestHandler(FileSystemEventHandler):
            def on_modified(self, event):
                self._handle(event)

            def on_created(self, event):
                self._handle(event)

            def on_moved(self, event):
                self._handle(event)

            def _handle(self, event):
                path = Path(getattr(event, "src_path", "") or "")
                if path.name not in {watcher._profile_path.name, watcher._home_path.name}:
                    return
                with watcher._debounce_lock:
                    if watcher._debounce_timer is not None:
                        watcher._debounce_timer.cancel()
                    watcher._debounce_timer = threading.Timer(
                        watcher._debounce, watcher._reload
                    )
                    watcher._debounce_timer.daemon = True
                    watcher._debounce_timer.start()

            def _reload(self):
                watcher._reload()

        self._observer = Observer()
        self._observer.daemon = True
        for directory in watched:
            self._observer.schedule(_ManifestHandler(), str(directory), recursive=False)
        self._observer.start()
        logger.info("插件清单热重载已启动，监听: %s", ", ".join(str(p) for p in watched))
        return True

    def stop(self) -> None:
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=3)
            except Exception:
                pass
            finally:
                self._observer = None

    def _compose(self) -> List[PluginSpec]:
        layers: List[Tuple[str, List[Dict[str, Any]]]] = [(LAYER_BASE, [])]
        if self._base_manifest.exists():
            layers = [
                (LAYER_BASE, parse_plugin_rows(load_yaml_document(self._base_manifest)))
            ]
        if self._profile_path.exists():
            layers.append((LAYER_PROFILE, parse_plugin_rows(load_yaml_document(self._profile_path))))
        if self._home_path.exists():
            layers.append((LAYER_HOME, parse_plugin_rows(load_yaml_document(self._home_path))))
        return compose_layers(*layers)

    def _reload(self) -> None:
        try:
            if not plugins_enabled():
                return
            new_specs = self._compose()
            report = apply_profile_diff(self._ctx, self._current_specs, new_specs)
            self._current_specs = new_specs
            for fn in self._listeners:
                try:
                    fn(report)
                except Exception as error:
                    logger.warning("清单热重载回调失败: %s", error)
        except Exception as error:
            logger.warning("清单热重载失败: %s", error)


# ────────────────────────── 配置树 dump（不激活）──────────────────────────


def dump_profile(
    *,
    base_manifest: Optional[Path] = None,
    profile_path: Optional[Path] = None,
    home_patch_path: Optional[Path] = None,
) -> str:
    """打印组合后的插件树（不导入、不激活），用于排障与装配检查。"""
    base_path = Path(base_manifest or DEFAULT_BASE_MANIFEST)
    profile_path = Path(profile_path or DEFAULT_PROFILE_PATH)
    home_path = Path(home_patch_path or DEFAULT_HOME_PATCH)

    layers: List[Tuple[str, Path]] = [(LAYER_BASE, base_path)]
    if profile_path.exists():
        layers.append((LAYER_PROFILE, profile_path))
    if home_path.exists():
        layers.append((LAYER_HOME, home_path))

    lines = ["Xenon 插件装配（组合后，未激活）", "─" * 40]
    layer_rows: List[Tuple[str, List[Dict[str, Any]]]] = []
    for label, path in layers:
        marker = "" if path.exists() else "（不存在，跳过）"
        lines.append(f"[{label}] {path} {marker}")
        rows = parse_plugin_rows(load_yaml_document(path)) if path.exists() else []
        layer_rows.append((label, rows))

    lines.append("")
    lines.append("最终插件行（按 id 寻址，后写覆盖先写）:")
    if not plugins_enabled():
        lines.append(f"  （装配已禁用：{PLUGINS_ENV_KEY}=off）")
        return "\n".join(lines)
    try:
        specs = compose_layers(*layer_rows)
    except LoadError as error:
        lines.append(f"  ERROR {error}")
        return "\n".join(lines)
    if not specs:
        lines.append("  （空清单）")
    for spec in specs:
        state = "enabled" if spec.enabled else "disabled"
        optional = ", optional" if spec.optional else ""
        deps = f", deps=[{', '.join(spec.dependencies)}]" if spec.dependencies else ""
        lines.append(
            f"  - {state:8s} {spec.id:16s} {spec.name}{optional}{deps}"
        )
        if spec.config:
            lines.append(f"      config: {spec.config!r}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="xenon-loader", description="Xenon 插件加载器")
    parser.add_argument("--dump-profile", action="store_true", help="打印组合后的插件树（不激活）")
    parser.add_argument("--profile", default=None, help="自定义项目清单路径")
    args = parser.parse_args(argv)

    if args.dump_profile:
        print(dump_profile(profile_path=args.profile))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
