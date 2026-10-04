from __future__ import annotations

from pathlib import Path
from typing import Any


def load_prompts(*, prompts_dir: Path, logger: Any) -> str:
    prompts_content = ""

    if not prompts_dir.exists():
        logger.warning("prompts 文件夹不存在: %s", prompts_dir)
        return prompts_content

    prompt_files = sorted(prompts_dir.rglob("*"), key=lambda path: str(path).lower())
    for file_path in prompt_files:
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                continue

            relative_path = file_path.relative_to(prompts_dir).as_posix()
            prompts_content += f"\n--- [{relative_path}] ---\n"
            prompts_content += content
            if not content.endswith("\n"):
                prompts_content += "\n"
        except Exception as error:
            logger.error("读取 prompts 文件失败 %s: %s", file_path, error)

    return prompts_content


def build_system_prompt(
    *,
    system_prompt_base: str,
    prompts_dir: Path,
    logger: Any,
    fragments: Any = None,
) -> str:
    """组装系统提示词。

    fragments 非空时（P3 插件路径）：base + 各插件注册的片段（按激活顺序）。
    fragments 为空时（未装配插件/回退路径）：base + prompts/ 目录内容（旧行为）。
    """
    if fragments:
        # 保留片段原文（不 strip），仅过滤空白片段——与旧路径 load_prompts 的输出逐字节一致
        body = "\n\n".join(
            str(fragment) for fragment in fragments if str(fragment or "").strip()
        )
        return system_prompt_base + ("\n" + body if body else "")

    prompts = load_prompts(prompts_dir=prompts_dir, logger=logger)
    if prompts.strip():
        return system_prompt_base + "\n" + prompts
    return system_prompt_base + "\n（prompts 文件夹当前为空或不存在）\n"
