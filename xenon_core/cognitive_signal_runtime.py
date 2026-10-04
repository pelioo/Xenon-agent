from __future__ import annotations

from typing import Any, Callable, List, Optional


def get_cognitive_network_summary(
    *,
    cognitive_network: Any,
    cached_summary: str,
    logger: Any,
    set_cached_summary_fn: Callable[[str], None],
    current_query: Optional[str] = None,
    current_phase: Optional[str] = None,
    current_intent: Optional[str] = None,
    recent_failures: Optional[List[str]] = None,
) -> str:
    """Build and cache a compact cognitive-state summary."""
    try:
        if current_phase or current_intent or recent_failures:
            summary = cognitive_network.build_phase_summary(
                current_query=current_query,
                current_phase=current_phase,
                current_intent=current_intent,
                recent_failures=recent_failures or [],
            )
        else:
            summary = cognitive_network.build_summary(current_query=current_query)
        if summary:
            set_cached_summary_fn(summary)
        return summary
    except Exception as error:
        logger.error("Failed to build cognitive network summary: %s", error)
        return cached_summary
