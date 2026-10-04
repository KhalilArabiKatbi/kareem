"""Meta-Done conditions. Pure functions over Python-owned state. No LLM input reaches this module."""
from __future__ import annotations

from . import config


def check(state: dict, max_iterations: int, window: int = config.STAGNATION_WINDOW,
          min_improvement: float = config.MIN_IMPROVEMENT) -> str | None:
    """Return a halt reason, or None to continue. `state["next_iteration"]` iterations are closed."""
    closed = state["next_iteration"]

    # 3. hard limit
    if closed >= max_iterations:
        return f"max-iterations reached ({closed}/{max_iterations})"

    if closed < window:
        return None

    # 1. no new dimension added in the last `window` iterations
    last_added = state.get("last_dimension_added_iteration", -1)
    if last_added < closed - window:
        return (
            f"state space stagnant: no dimension added in the last {window} iterations "
            f"(last addition at iteration {last_added})"
        )

    # 2. best score in the last `window` iterations did not beat the earlier best by > min_improvement
    scores = {int(k): v for k, v in state.get("scores", {}).items() if v is not None}
    before = [v for i, v in scores.items() if i < closed - window]
    recent = [v for i, v in scores.items() if i >= closed - window]
    if before:
        best_before = max(before)
        best_recent = max(recent) if recent else None
        if best_recent is None or best_recent - best_before <= min_improvement:
            return (
                f"score plateau: best of last {window} iterations = {best_recent}, "
                f"best before = {best_before}; improvement <= {min_improvement}"
            )
    return None
