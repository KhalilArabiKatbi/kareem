"""Independent re-derivation of the routing rules, used to audit every router decision.

This is intentionally a second, table-driven implementation. If `router.py` and this module ever
disagree, the router log verdict becomes ROUTER_MISMATCH.
"""
from __future__ import annotations

from dataclasses import asdict

QUALITY_THRESHOLD = 60
META_STREAK_LIMIT = 2

# level -> (model, effort)
_TABLE = {
    0: ("sonnet-5", "standard"),
    1: ("sonnet-5", "high"),
    2: ("opus-5.5", "standard"),
    3: ("opus-5.5", "high"),
}
_START_LEVEL = {
    # (agent, condition) -> starting level
    ("worker", "low_score"): 2,
    ("judge", "low_score"): 2,
    ("meta", "meta_streak"): 3,
}


def expected_route(ctx) -> tuple[str, str]:
    c = asdict(ctx) if not isinstance(ctx, dict) else ctx
    agent, purpose = c["agent"], c.get("purpose", "primary")
    retry = bool(c.get("step_retry", 0))
    if agent == "doc" and purpose == "format":
        return "fable-5.1", ("high" if retry else "standard")

    start = 0
    score = c.get("last_judge_score")
    if score is not None and score < QUALITY_THRESHOLD:
        start = max(start, _START_LEVEL.get((agent, "low_score"), 0))
    if c.get("meta_invalid_streak", 0) >= META_STREAK_LIMIT:
        start = max(start, _START_LEVEL.get((agent, "meta_streak"), 0))
    level = min(3, start + int(c.get("escalation", 0)))
    model, effort = _TABLE[level]
    if retry:
        effort = "high"
    return model, effort


def audit(ctx, decision) -> dict:
    exp_model, exp_effort = expected_route(ctx)
    got = (decision.model, decision.effort) if decision is not None else (None, None)
    return {
        "expected_model": exp_model,
        "expected_effort": exp_effort,
        "match": got == (exp_model, exp_effort),
    }
