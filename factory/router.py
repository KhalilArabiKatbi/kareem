"""Model & effort router: the escalation ladder.

The LLM never picks a model. Python assigns one from failure history using the rules below.

Ladder levels
    L0 = sonnet-5  + standard
    L1 = sonnet-5  + high
    L2 = opus-5.5  + standard
    L3 = opus-5.5  + high

Rules (applied in this order)
    R1 baseline     every core agent starts at L0
    R3 quality      last judge score < 60  -> worker and judge start at L2 for the next iteration
    R4 complexity   meta failed twice in a row -> meta starts at L3
    R6 iteration    each iteration-level failure bumps the failing agent one level (cap L3);
                    tracked per iteration by EscalationTracker, so parallel iterations never interact
    R2 micro        in-place step retry -> effort forced to high on the same model
    R5 fable        fable-5.1 only for doc/format; anything else raises RoutingViolationError
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config
from .errors import RoutingViolationError

LADDER: tuple[tuple[str, str], ...] = (
    ("sonnet-5", "standard"),
    ("sonnet-5", "high"),
    ("opus-5.5", "standard"),
    ("opus-5.5", "high"),
)
MAX_LEVEL = len(LADDER) - 1
FABLE = "fable-5.1"
FABLE_ALLOWED = ("doc", "format")


@dataclass(frozen=True)
class RouteContext:
    iteration: int
    agent: str
    purpose: str = "primary"          # doc uses "draft", "format", "final"
    attempt: int = 0                  # iteration-level attempt index (0..2)
    step_retry: int = 0               # 0 = first call of the step, 1 = in-place retry
    escalation: int = 0               # iteration-level bumps recorded for this agent
    last_judge_score: int | None = None
    meta_invalid_streak: int = 0


@dataclass(frozen=True)
class RouteDecision:
    agent: str
    purpose: str
    model: str
    effort: str
    level: int
    rules: tuple[str, ...]

    @property
    def model_id(self) -> str:
        return config.MODEL_IDS[self.model]

    @property
    def cli_effort(self) -> str:
        return config.EFFORT_FLAGS[self.effort]


def enforce_fable_constraint(agent: str, purpose: str, model: str) -> None:
    """Rule 5. Raises RoutingViolationError for any forbidden Fable pairing."""
    if model == FABLE and (agent, purpose) != FABLE_ALLOWED:
        raise RoutingViolationError(
            f"RoutingViolationError: {FABLE} routed to agent={agent!r} purpose={purpose!r}; "
            f"{FABLE} is reserved for the Documenting Agent's Markdown formatting pass.",
            agent=agent,
        )
    if agent not in config.ALL_ROUTABLE:
        raise RoutingViolationError(f"unknown agent {agent!r}", agent=agent)
    if agent == "simulator":
        # The simulator is pure Python. Routing any model to it is a violation.
        raise RoutingViolationError("the simulator must never be routed to an LLM", agent=agent)


class EscalationTracker:
    """Rule 6 bookkeeping for ONE iteration (or one wave's Meta phase).

    Each parallel iteration owns its own tracker, so one iteration's failures can never change the
    routing of another. The router itself is stateless.
    """

    def __init__(self):
        self._levels: dict[str, int] = {}

    def escalate(self, agent: str) -> int:
        level = min(MAX_LEVEL, self._levels.get(agent, 0) + 1)
        self._levels[agent] = level
        return level

    def escalation(self, agent: str) -> int:
        return self._levels.get(agent, 0)


class ModelRouter:
    """Stateless: route() is a pure function of RouteContext (plus test-only overrides)."""

    def __init__(self, overrides: dict[str, str] | None = None):
        # `overrides` exists only so tests can inject a faulty route and prove the guard fires.
        self._overrides = dict(overrides or {})

    # ---- routing ----
    def route(self, ctx: RouteContext) -> RouteDecision:
        agent, purpose = ctx.agent, ctx.purpose
        if (agent, purpose) == FABLE_ALLOWED:
            effort = "high" if ctx.step_retry else "standard"
            rules = ("R5 fable-format",) + (("R2 micro-retry",) if ctx.step_retry else ())
            decision = RouteDecision(agent, purpose, FABLE, effort, -1, rules)
        else:
            level = 0
            rules = ["R1 baseline"]
            if (
                agent in ("worker", "judge")
                and ctx.last_judge_score is not None
                and ctx.last_judge_score < config.QUALITY_THRESHOLD
            ):
                level = max(level, 2)
                rules.append(f"R3 quality(last_score={ctx.last_judge_score})")
            if agent == "meta" and ctx.meta_invalid_streak >= config.META_FAIL_ESCALATION:
                level = max(level, 3)
                rules.append(f"R4 complexity(meta_invalid_streak={ctx.meta_invalid_streak})")
            if ctx.escalation:
                level = min(MAX_LEVEL, level + ctx.escalation)
                rules.append(f"R6 iteration-escalation(+{ctx.escalation})")
            model, effort = LADDER[level]
            if ctx.step_retry and effort != "high":
                effort = "high"
                rules.append("R2 micro-retry")
            decision = RouteDecision(agent, purpose, model, effort, level, tuple(rules))

        forced = self._overrides.get(agent)
        if forced is not None:
            decision = RouteDecision(
                agent, purpose, forced, decision.effort, decision.level, decision.rules + ("TEST-OVERRIDE",)
            )
        enforce_fable_constraint(decision.agent, decision.purpose, decision.model)
        return decision


# ---- router_log.json ----------------------------------------------------------------------
class FanOutRouterLog:
    """Records one decision into several experiments' router logs (the wave's shared Meta call)."""

    def __init__(self, logs: list["RouterLog"]):
        self.logs = logs

    def record(self, *args, **kwargs) -> None:
        for log in self.logs:
            log.record(*args, **kwargs)


@dataclass
class RouterLog:
    path: Path
    experiment: str
    entries: list[dict] = field(default_factory=list)

    def record(self, ctx: RouteContext, decision: RouteDecision | None, audit: dict, outcome: str,
               error: str | None = None, duration_s: float | None = None, cost_usd: float | None = None) -> None:
        self.entries.append(
            {
                "seq": len(self.entries),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "context": asdict(ctx),
                "decision": None if decision is None else {
                    "model": decision.model,
                    "model_id": decision.model_id,
                    "effort": decision.effort,
                    "cli_effort": decision.cli_effort,
                    "level": decision.level,
                    "rules": list(decision.rules),
                },
                "audit": audit,
                "outcome": outcome,
                "error": error,
                "duration_s": duration_s,
                "cost_usd": cost_usd,
            }
        )
        self.flush()

    def verdict(self) -> str:
        if not self.entries:
            return "EMPTY"
        for e in self.entries:
            if e["outcome"] == "routing_violation":
                continue  # a blocked violation is the guard working, not a routing error
            if not e["audit"].get("match"):
                return "ROUTER_MISMATCH"
            d = e["decision"] or {}
            if d.get("model") == FABLE and (e["context"]["agent"], e["context"]["purpose"]) != FABLE_ALLOWED:
                return "FABLE_LEAK"
        return "ROUTER_VERIFIED"

    def flush(self) -> None:
        payload = {
            "experiment": self.experiment,
            "ladder": [list(x) for x in LADDER],
            "rules": {
                "R1": "baseline sonnet-5/standard for meta, worker, judge, doc",
                "R2": "in-place step retry after invalid JSON or crash -> effort high",
                "R3": f"last judge score < {config.QUALITY_THRESHOLD} -> worker and judge at opus-5.5/standard",
                "R4": f"meta failed {config.META_FAIL_ESCALATION}x in a row -> meta at opus-5.5/high",
                "R5": "fable-5.1 only for doc/format; otherwise RoutingViolationError",
                "R6": "iteration-level failure bumps the failing agent one ladder level",
            },
            "verdict": self.verdict(),
            "decisions": self.entries,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)
