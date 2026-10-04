"""Observation adapters: turn a real agent loop's telemetry into the policy's schema.

The policy was trained on the simulator's 20-key observation dict (tokens measured in
thousands, capacity 128k, horizon 40). This module produces that schema from real
telemetry so callers never hand-assemble dicts or maintain counters themselves.

Two ways in:

1. `SessionTracker` (recommended) — a stateful per-session object that owns the
   history list and all counters (steps_since_*, error_streak, last_action, ...):

       tracker = SessionTracker(capacity_tokens=200_000)
       for step in agent_steps:
           obs = tracker.observe(tokens_used=count_tokens(ctx), tool_tokens=..., ...)
           decision = policy.decide(tracker.history)
           apply_intervention(decision.action)          # your harness does the work
           tracker.applied(decision.action)             # then tell the tracker

2. `build_observation(...)` — a pure function for callers who track state themselves.

Counter semantics mirror the simulator (factory/simulator/env.py): `since_*` counters
increment once per step and reset when the matching intervention is applied; COMPACT
clears the error streak, REINJECT_INSTRUCTIONS halves it, CHECKPOINT_RESET clears it.
"""
from __future__ import annotations

from .actions import Action

OBS_KEYS: tuple[str, ...] = (
    "step", "horizon", "tokens_used", "capacity", "utilization", "tokens_added_last",
    "tool_tokens", "tool_fraction", "redundancy_est", "stale_est",
    "steps_since_instruction", "steps_since_compaction", "last_error", "error_streak",
    "topic_shift", "recall_miss", "last_action", "last_success", "confidence",
    "latency_ms", "progress",
)

# Sim-like latency estimate when the caller doesn't measure it: base + per-token cost,
# deterministic part of the simulator's latency model.
_LATENCY_BASE_MS = 350.0
_LATENCY_PER_KTOK_MS = 9.0


def build_observation(
    *,
    step: int,
    tokens_used: float,
    capacity: float,
    horizon: int = 40,
    tokens_added_last: float = 0.0,
    tool_tokens: float = 0.0,
    system_tokens: float = 4.0,
    redundancy_est: float = 0.0,
    stale_est: float = 0.0,
    steps_since_instruction: int = 0,
    steps_since_compaction: int = 0,
    last_error: int = 0,
    error_streak: int = 0,
    topic_shift: int = 0,
    recall_miss: int = 0,
    last_action: int = 0,
    last_success: int = 1,
    confidence: float = 0.5,
    latency_ms: float | None = None,
    progress: int = 0,
) -> dict:
    """Assemble one observation dict. All token quantities are in THOUSANDS, matching
    the training units. Most callers should use SessionTracker instead of calling this
    directly."""
    content = max(1e-6, tokens_used - system_tokens)
    if latency_ms is None:
        latency_ms = _LATENCY_BASE_MS + _LATENCY_PER_KTOK_MS * tokens_used
    return {
        "step": int(step),
        "horizon": int(horizon),
        "tokens_used": float(tokens_used),
        "capacity": float(capacity),
        "utilization": float(tokens_used) / max(1e-6, float(capacity)),
        "tokens_added_last": float(tokens_added_last),
        "tool_tokens": float(tool_tokens),
        "tool_fraction": float(tool_tokens) / content,
        "redundancy_est": min(1.0, max(0.0, float(redundancy_est))),
        "stale_est": min(1.0, max(0.0, float(stale_est))),
        "steps_since_instruction": int(steps_since_instruction),
        "steps_since_compaction": int(steps_since_compaction),
        "last_error": int(last_error),
        "error_streak": int(error_streak),
        "topic_shift": int(topic_shift),
        "recall_miss": int(recall_miss),
        "last_action": int(last_action),
        "last_success": int(last_success),
        "confidence": min(1.0, max(0.0, float(confidence))),
        "latency_ms": float(latency_ms),
        "progress": int(progress),
    }


class SessionTracker:
    """Owns the observation history and counters for one agent session.

    Token arguments are REAL token counts (not thousands); they are converted to the
    policy's training units internally. `capacity_tokens` should be the model's context
    window; the policy was trained at 128k, so very different windows sit outside the
    training distribution (see the sim-to-real caveats in the package README).
    """

    def __init__(
        self,
        capacity_tokens: int = 128_000,
        system_tokens: int = 4_000,
        horizon: int = 40,
    ):
        self.capacity = capacity_tokens / 1000.0
        self.system_tokens = system_tokens / 1000.0
        self.horizon = int(horizon)
        self.history: list[dict] = []
        self._since_instr = 0
        self._since_compact = 0
        self._error_streak = 0
        self._last_action = int(Action.NOOP)

    def observe(
        self,
        *,
        tokens_used: int,
        tool_tokens: int = 0,
        last_error: bool = False,
        topic_shift: bool = False,
        recall_miss: bool = False,
        last_success: bool = True,
        confidence: float = 0.5,
        latency_ms: float | None = None,
        progress: int = 0,
        redundancy_est: float = 0.0,
        stale_est: float = 0.0,
    ) -> dict:
        """Snapshot the current step and append it to the history.

        `redundancy_est` / `stale_est` are caller-estimated shares in [0, 1] (duplicated
        content share; never-referenced old content share) — pass 0.0 if unmeasured,
        which is the value they hold early in training sessions.
        """
        used_k = tokens_used / 1000.0
        step = len(self.history)
        if step > 0:
            self._since_instr += 1
            self._since_compact += 1
        if last_error:
            self._error_streak += 1
        else:
            self._error_streak = 0
        prev_used = self.history[-1]["tokens_used"] if self.history else 0.0
        obs = build_observation(
            step=step,
            horizon=self.horizon,
            tokens_used=used_k,
            capacity=self.capacity,
            tokens_added_last=used_k - prev_used,
            tool_tokens=tool_tokens / 1000.0,
            system_tokens=self.system_tokens,
            redundancy_est=redundancy_est,
            stale_est=stale_est,
            steps_since_instruction=self._since_instr,
            steps_since_compaction=self._since_compact,
            last_error=int(last_error),
            error_streak=self._error_streak,
            topic_shift=int(topic_shift),
            recall_miss=int(recall_miss),
            last_action=self._last_action,
            last_success=int(last_success),
            confidence=confidence,
            latency_ms=latency_ms,
            progress=progress,
        )
        self.history.append(obs)
        return obs

    def applied(self, action: Action | str | int) -> None:
        """Record that the harness actually applied an intervention this step.

        Call this AFTER applying the action, with the action you applied (normally the
        policy's recommendation, or NOOP when a decision was deferred/overridden) — the
        next observation's counters reflect it.
        """
        action = Action(action if not isinstance(action, str) else Action[action])
        self._last_action = int(action)
        if action in (Action.COMPACT, Action.CHECKPOINT_RESET):
            self._since_compact = 0
        if action in (Action.REINJECT_INSTRUCTIONS, Action.CHECKPOINT_RESET):
            self._since_instr = 0
        if action is Action.COMPACT or action is Action.CHECKPOINT_RESET:
            self._error_streak = 0
        elif action is Action.REINJECT_INSTRUCTIONS:
            self._error_streak //= 2

    def reset(self) -> None:
        """Start a new session (clears history and all counters)."""
        self.history.clear()
        self._since_instr = 0
        self._since_compact = 0
        self._error_streak = 0
        self._last_action = int(Action.NOOP)
