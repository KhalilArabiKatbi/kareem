"""Context-health intervention policy.

A small MLP ensemble that watches an autonomous agent's context telemetry every step
and recommends one of six interventions (NOOP, COMPACT, PRUNE_TOOLS,
REINJECT_INSTRUCTIONS, CHECKPOINT_RESET, RETRIEVE_MEMORY). Trained by the
context-health factory (see factory/ and the docs ladder under specs/optimizer/);
scores 81.4/100 on hidden confirmation scenarios where doing nothing = 0, a
hand-written heuristic = 50, and a clairvoyant planner = 100.

Quickstart:

    import kareem

    agent = kareem.load_policy()                 # bundled weights
    tracker = kareem.SessionTracker(capacity_tokens=200_000)

    for step in agent_steps:
        obs = tracker.observe(tokens_used=count_tokens(ctx),
                              tool_tokens=tool_output_tokens(ctx),
                              last_error=..., last_success=..., progress=...)
        # NOTE: confidence is a 6-way softmax over actions trained with soft targets;
        # typical max-prob is 0.3-0.4. Tune min_confidence on shadow-mode logs.
        decision = agent.decide(tracker.history, min_confidence=0.35)
        if not decision.deferred:
            apply_intervention(decision.action)        # your harness implements these
            tracker.applied(decision.action)
        else:
            tracker.applied(kareem.Action.NOOP)
"""
from __future__ import annotations

from pathlib import Path

from .actions import ACTION_COSTS, ACTION_EFFECTS, ACTION_NAMES, Action
from .observe import OBS_KEYS, SessionTracker, build_observation
from .runtime import Decision, InterventionPolicy

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

__all__ = [
    "Action",
    "ACTION_NAMES",
    "ACTION_COSTS",
    "ACTION_EFFECTS",
    "Decision",
    "InterventionPolicy",
    "SessionTracker",
    "OBS_KEYS",
    "build_observation",
    "load_policy",
]

__version__ = "0.1.0"


def load_policy(directory: str | Path | None = None) -> InterventionPolicy:
    """Load the intervention policy. With no argument, loads the bundled weights."""
    return InterventionPolicy.load(directory or _WEIGHTS_DIR)
