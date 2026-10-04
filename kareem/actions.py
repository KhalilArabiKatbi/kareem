"""The six context-health interventions, in the policy's logit order.

The integer values are load-bearing: observation dicts carry `last_action` as an
index into this order, and the saved weights emit logits in this order.
"""
from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    NOOP = 0
    COMPACT = 1
    PRUNE_TOOLS = 2
    REINJECT_INSTRUCTIONS = 3
    CHECKPOINT_RESET = 4
    RETRIEVE_MEMORY = 5


ACTION_NAMES: tuple[str, ...] = tuple(a.name for a in Action)

# Return-units cost of each action, from the simulator's reward model. Useful when a
# harness wants to weigh a recommendation against its disruption.
ACTION_COSTS: dict[Action, float] = {
    Action.NOOP: 0.0,
    Action.COMPACT: 0.8,
    Action.PRUNE_TOOLS: 0.25,
    Action.REINJECT_INSTRUCTIONS: 0.35,
    Action.CHECKPOINT_RESET: 2.0,
    Action.RETRIEVE_MEMORY: 0.35,
}

# What a host agent harness should do for each action.
ACTION_EFFECTS: dict[Action, str] = {
    Action.NOOP: "nothing",
    Action.COMPACT: "summarise the conversation and replace it with the summary; clears retry transcripts and the error streak",
    Action.PRUNE_TOOLS: "drop old tool outputs from the context, keeping the most recent one",
    Action.REINJECT_INSTRUCTIONS: "re-append the system prompt / task instructions; halves the error streak",
    Action.CHECKPOINT_RESET: "start a fresh context seeded with a checkpoint summary",
    Action.RETRIEVE_MEMORY: "fetch facts from long-term memory / notes / RAG back into the context",
}
