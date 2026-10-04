"""Exception hierarchy.

MicroFail  -> the iteration retry loop catches it, escalates, and retries (max 3, then poison).
MacroFail  -> never caught by the retry loop. The factory aborts.
InfraUnavailable -> the LLM transport is down or out of quota. The factory halts with resumable state.
"""
from __future__ import annotations


class FactoryError(Exception):
    """Base class for all harness errors."""


# ---- Micro-fails ---------------------------------------------------------------------------
class MicroFail(FactoryError):
    agent: str = "unknown"

    def __init__(self, message: str, agent: str | None = None):
        super().__init__(message)
        if agent is not None:
            self.agent = agent


class AgentCallError(MicroFail):
    """The LLM process crashed, timed out, or returned an error envelope."""


class AgentOutputError(MicroFail):
    """The agent returned invalid JSON or JSON that fails its schema."""


class MetaProposalInvalid(AgentOutputError):
    """The Meta-Agent's mutation or coordinate is not valid for the current state space."""

    agent = "meta"


class JudgeAnchorError(AgentOutputError):
    """The Judge score drifted too far from the deterministic simulator score."""

    agent = "judge"


class WorkerCodeError(MicroFail):
    """The Worker's model.py failed the static gate."""

    agent = "worker"


class SimulatorError(MicroFail):
    """model.py crashed inside the simulator or the simulator output is invalid."""

    agent = "worker"


class SimulatorTimeout(SimulatorError):
    """The simulator exceeded its hard timeout."""


class RoutingViolationError(MicroFail):
    """The router produced a forbidden model/agent pairing (Fable constraint)."""


# ---- Macro-fails ---------------------------------------------------------------------------
class MacroFail(FactoryError):
    """Catastrophic failure. The factory must abort."""


class IsolationViolation(MacroFail):
    """A prompt contains data from another experiment."""


class ReadOnlyViolation(MacroFail):
    """ai_docs changed during the run."""


# ---- Infrastructure ------------------------------------------------------------------------
class InfraUnavailable(FactoryError):
    """The LLM transport stayed unavailable after the full backoff schedule."""
