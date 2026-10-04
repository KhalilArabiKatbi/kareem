"""Claude Agent SDK wrapper with REAL actuation — the step beyond the hook.

The Claude Code hook (claude_code_hook.py) can only recommend interventions. This
wrapper owns the agent loop via the Claude Agent SDK, so it can actually perform them
at turn boundaries:

  NOOP                   nothing
  REINJECT_INSTRUCTIONS  original instructions are prepended to the next prompt
  RETRIEVE_MEMORY        content from the caller's `memory` callable is prepended
  COMPACT / CHECKPOINT_RESET
                         the current session is asked for a continuation summary,
                         then a FRESH session is started with that summary as a
                         checkpoint (the SDK cannot edit a live session's history,
                         so compaction = checkpoint + new session)
  PRUNE_TOOLS            not actuable through the SDK (no history editing); degrades
                         to a "don't re-read old outputs" instruction. Honest no-op
                         for the context, signal-only.

Usage:

    import asyncio
    from claude_agent_sdk import ClaudeAgentOptions
    from kareem.integrations.claude_agent import GuardedAgent

    async def main():
        async with GuardedAgent(
            options=ClaudeAgentOptions(allowed_tools=["Read", "Bash"], cwd="."),
            instructions="<the original task instructions>",
            min_confidence=0.35,
        ) as agent:
            async for msg in agent.run_turn("Fix the failing test in parser.py"):
                ...                                # stream SDK messages to your UI
            async for msg in agent.run_turn("Now add a regression test"):
                ...

    asyncio.run(main())

Telemetry per turn comes from the SDK message stream: token usage from the
ResultMessage's usage dict when present (char/4 estimate otherwise), tool-output
tokens and error signals from ToolResultBlocks. Progress is caller-supplied via
`progress_callback` (None -> 0, matching the hook).

Requires: pip install claude-agent-sdk  (lazy-imported; `import kareem` alone never
pulls it in). Pass `client_factory` to substitute a stub for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

from ..actions import Action
from ..observe import SessionTracker
from ..runtime import InterventionPolicy, Decision
from ..shadow import ShadowLog

_SUMMARY_PROMPT = (
    "Summarize this session for continuation in a fresh context. Cover: the task goal, "
    "decisions made and why, current state (files touched, errors seen), and the exact "
    "next step. Be terse; this summary IS the new context."
)

_PRUNE_NOTE = (
    "[Context-health monitor] Tool output is dominating the context. Do not re-read "
    "large outputs you already consumed; refer to your earlier summaries instead."
)

_REINJECT_PREFIX = (
    "[Context-health monitor] Re-anchoring: the original task instructions follow. "
    "Re-read them before continuing.\n\n"
)


@dataclass
class TurnRecord:
    """What the policy saw and did around one turn."""
    obs: dict
    decision: Decision
    actuated: bool


class GuardedAgent:
    def __init__(
        self,
        *,
        options: Any = None,
        policy: InterventionPolicy | None = None,
        instructions: str = "",
        memory: Callable[[], str | Awaitable[str]] | None = None,
        min_confidence: float = 0.35,
        capacity_tokens: int = 200_000,
        horizon: int = 40,
        progress_callback: Callable[[], int] | None = None,
        shadow_log_path: str | None = None,
        client_factory: Callable[..., Any] | None = None,
    ):
        if policy is None:
            from .. import load_policy
            policy = load_policy()
        if client_factory is None:
            try:
                from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
            except ImportError as e:
                raise ImportError(
                    "GuardedAgent needs the Claude Agent SDK: pip install claude-agent-sdk"
                ) from e
            client_factory = ClaudeSDKClient
            if options is None:
                options = ClaudeAgentOptions()
        self.options = options
        self.policy = policy
        self.instructions = instructions
        self.memory = memory
        self.min_confidence = min_confidence
        self.progress_callback = progress_callback
        self._client_factory = client_factory
        self._client: Any = None
        self.tracker = SessionTracker(capacity_tokens=capacity_tokens, horizon=horizon)
        self.shadow = ShadowLog(shadow_log_path, session="sdk") if shadow_log_path else None
        self.turns: list[TurnRecord] = []
        self._prev_tokens = 0
        self._tool_tokens = 0
        self._last_error = False
        self.session_resets = 0  # how many times the policy checkpoint-reset the context

    # -- lifecycle -----------------------------------------------------------

    async def __aenter__(self) -> "GuardedAgent":
        self._client = self._client_factory(self.options)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.__aexit__(*exc)
            self._client = None

    # -- main entry ----------------------------------------------------------

    async def run_turn(self, prompt: str) -> AsyncIterator[Any]:
        """Observe -> decide -> actuate -> run one user turn, yielding SDK messages."""
        obs = self._observe()
        decision = self.policy.decide(self.tracker.history, min_confidence=self.min_confidence)
        actuated = False

        if not decision.deferred and decision.action is not Action.NOOP:
            prompt, actuated = await self._actuate(decision.action, prompt)

        if actuated and decision.action in (Action.COMPACT, Action.CHECKPOINT_RESET):
            self.tracker.reset()  # fresh context: old history no longer describes it
        else:
            self.tracker.applied(decision.action if actuated else Action.NOOP)

        self.turns.append(TurnRecord(obs=obs, decision=decision, actuated=actuated))
        if self.shadow:
            self.shadow.record(obs["step"], obs, decision, applied=decision.action if actuated else Action.NOOP)

        await self._client.query(prompt)
        async for msg in self._client.receive_response():
            self._absorb(msg)
            yield msg

    # -- telemetry -----------------------------------------------------------

    def _observe(self) -> dict:
        return self.tracker.observe(
            tokens_used=self._prev_tokens,
            tool_tokens=self._tool_tokens,
            last_error=self._last_error,
            last_success=not self._last_error,
            progress=self.progress_callback() if self.progress_callback else 0,
        )

    def _absorb(self, msg: Any) -> None:
        """Pull telemetry out of one SDK message (duck-typed, so stubs work in tests)."""
        usage = getattr(msg, "usage", None)
        if isinstance(usage, dict):
            self._prev_tokens = int(
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )
        content = getattr(msg, "content", None) or []
        saw_error = False
        for block in content:
            if getattr(block, "tool_use_id", None) is not None:  # ToolResultBlock
                self._tool_tokens += max(1, len(str(getattr(block, "content", ""))) // 4)
                saw_error = saw_error or bool(getattr(block, "is_error", False))
            text = getattr(block, "text", None)
            if text and usage is None:  # rough fallback when no usage dict is available
                self._prev_tokens += len(text) // 4
        if saw_error:
            self._last_error = True
        elif getattr(msg, "duration_ms", None) is not None:  # ResultMessage closes a turn
            if not saw_error:
                self._last_error = False

    # -- actuation -----------------------------------------------------------

    async def _actuate(self, action: Action, prompt: str) -> tuple[str, bool]:
        if action is Action.REINJECT_INSTRUCTIONS and self.instructions:
            return _REINJECT_PREFIX + self.instructions + "\n\n" + prompt, True
        if action is Action.RETRIEVE_MEMORY and self.memory is not None:
            content = self.memory()
            if hasattr(content, "__await__"):
                content = await content
            if content:
                return f"[Context-health monitor] Retrieved memory:\n{content}\n\n" + prompt, True
            return prompt, False
        if action is Action.PRUNE_TOOLS:
            return _PRUNE_NOTE + "\n\n" + prompt, True
        if action in (Action.COMPACT, Action.CHECKPOINT_RESET):
            summary = await self._summarize_session()
            self.session_resets += 1
            return (
                "[Context checkpoint from the previous session — treat it as the full prior state]\n\n"
                + summary
                + "\n\n[Continue the task]\n\n"
                + prompt
            ), True
        return prompt, False

    async def _summarize_session(self) -> str:
        """Ask the live session to summarize itself, then replace it with a fresh one."""
        summary_parts: list[str] = []
        try:
            await self._client.query(_SUMMARY_PROMPT)
            async for msg in self._client.receive_response():
                for block in getattr(msg, "content", None) or []:
                    text = getattr(block, "text", None)
                    if text:
                        summary_parts.append(text)
        except Exception:
            summary_parts.append("(summary step failed; continue with partial context)")
        summary = "\n".join(summary_parts).strip() or "(empty summary)"
        # swap to a fresh session
        await self._client.__aexit__(None, None, None)
        self._client = self._client_factory(self.options)
        await self._client.__aenter__()
        return summary
