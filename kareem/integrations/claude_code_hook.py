"""Claude Code hook: context-health policy in shadow/recommend mode.

Runs inside Claude Code's hook lifecycle (https://code.claude.com/docs/en/hooks).
Reads the hook event JSON from stdin, tracks context telemetry per session, consults
the intervention policy, and — when the policy is confident — injects a recommendation
into Claude's context via hookSpecificOutput.additionalContext.

What this hook CAN do: observe real usage, log decisions for shadow analysis, and
recommend interventions (e.g. "run /compact now").
What it CANNOT do: actuate them. Hooks cannot trigger compaction, prune tool outputs,
or rewrite history — that needs the Agent SDK wrapper. This is deliberate: shadow
mode first, actuation after the recommendations check out (see policy card caveats).

Events handled:
  SessionStart          -> reset per-session state (source "compact"/"clear" = fresh context)
  PostToolUse           -> observe a successful step, maybe decide + recommend
  PostToolUseFailure    -> same, with last_error=1

Performance: the common path (early session, low utilization, no errors) is pure
stdlib — torch and the policy are only imported once utilization crosses
KAREEM_MIN_UTIL or an error streak forms, keeping per-tool-call overhead near zero
early in a session.

Config via environment variables:
  KAREEM_DISABLED=1        turn the hook off entirely
  KAREEM_CAPACITY_TOKENS   context window of the model (default 200000)
  KAREEM_MIN_CONFIDENCE    min softmax confidence to emit a recommendation (default 0.35)
  KAREEM_MIN_UTIL          utilization floor for consulting the policy (default 0.25)
  KAREEM_STATE_DIR         state + shadow logs (default ~/.kareem)

settings.json snippet:

  {
    "hooks": {
      "SessionStart":        [{"hooks": [{"type": "command", "timeout": 10, "command": "python -m kareem.integrations.claude_code_hook"}]}],
      "PostToolUse":         [{"matcher": "*", "hooks": [{"type": "command", "timeout": 10, "command": "python -m kareem.integrations.claude_code_hook"}]}],
      "PostToolUseFailure":  [{"matcher": "*", "hooks": [{"type": "command", "timeout": 10, "command": "python -m kareem.integrations.claude_code_hook"}]}]
    }
  }

(Python must be the interpreter where the `kareem` package + torch are installed.)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_RECOMMEND_COOLDOWN_STEPS = 8

# additionalContext text per action. Written as guidance to Claude, since the hook
# cannot perform the intervention itself.
_RECOMMENDATIONS = {
    "COMPACT": "Context-health monitor: context utilization is high and still growing. Finish your current atomic step, then ask the user to run /compact (or do so at the next natural boundary) before starting new subtasks.",
    "PRUNE_TOOLS": "Context-health monitor: tool output is dominating the context. Avoid re-reading large outputs you already consumed; refer to your earlier summaries of them instead of pasting content again.",
    "REINJECT_INSTRUCTIONS": "Context-health monitor: the session is many steps from its original instructions. In your next message, briefly restate the current task goal and its constraints to re-anchor yourself before proceeding.",
    "CHECKPOINT_RESET": "Context-health monitor: context integrity is degraded. Summarise the essential state (goal, decisions, file paths, errors) into a short checkpoint and suggest starting a fresh session from it.",
    "RETRIEVE_MEMORY": "Context-health monitor: an earlier fact appears to be missing. Before continuing, re-read the relevant file or notes rather than guessing from context.",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _state_dir() -> Path:
    d = Path(os.environ.get("KAREEM_STATE_DIR", Path.home() / ".kareem"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fresh_state() -> dict:
    return {
        "history": [],
        "since_instr": 0,
        "since_compact": 0,
        "error_streak": 0,
        "last_action": 0,
        "prev_tokens": 0,
        "tool_tokens": 0,
        "last_emit_step": -999,
        "last_emit_action": None,
    }


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return _fresh_state()


def _save_state(path: Path, state: dict) -> None:
    try:
        path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def _context_tokens(transcript_path: str) -> int | None:
    """Current context size from the transcript: the last assistant message's usage
    (input + cache reads + cache writes) is the size of the context that call saw."""
    try:
        text = Path(transcript_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    last = None
    for line in text.splitlines():
        if '"usage"' not in line or '"assistant"' not in line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = (rec.get("message") or {}).get("usage")
        if usage:
            last = usage
    if last is None:
        return None
    return int(
        last.get("input_tokens", 0)
        + last.get("cache_read_input_tokens", 0)
        + last.get("cache_creation_input_tokens", 0)
    )


def _handle_session_start(event: dict) -> None:
    # startup/resume/clear/compact: in all cases the effective context is fresh or
    # re-anchored, so per-session tracking starts over.
    state_path = _state_dir() / "sessions" / f"{event.get('session_id', 'unknown')}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _save_state(state_path, _fresh_state())


def _handle_tool_event(event: dict, failed: bool) -> None:
    from kareem.actions import Action
    from kareem.observe import build_observation
    from kareem.shadow import ShadowLog

    session = event.get("session_id", "unknown")
    state_path = _state_dir() / "sessions" / f"{session}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_path)

    capacity = _env_int("KAREEM_CAPACITY_TOKENS", 200_000)
    tokens = _context_tokens(event.get("transcript_path", "")) or state["prev_tokens"]
    response = event.get("tool_response")
    if not isinstance(response, dict):  # harness may pass a bare string
        response = {"output": "" if response is None else str(response)}
    tool_out = response.get("output") or response.get("error") or ""
    state["tool_tokens"] += max(1, len(str(tool_out)) // 4)

    step = len(state["history"])
    if step > 0:
        state["since_instr"] += 1
        state["since_compact"] += 1
    state["error_streak"] = state["error_streak"] + 1 if failed else 0

    obs = build_observation(
        step=step,
        horizon=40,
        tokens_used=tokens / 1000.0,
        capacity=capacity / 1000.0,
        tokens_added_last=(tokens - state["prev_tokens"]) / 1000.0,
        tool_tokens=state["tool_tokens"] / 1000.0,
        steps_since_instruction=state["since_instr"],
        steps_since_compaction=state["since_compact"],
        last_error=int(failed),
        error_streak=state["error_streak"],
        last_action=state["last_action"],
        last_success=0 if failed else 1,
        confidence=0.5,  # unknowable from the hook surface
        progress=0,      # unknowable from the hook surface
    )
    state["history"].append(obs)
    state["prev_tokens"] = tokens

    log = ShadowLog(_state_dir() / "shadow" / f"{session}.jsonl", session=session)

    # Cheap path: skip the policy entirely when nothing interesting is happening.
    util = obs["utilization"]
    if util < _env_float("KAREEM_MIN_UTIL", 0.25) and state["error_streak"] == 0:
        log.record(step, obs, decision=None, applied=state["last_action"])
        _save_state(state_path, state)
        return

    # Slow path: consult the policy (imports torch — ~1-2s once util is high).
    from kareem import load_policy

    decision = load_policy().decide(state["history"], min_confidence=_env_float("KAREEM_MIN_CONFIDENCE", 0.35))
    # applied stays None: in recommend mode the hook can't observe what the harness did
    log.record(step, obs, decision=decision)

    emit = (
        decision.action is not Action.NOOP
        and not decision.deferred
        and (step - state["last_emit_step"] >= _RECOMMEND_COOLDOWN_STEPS or decision.action_name != state["last_emit_action"])
    )
    if emit:
        state["last_emit_step"] = step
        state["last_emit_action"] = decision.action_name
        if decision.action is Action.REINJECT_INSTRUCTIONS:
            state["since_instr"] = 0  # the emitted nudge serves as the re-injection
        _save_state(state_path, state)
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": event.get("hook_event_name", "PostToolUse"),
                "additionalContext": _RECOMMENDATIONS[decision.action_name],
            },
            "suppressOutput": True,
        }))
        return

    _save_state(state_path, state)


def main() -> None:
    try:
        if os.environ.get("KAREEM_DISABLED") == "1":
            return
        event = json.load(sys.stdin)
        name = event.get("hook_event_name", "")
        if name == "SessionStart":
            _handle_session_start(event)
        elif name == "PostToolUse":
            _handle_tool_event(event, failed=False)
        elif name == "PostToolUseFailure":
            _handle_tool_event(event, failed=True)
    except Exception:
        # A hook must never break the user's session: swallow everything, exit 0.
        pass


if __name__ == "__main__":
    main()
