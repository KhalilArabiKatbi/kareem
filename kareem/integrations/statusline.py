"""Claude Code status line: show what Kareem currently recommends, under the chat.

Claude Code runs this command after each assistant message, feeding session info as JSON on stdin,
and shows the first line of stdout. The line is built from the shadow log the hook already writes
(`~/.kareem/shadow/<session>.jsonl`), so it is cheap (stdlib only, no torch). If the hook has not
logged anything yet, it falls back to reading context usage straight from the transcript.

    Kareem  34% ctx  OK
    Kareem  71% ctx  PRUNE_TOOLS 0.52  drop old tool output
    Kareem  off

settings.json:

    {"statusLine": {"type": "command", "command": "python -m kareem.integrations.statusline"}}

Toggle (also honoured by the hook): `python -m kareem.integrations.statusline toggle [on|off|status]`
or the `kareem-toggle` console script. Off = a flag file `~/.kareem/disabled`; `KAREEM_DISABLED=1` also works.

Env: KAREEM_STATE_DIR, KAREEM_CAPACITY_TOKENS (default 200000), KAREEM_MIN_CONFIDENCE (default 0.35).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ADVICE = {
    "COMPACT": "compact soon",
    "PRUNE_TOOLS": "drop old tool output",
    "REINJECT_INSTRUCTIONS": "restate the task",
    "CHECKPOINT_RESET": "checkpoint + fresh session",
    "RETRIEVE_MEMORY": "re-read lost facts",
}
_RESET, _DIM, _GREEN, _YELLOW, _RED = "\033[0m", "\033[2m", "\033[32m", "\033[33m", "\033[31m"


def state_dir() -> Path:
    return Path(os.environ.get("KAREEM_STATE_DIR", Path.home() / ".kareem"))


def is_disabled() -> bool:
    return os.environ.get("KAREEM_DISABLED") == "1" or (state_dir() / "disabled").exists()


def _last_record(path: Path) -> dict | None:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            tail = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _transcript_util(transcript_path: str) -> float | None:
    from .claude_code_hook import _context_tokens  # stdlib-only helper shared with the hook
    tokens = _context_tokens(transcript_path) if transcript_path else None
    if tokens is None:
        return None
    return tokens / float(os.environ.get("KAREEM_CAPACITY_TOKENS", 200_000))


def render(event: dict) -> str:
    if is_disabled():
        return f"{_DIM}Kareem off{_RESET}"
    session = event.get("session_id", "")
    rec = _last_record(state_dir() / "shadow" / f"{session}.jsonl") if session else None
    util = None
    action, conf = None, None
    if rec:
        util = (rec.get("obs") or {}).get("utilization")
        action, conf = rec.get("action"), rec.get("confidence")
    if util is None:
        util = _transcript_util(event.get("transcript_path", ""))
    if util is None:
        return f"{_DIM}Kareem waiting for first tool call{_RESET}"
    pct = f"{util * 100:.0f}% ctx"
    min_conf = float(os.environ.get("KAREEM_MIN_CONFIDENCE", 0.35))
    if action and action != "NOOP" and conf is not None and conf >= min_conf:
        colour = _RED if action in ("COMPACT", "CHECKPOINT_RESET") and util >= 0.8 else _YELLOW
        return f"Kareem  {pct}  {colour}{action} {conf:.2f}{_RESET}  {_DIM}{_ADVICE.get(action, '')}{_RESET}"
    return f"Kareem  {pct}  {_GREEN}OK{_RESET}"


def main() -> None:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
        print(render(event))
    except Exception:  # a status line must never error
        print("Kareem ?")


def toggle_main(argv: list[str] | None = None) -> None:
    arg = (argv if argv is not None else sys.argv[1:])
    mode = arg[0] if arg else "status"
    flag = state_dir() / "disabled"
    if mode == "off":
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("disabled by kareem-toggle\n", encoding="utf-8")
    elif mode == "on":
        flag.unlink(missing_ok=True)
    elif mode != "status":
        print("usage: kareem-toggle [on|off|status]")
        return
    print("Kareem is " + ("OFF" if is_disabled() else "ON"))


if __name__ == "__main__":
    if sys.argv[1:2] == ["toggle"]:
        toggle_main(sys.argv[2:])
    else:
        main()
