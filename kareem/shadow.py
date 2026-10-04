"""Shadow-mode logging and analysis.

Shadow mode = run the policy next to a live agent without acting on its
recommendations, then study the log before enabling anything. This is the rollout
path the policy card prescribes, and the way to pick a `min_confidence` threshold
from real data instead of guessing.

Log format: one JSON object per line (JSONL):

    {"ts": ..., "session": "...", "step": 3, "obs": {...},
     "action": "COMPACT", "confidence": 0.41, "probabilities": {...},
     "applied": "NOOP"}                     # what the agent harness actually did

CLI:

    python -m kareem.shadow report <log.jsonl> [--thresholds 0.25 0.3 0.35 0.4]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .actions import ACTION_NAMES, Action


class ShadowLog:
    """Appends one record per agent step to a JSONL file."""

    def __init__(self, path: str | Path, session: str = "default"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session = session

    def record(self, step: int, obs: dict, decision, applied: Action | str | int | None = None) -> None:
        """`decision` is a runtime.Decision, or None when the policy was not consulted
        this step (e.g. the cheap pre-filter skipped it)."""
        rec = {
            "ts": round(time.time(), 3),
            "session": self.session,
            "step": int(step),
            "obs": obs,
            "action": None,
            "confidence": None,
            "probabilities": None,
            "applied": None if applied is None else (Action(applied).name if not isinstance(applied, str) else applied),
        }
        if decision is not None:
            rec["action"] = decision.action_name
            rec["confidence"] = decision.confidence
            rec["probabilities"] = decision.probabilities
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def load_log(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def report(path: str | Path, thresholds: list[float] | None = None) -> str:
    """Summarise a shadow log: action histogram, confidence stats, and a threshold
    sweep showing how often the policy would have intervened at each min_confidence.

    The threshold sweep is the practical output: pick the lowest threshold whose
    intervention rate you can afford, then pass it as `min_confidence` to decide().
    """
    thresholds = thresholds or [0.25, 0.30, 0.35, 0.40, 0.50]
    records = load_log(path)
    decided = [r for r in records if r.get("action") is not None]
    lines = [f"shadow log: {path}", f"steps logged: {len(records)}, policy decisions: {len(decided)}"]
    if not decided:
        lines.append("no policy decisions recorded yet (pre-filter skipped all steps?)")
        return "\n".join(lines)

    hist = {a: 0 for a in ACTION_NAMES}
    confs = []
    interventions = 0
    for r in decided:
        hist[r["action"]] += 1
        confs.append(r["confidence"])
        if r["action"] != "NOOP":
            interventions += 1
    lines.append("")
    lines.append("action histogram:")
    for a in ACTION_NAMES:
        if hist[a]:
            lines.append(f"  {a:24s} {hist[a]:4d}  ({hist[a] / len(decided):.0%})")
    confs.sort()
    lines.append("")
    lines.append(
        f"confidence: min={confs[0]:.2f} median={confs[len(confs) // 2]:.2f} max={confs[-1]:.2f}"
    )
    lines.append(f"intervention rate (action != NOOP): {interventions / len(decided):.0%}")
    lines.append("")
    lines.append("threshold sweep — % of steps where the policy would intervene at min_confidence:")
    for t in thresholds:
        n = sum(1 for r in decided if r["action"] != "NOOP" and r["confidence"] >= t)
        lines.append(f"  {t:.2f} -> {n:4d} steps ({n / len(decided):.0%})")

    applied = [r for r in decided if r.get("applied") is not None]
    if applied:
        agree = sum(1 for r in applied if r["applied"] == r["action"])
        lines.append("")
        lines.append(f"agreement with what the harness actually did: {agree}/{len(applied)} ({agree / len(applied):.0%})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kareem-shadow")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("report", help="summarise a shadow log")
    rep.add_argument("log", help="path to a shadow .jsonl log")
    rep.add_argument("--thresholds", type=float, nargs="*", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "report":
        print(report(args.log, args.thresholds))


if __name__ == "__main__":
    main()
