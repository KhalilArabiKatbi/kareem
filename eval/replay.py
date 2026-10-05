"""Replay real Claude Code transcripts through Kareem (no LLM calls).

One step = one tool_result record, mirroring the PostToolUse hook: context size comes from
the latest assistant `usage`, tool tokens accumulate from result sizes, errors from
`is_error`. A usage drop > 50% (or a compact boundary) resets tracking, like SessionStart.

  .venv/Scripts/python.exe eval/replay.py [--min-size 500000] [--capacity 200000] [--workers 12]

Output (gitignored): eval/out/replay/<session>.jsonl + eval/out/replay_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "eval" / "out"
PROJECTS = Path.home() / ".claude" / "projects"
MIN_CONF = 0.35
MIN_UTIL = 0.25


def find_sessions(min_size: int) -> list[Path]:
    out = []
    for p in PROJECTS.glob("*/*.jsonl"):
        if "claude-mem" in p.parent.name or p.stat().st_size < min_size:
            continue
        out.append(p)
    return sorted(out, key=lambda p: -p.stat().st_size)


def extract_steps(path: Path) -> list[dict]:
    from eval.transcript import parse, steps_with_estimates
    return steps_with_estimates(parse(path))


def replay(args: tuple[str, int, bool]) -> dict:
    path, capacity = Path(args[0]), args[1]
    rescale, min_conf = args[3], args[4]
    tag = ("est" if args[2] else "zero") + ("_rescale" if rescale else "") + (f"_c{min_conf}" if min_conf != MIN_CONF else "")
    f = 128_000 / capacity if rescale else 1.0   # map real units into the trained 128k range
    if rescale:
        capacity = 128_000
    from kareem import load_policy
    from kareem.observe import SessionTracker
    from kareem.actions import Action

    policy = load_policy()
    tr = SessionTracker(capacity_tokens=capacity)
    steps = extract_steps(path)
    out_dir = OUT / f"replay_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, rows, prev, tool_tok = Counter(), [], 0, 0
    for i, s in enumerate(steps):
        if s["compact_before"] or (prev > 50_000 and s["tokens"] < 0.5 * prev):
            tr.reset(); tool_tok = 0
        prev = s["tokens"]
        tool_tok += max(1, s["chars"] // 4)
        obs = tr.observe(tokens_used=int(s["tokens"] * f), tool_tokens=int(min(tool_tok, s["tokens"]) * f),
                         last_error=s["error"], last_success=not s["error"],
                         redundancy_est=s["redundancy_est"] if args[2] else 0.0, stale_est=s["stale_est"] if args[2] else 0.0)
        if rescale:  # keep counters inside the 40-step training horizon
            for k in ("step", "steps_since_instruction", "steps_since_compaction", "error_streak"):
                obs[k] = min(obs[k], obs["horizon"] - 1)
        if obs["utilization"] < MIN_UTIL and obs["error_streak"] == 0:
            action, conf, deferred, probs = "NOOP", None, True, None
        else:
            d = policy.decide(tr.history, min_confidence=min_conf)
            action, conf, deferred, probs = d.action_name, d.confidence, d.deferred, d.probabilities
        eff = "NOOP" if deferred else action
        counts[eff] += 1
        rows.append({"i": i, "ts": s["ts"], "uuid": s["uuid"], "tokens": s["tokens"],
                     "util": round(obs["utilization"], 4), "stale": obs["stale_est"], "redundancy": obs["redundancy_est"], "err_streak": obs["error_streak"],
                     "action": action, "effective": eff, "conf": conf, "probs": probs})
        tr.applied(Action.NOOP)  # shadow mode: nothing actuated
    (out_dir / f"{path.stem}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return {"session": path.stem, "project": path.parent.name, "steps": len(steps), "mean_stale": round(sum(r["stale"] for r in rows) / max(1, len(rows)), 3),
            "max_tokens": max((s["tokens"] for s in steps), default=0),
            "max_util": max((r["util"] for r in rows), default=0), "effective": dict(counts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-size", type=int, default=500_000)
    ap.add_argument("--capacity", type=int, default=200_000)
    ap.add_argument("--estimates", type=int, default=1)
    ap.add_argument("--rescale", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=MIN_CONF)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    sessions = find_sessions(a.min_size)
    print(f"{len(sessions)} sessions")
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(replay, [(str(p), a.capacity, bool(a.estimates), bool(a.rescale), a.min_conf) for p in sessions]))
    tot = Counter()
    for r in res:
        tot.update(r["effective"])
    n = sum(tot.values())
    summary = {"sessions": res, "total_steps": n, "action_share": {k: round(v / n, 4) for k, v in tot.items()} if n else {}}
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / f"replay_summary_{'est' if a.estimates else 'zero'}{'_rescale' if a.rescale else ''}_c{a.min_conf}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("steps", n, "share", summary["action_share"])
    for r in res[:15]:
        print(f'{r["project"][:28]:28} {r["session"][:8]} steps={r["steps"]:4} maxtok={r["max_tokens"]:7} maxutil={r["max_util"]:.2f} {r["effective"]}')


if __name__ == "__main__":
    main()
