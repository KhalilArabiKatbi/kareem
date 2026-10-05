"""Summarise the A/B results: per arm mean score, cost, peak tokens, resets; paired differences vs native.

  python -m eval.ab.report
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "out" / "ab" / "results.jsonl"


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def sd(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else float("nan")


def main() -> None:
    rows = [json.loads(line) for line in RES.read_text(encoding="utf-8").splitlines()]
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    print(f"{len(rows)} runs\n")
    print(f"{'arm':10} {'n':>3} {'score':>11} {'behav':>6} {'recall':>7} {'conv':>5} {'cost$':>7} {'peakK':>7} {'resets':>7} {'acts/run':>9}")
    for arm in ("native", "threshold", "kareem"):
        rs = by.get(arm, [])
        if not rs:
            continue
        sc = [r["score"]["total"] for r in rs]
        acts = [sum(t["actuated"] for t in r["turn_log"]) for r in rs]
        print(f"{arm:10} {len(rs):3d} {mean(sc):6.2f}±{sd(sc):4.2f} {mean([r['score']['behavior'] for r in rs]):6.2f} "
              f"{mean([r['score']['recall'] for r in rs]):7.2f} {mean([r['score']['convention'] for r in rs]):5.2f} "
              f"{mean([r['cost'] for r in rs]):7.2f} {mean([r['peak_tokens'] for r in rs]) / 1000:7.1f} "
              f"{mean([r['resets'] for r in rs]):7.2f} {mean(acts):9.2f}")
    # paired by (seed, rep) against native
    key = lambda r: (r["seed"], r.get("rep", 0))  # noqa: E731
    nat = {key(r): r for r in by.get("native", [])}
    print("\npaired vs native (same task, same repeat):")
    for arm in ("threshold", "kareem"):
        d_s, d_c = [], []
        for r in by.get(arm, []):
            n = nat.get(key(r))
            if n:
                d_s.append(r["score"]["total"] - n["score"]["total"])
                d_c.append(r["cost"] - n["cost"])
        if d_s:
            se = lambda x: sd(x) / math.sqrt(len(x)) if len(x) > 1 else float("nan")  # noqa: E731
            print(f"  {arm:10} n={len(d_s)}  d_score={mean(d_s):+.2f} (se {se(d_s):.2f})  d_cost=${mean(d_c):+.2f} (se {se(d_c):.2f})")
    print("\nKareem action mix across turns:")
    mix = defaultdict(int)
    for r in by.get("kareem", []):
        for t in r["turn_log"]:
            mix["NOOP" if t["deferred"] else t["action"]] += 1
    print("  ", dict(mix))
    low = [(r["arm"], r["seed"], r.get("rep", 0), r["score"]) for r in rows if r["score"]["total"] < r["score"]["max"]]
    print(f"\nruns below max score ({len(low)}):")
    for x in low[:20]:
        print("  ", x)


if __name__ == "__main__":
    main()
