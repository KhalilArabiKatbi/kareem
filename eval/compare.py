"""Compare Kareem (replay) against the Opus causal judge on the same checkpoints.

  python -m eval.compare [--replay replay_est]

Writes eval/out/compare.json and prints agreement, confusion matrix, per-action precision/recall,
agreement by utilization bucket, and the disagreements with the judge's reasons.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
ACTIONS = ["NOOP", "COMPACT", "PRUNE_TOOLS", "REINJECT_INSTRUCTIONS", "CHECKPOINT_RESET", "RETRIEVE_MEMORY"]


def load(replay: str):
    rows = {}
    for f in (OUT / replay).glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows[(f.stem, r["i"])] = r
    pairs = []
    for f in (OUT / "judge").glob("*.json"):
        j = json.loads(f.read_text(encoding="utf-8"))
        lab = j.get("label") or {}
        k = (j["session"], j["step"])
        if lab.get("action") and k in rows:
            pairs.append((k, rows[k], lab))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default="replay_est")
    a = ap.parse_args()
    pairs = load(a.replay)
    n = len(pairs)
    if not n:
        print("no pairs")
        return
    conf = defaultdict(Counter)  # conf[judge][kareem]
    buckets = defaultdict(lambda: [0, 0])
    for _, k, j in pairs:
        conf[j["action"]][k["effective"]] += 1
        b = min(int(k["util"] * 5), 4)
        buckets[b][1] += 1
        buckets[b][0] += j["action"] == k["effective"]
    agree = sum(conf[x][x] for x in ACTIONS)
    print(f"{n} checkpoints | exact agreement {agree}/{n} = {agree / n:.1%}")
    print(f"judge action share: { {x: sum(conf[x].values()) for x in ACTIONS if sum(conf[x].values())} }")
    kc = Counter(k["effective"] for _, k, _ in pairs)
    print(f"kareem action share: {dict(kc)}")
    print("\nconfusion (rows=judge, cols=kareem):")
    print(" " * 24 + "".join(f"{x[:7]:>9}" for x in ACTIONS))
    for x in ACTIONS:
        print(f"{x:24}" + "".join(f"{conf[x][y]:9d}" for y in ACTIONS))
    print("\nper action (treat judge as reference):")
    per = {}
    for x in ACTIONS:
        tp = conf[x][x]
        fn = sum(conf[x].values()) - tp
        fp = sum(conf[y][x] for y in ACTIONS) - tp
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        per[x] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r}
        if tp + fp + fn:
            print(f"  {x:24} tp={tp:3} fp={fp:3} fn={fn:3} P={p if p is None else round(p, 2)} R={r if r is None else round(r, 2)}")
    print("\nagreement by utilization bucket:")
    for b in sorted(buckets):
        ok, tot = buckets[b]
        print(f"  util {b * 0.2:.1f}-{b * 0.2 + 0.2:.1f}: {ok}/{tot} = {ok / tot:.0%}")
    # intervention-level view: did both agree that *something* should happen?
    jy = sum(j["action"] != "NOOP" for _, _, j in pairs)
    ky = sum(k["effective"] != "NOOP" for _, k, _ in pairs)
    both = sum(j["action"] != "NOOP" and k["effective"] != "NOOP" for _, k, j in pairs)
    print(f"\nintervene? judge={jy} kareem={ky} both={both}")
    print("\nmissed by kareem (judge wants action, kareem NOOP):")
    miss = [(key, k, j) for key, k, j in pairs if j["action"] != "NOOP" and k["effective"] == "NOOP"]
    for (s, st), k, j in miss[:25]:
        print(f"  {s[:8]} step {st} util={k['util']:.2f} streak={k['err_streak']} stale={k['stale']:.2f} red={k['redundancy']:.2f} "
              f"judge={j['action']}({j['confidence']:.2f}): {j['reason'][:140]}")
    (OUT / "compare.json").write_text(json.dumps({"n": n, "agreement": agree / n, "confusion": {x: dict(conf[x]) for x in ACTIONS},
                                                  "per_action": per}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
