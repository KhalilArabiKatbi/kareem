"""Are Kareem's input features enough to predict what Opus wants? Leave-one-session-out CV.

Target: judge says intervene (action != NOOP). Feature sets:
  util_only : utilization alone (a threshold heuristic)
  kareem    : what the MLP can see (util, stale, redundancy, error streak, tool_fraction, step)
  kareem+   : kareem + candidate new features (big-result counts, result bulk, steps since compact, error rate)

Logistic regression (numpy, L2). Metrics: ROC-AUC and precision/recall at the best-F1 threshold.

  python -m eval.diagnose
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.transcript import parse, steps_with_estimates  # noqa: E402
from eval.replay import PROJECTS  # noqa: E402

OUT = ROOT / "eval" / "out"
KAREEM = ["util", "stale", "redundancy", "err_streak", "tool_fraction", "step"]
EXTRA = ["big_recent", "bulk_recent", "since_compact", "err_rate", "log_chars_last"]


def session_path(stem: str) -> Path:
    return next(PROJECTS.glob(f"*/{stem}.jsonl"))


def features(stem: str) -> dict[int, dict]:
    ev = parse(session_path(stem))
    st = steps_with_estimates(ev)
    out, tool_tok, last_compact, err_run = {}, 0, 0, 0
    for i, s in enumerate(st):
        if s["compact_before"]:
            last_compact, tool_tok = i, 0
        tool_tok += max(1, s["chars"] // 4)
        err_run = err_run + 1 if s["error"] else 0
        win = st[max(0, i - 39): i + 1]
        tokens = max(1, s["tokens"])
        out[i] = {"util": tokens / 1e6, "stale": s["stale_est"], "redundancy": s["redundancy_est"],
                  "err_streak": min(err_run, 39), "tool_fraction": min(1.0, tool_tok / tokens), "step": min(i, 39),
                  "big_recent": sum(w["chars"] > 50_000 for w in win) / 10,
                  "bulk_recent": min(1.0, sum(w["chars"] for w in win) / 4 / tokens),
                  "since_compact": min(i - last_compact, 400) / 100,
                  "err_rate": sum(w["error"] for w in win) / len(win),
                  "log_chars_last": math.log10(1 + s["chars"])}
    return out


def fit(X, y, l2=1.0, iters=800, lr=0.3):
    w = np.zeros(X.shape[1]); b = 0.0
    wts = np.where(y == 1, 0.5 / max(1, y.sum()) * len(y), 0.5 / max(1, (1 - y).sum()) * len(y))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = (p - y) * wts
        w -= lr * (X.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return w, b


def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def best_f1(y, s):
    best = (0, 0, 0, 0)
    for t in np.unique(s):
        pred = s >= t
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f = 2 * p * r / (p + r) if p + r else 0
        if f > best[0]:
            best = (f, p, r, t)
    return best


def main() -> None:
    rows = []
    cache: dict[str, dict] = {}
    for f in (OUT / "judge").glob("*.json"):
        j = json.loads(f.read_text(encoding="utf-8"))
        lab = j.get("label") or {}
        if not lab.get("action"):
            continue
        if j["session"] not in cache:
            cache[j["session"]] = features(j["session"])
        ft = cache[j["session"]].get(j["step"])
        if ft:
            rows.append((j["session"], ft, lab["action"]))
    sess = np.array([r[0] for r in rows])
    y = np.array([r[2] != "NOOP" for r in rows], dtype=float)
    print(f"{len(rows)} checkpoints, {int(y.sum())} intervene ({y.mean():.0%}), {len(set(sess))} sessions")
    sets = {"util_only": ["util"], "kareem": KAREEM, "kareem+": KAREEM + EXTRA}
    for name, cols in sets.items():
        X = np.array([[r[1][c] for c in cols] for r in rows], dtype=float)
        scores = np.zeros(len(y))
        for s in set(sess):  # leave one session out
            te = sess == s
            mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
            w, b = fit((X[~te] - mu) / sd, y[~te])
            scores[te] = ((X[te] - mu) / sd) @ w + b
        f, p, r, t = best_f1(y, scores)
        print(f"{name:10} AUC={auc(y, scores):.3f}  bestF1={f:.2f} (P={p:.2f} R={r:.2f})")
        if name == "kareem+":
            mu, sd = X.mean(0), X.std(0) + 1e-6
            w, _ = fit((X - mu) / sd, y)
            print("  standardized weights:", {c: round(float(v), 2) for c, v in sorted(zip(cols, w), key=lambda z: -abs(z[1]))})


if __name__ == "__main__":
    main()
