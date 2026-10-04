"""Python-computed search analytics for the Meta-Agent.

Aggregates only (no experiment ids, no logs): per-iteration metric rows, per-dimension marginals,
rank correlations, paired comparisons against the best iteration with standard errors, and the
status of the stopping rules. The code is generic over whatever dimensions exist; it knows no
dimension names.
"""
from __future__ import annotations

import json
import math

from . import config
from .state_store import read_json


def _canon(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 4 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else None


def _paired(a: list[float], b: list[float], scale: float) -> tuple[float, float]:
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0.0
    return round(m * scale, 2), round(se * scale, 2)


FAMILY_TURNS = {"steady_dialogue": 8}  # 50 turns round-robin over 7 families: the first family gets 8, the rest 7


def _median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def compute(state: dict, space: dict, width: int = config.PARALLEL_WIDTH) -> dict:
    base = state.get("baselines") or {}
    heur, clair = base.get("heuristic_return"), base.get("clairvoyant_return")
    scale = 50.0 / (clair - heur) if heur is not None and clair is not None and clair > heur else 1.0

    ledger = {e["iteration"]: e for e in space.get("ledger", [])}
    rows, turn_returns = [], {}
    for it in state.get("iterations", []):
        if it["status"] != "scored":
            continue
        i = it["iteration"]
        m = read_json(config.exp_dir(i) / "sim_metrics.json", None)
        if not m:
            continue
        t = m["training"]
        ft = m.get("finetune") or {}
        fam = m.get("family", {})
        rows.append({
            "iteration": i,
            "judge": it["score"],
            "sim_score": m["sim_score"],
            "mean_return": m["mean_return"],
            "noop_share": m.get("noop_share"),
            "val_acc": t["val_acc"],
            "val_regret": t.get("val_regret"),
            "teacher": t.get("teacher"),
            "target": t.get("target"),
            "finetune_accepted": ft.get("accepted") if ft else None,
            "features": t["features"],
            "params": t["params"],
            # below-heuristic runs are almost always broken implementations; duplicates add no information
            "valid": m["sim_score"] >= 50.0 and it.get("duplicate_of") is None,
            "duplicate_of": it.get("duplicate_of"),
            "weighted_family_gap_to_clairvoyant": {
                # return gap x the family's share of the 50 turns = what closing it would add to the mean
                k: round((v["clairvoyant"] - v["return"]) * FAMILY_TURNS.get(k, 7) / 50, 3) for k, v in fam.items()
            },
            "effective_config": m.get("effective_config"),
        })
        if m.get("turn_returns"):
            turn_returns[i] = m["turn_returns"]

    # per-dimension marginals over scored coordinates (generic: loops over whatever exists)
    marginals = {}
    dims = list(space.get("dimensions", {})) + list(space.get("pruned_dimensions", {}))
    sim_by_it = {r["iteration"]: r["sim_score"] for r in rows if r["valid"]}
    for d in dims:
        groups: dict[str, list[float]] = {}
        for i, e in ledger.items():
            if i in sim_by_it and d in e.get("coordinate", {}):
                groups.setdefault(_canon(e["coordinate"][d]), []).append(sim_by_it[i])
        if groups:
            marginals[d] = sorted(
                ({"value": json.loads(k), "n": len(v), "median_sim": round(_median(v), 2), "max_sim": max(v)}
                 for k, v in groups.items()),
                key=lambda g: -g["median_sim"],
            )

    # rank correlations with sim_score
    corr = {}
    valid_rows = [r for r in rows if r["valid"]]
    sims = [r["sim_score"] for r in valid_rows]

    def supported(xs):  # at least two distinct values, each seen at least twice
        counts = {}
        for x in xs:
            counts[x] = counts.get(x, 0) + 1
        return sum(1 for c in counts.values() if c >= 2) >= 2

    for key in ("noop_share", "val_acc", "val_regret", "features", "params"):
        vals = [r.get(key) for r in valid_rows]
        if vals and all(v is not None for v in vals):
            corr[key] = spearman([float(v) for v in vals], sims)
    for d in dims:
        xs, ys = [], []
        for r in valid_rows:
            v = ledger.get(r["iteration"], {}).get("coordinate", {}).get(d)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                xs.append(float(v))
                ys.append(r["sim_score"])
        if len(xs) >= 4 and supported(xs):
            corr[f"dim:{d}"] = spearman(xs, ys)

    # which values of each simulator-contract lever have actually taken effect so far
    lever_coverage: dict[str, list] = {}
    for r in valid_rows:
        for k, v in (r.get("effective_config") or {}).items():
            vals = lever_coverage.setdefault(k, [])
            if v not in vals:
                vals.append(v)

    # paired comparison of every scored iteration against the best one (same 50 evaluation turns)
    paired = []
    best = state.get("best")
    if best and best["iteration"] in turn_returns:
        bt = turn_returns[best["iteration"]]
        for i, tr in sorted(turn_returns.items()):
            if i == best["iteration"]:
                continue
            dm, se = _paired(tr, bt, scale)
            paired.append({"iteration": i, "delta_sim_vs_best": dm, "se_sim": se})
    ses = sorted(p["se_sim"] for p in paired)
    typical_se = ses[len(ses) // 2] if ses else None

    # stopping-rule status (information only; Python decides)
    closed = state.get("next_iteration", 0)
    w = config.STAGNATION_WINDOW
    scores = {int(k): v for k, v in state.get("scores", {}).items() if v is not None}
    horizon_start = closed + width - w  # window start at the next check
    older = [v for i, v in scores.items() if i < horizon_start]
    last_add = state.get("last_dimension_added_iteration", -1)
    added = {}
    for m in space.get("mutation_log", []):
        if m["op"] == "add_dimension":
            added[m["dimension"]] = added.get(m["dimension"], 0) + 1

    return {
        "objective": "maximise sim_score (closed-loop mean return); the judge score stays within +-3 of sim_score",
        "sim_points_per_return_point": round(scale, 3),
        "typical_paired_se_sim": typical_se,
        "noise_note": "differences smaller than about 2 x typical_paired_se_sim are not distinguishable",
        "rows": rows,
        "marginals": marginals,
        "spearman_with_sim_score": corr,
        "paired_vs_best": paired,
        "stopping_rules": {
            "iterations_closed": closed,
            "best_judge_score": max(scores.values()) if scores else None,
            "judge_score_to_beat_next_window": (max(older) + config.MIN_IMPROVEMENT) if older else None,
            "last_dimension_added_iteration": last_add,
            "no_new_dimension_halt_after_iteration": last_add + w,
        },
        "readded_dimensions": sorted(k for k, v in added.items() if v > 1),
        "lever_coverage": lever_coverage,
        "duplicates": [{"iteration": r["iteration"], "same_model_as": r["duplicate_of"]} for r in rows if r["duplicate_of"] is not None],
        "invalid_iterations": [r["iteration"] for r in rows if not r["valid"]],
        "seed_spread_sim": config.SEED_SPREAD_SIM,
        "seed_spread_note": "retraining an identical recipe with different seeds moves sim_score by about this "
                            "much (probe P-F); treat smaller differences as noise",
    }
