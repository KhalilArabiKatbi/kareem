"""Grokking probe: does closed-loop return keep improving far beyond the simulator's training budget?

    python -m factory.probes.grokking <model.py> [<model.py> ...]

For each recipe (its featurize + build_model + TRAIN_CONFIG teacher/target/tau/lr/batch) and each
weight decay in {0, 0.01, 0.1}, one model is trained with AdamW for up to 100k optimizer steps on the
behaviour states. At log-spaced checkpoints it records train/val accuracy, val regret (teacher-value
gap of the chosen action) and closed-loop return on the 280-scenario development set (never the
evaluation set). Grokking-like = a late checkpoint (>= 10k steps) beating the best early checkpoint
(<= 3k steps, i.e. the regime the simulator budget allows) by >= 0.3 return at >= 2 standard errors.
Results: specs/optimizer/grokking_results.json
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from factory import config
from factory.probes import lab
from factory.simulator import dataset as D

CHECKPOINTS = (500, 1_000, 2_000, 3_000, 5_000, 8_000, 12_000, 20_000, 30_000, 50_000, 75_000, 100_000)
WEIGHT_DECAYS = (0.0, 0.01, 0.1)
OUT = config.PLAN_DIR / "grokking_results.json"


def _run(args):
    recipe, wd = args
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    mod = lab.load_recipe(Path(recipe))
    tc = dict(getattr(mod, "TRAIN_CONFIG", {}))
    teacher = str(tc.get("teacher", "v2"))
    target = str(tc.get("target", "soft"))
    tau = float(tc.get("target_temperature", 0.5) or 0.5)
    lr = float(tc.get("lr", 1e-3))
    bs = int(tc.get("batch_size", 256))
    cache = D.load_cache(config.CACHE_DIR)
    eps = cache["train"]
    x, q = lab.build_xq(mod, eps, [ep["q"][teacher] for ep in eps], "train")
    xv, qv = lab.build_xq(mod, eps, [ep["q"][teacher] for ep in eps], "val")
    mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    xn, xvn = (x - mu) / sd, (xv - mu) / sd
    model = mod.build_model(x.shape[1], 6)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator().manual_seed(1234)
    scs = lab.dev_scenarios()
    rows, curve = xn.shape[0], []
    step, perm, pos = 0, torch.randperm(xn.shape[0], generator=gen), 0
    t0 = time.time()
    for cp in CHECKPOINTS:
        model.train()
        while step < cp:
            if pos >= rows:
                perm, pos = torch.randperm(rows, generator=gen), 0
            idx = perm[pos: pos + bs]
            pos += bs
            logits = model(xn[idx])
            qb = q[idx]
            if target == "hard":
                loss = torch.nn.functional.cross_entropy(logits, qb.argmax(1))
            else:
                loss = -(torch.softmax(qb / tau, 1) * torch.log_softmax(logits, 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        model.eval()
        with torch.no_grad():
            ptr, pva = model(xn).argmax(1), model(xvn).argmax(1)
        rets = lab.evaluate(mod, model, mu, sd, scs)
        curve.append({
            "step": cp, "epochs": round(cp * bs / rows, 1), "loss": round(float(loss), 4),
            "train_acc": round(float((ptr == q.argmax(1)).float().mean()), 4),
            "val_acc": round(float((pva == qv.argmax(1)).float().mean()), 4),
            "val_regret": round(float((qv.max(1).values - qv.gather(1, pva[:, None]).squeeze(1)).mean()), 4),
            "weight_norm": round(float(sum(p.pow(2).sum() for p in model.parameters()).sqrt()), 2),
            "dev_return": round(sum(rets) / len(rets), 3), "returns": rets,
        })
    return {"recipe": recipe, "weight_decay": wd, "seconds": round(time.time() - t0, 1), "curve": curve}


def analyse(run: dict) -> dict:
    early = [c for c in run["curve"] if c["step"] <= 3_000]
    late = [c for c in run["curve"] if c["step"] >= 10_000]
    be = max(early, key=lambda c: c["dev_return"])
    bl = max(late, key=lambda c: c["dev_return"])
    d = [a - b for a, b in zip(bl["returns"], be["returns"])]
    m = sum(d) / len(d)
    se = statistics.stdev(d) / math.sqrt(len(d))
    return {"best_early_step": be["step"], "best_early_return": be["dev_return"], "best_late_step": bl["step"],
            "best_late_return": bl["dev_return"], "late_minus_early": round(m, 3), "se": round(se, 3),
            "grokking_like": m >= 0.3 and m >= 2 * se}


def main(recipes: list[str]) -> None:
    jobs = [(r, wd) for r in recipes for wd in WEIGHT_DECAYS]
    with ProcessPoolExecutor(min(6, len(jobs))) as ex:
        runs = list(ex.map(_run, jobs))
    report = []
    for run in runs:
        verdict = analyse(run)
        print(f"\n{Path(run['recipe']).parent.name}  weight_decay={run['weight_decay']}  ({run['seconds']}s)  -> {verdict}")
        print("   step   epochs  train_acc val_acc val_regret |w|     dev_return")
        for c in run["curve"]:
            print(f"   {c['step']:>6} {c['epochs']:>7} {c['train_acc']:>9} {c['val_acc']:>7} {c['val_regret']:>10} "
                  f"{c['weight_norm']:>6} {c['dev_return']:>10}")
        for c in run["curve"]:
            c.pop("returns")
        report.append({**run, "verdict": verdict})
    OUT.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
