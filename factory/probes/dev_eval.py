"""Evaluate model.py files on the independent development set with the full simulator protocol.

    python -m factory.probes.dev_eval <model.py> [<model.py> ...]

Each model is trained exactly as the simulator would (its own TRAIN_CONFIG: teacher, target, data,
ensemble seeds, budgets) and then evaluated closed-loop on 200 development scenarios (seeds 70000+,
50 turns x 4, disjoint from training and evaluation), so comparing winners of different runs is
free of the selection bias of the evaluation set. Prints per-model mean return and paired deltas.
"""
from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from factory import config

RUNNER = r'''
import sys, json
from pathlib import Path
from factory.simulator import env as E, dataset as D, sim_runner as S
import os
BASE = int(os.environ.get("DEV_SEED_BASE", "70000"))
dev = [[E.Scenario(f"S-DEV-{k:03d}-{j}", E.FAMILIES[k % len(E.FAMILIES)], BASE + 4 * k + j) for j in range(4)]
       for k in range(50)]
E.eval_turns = lambda: dev
orig = D.load_cache
def load(cache_dir):
    c = orig(cache_dir)
    for turn in dev:
        for sc in turn:  # dev scenarios have no cached baselines; zeros keep the aggregator happy
            c["baselines"].append({"sid": sc.sid, "family": sc.family, "noop": 0.0, "heuristic": 0.0, "oracle": 0.0,
                                   "teacher_v2": 0.0, "clairvoyant": 0.0})
    return c
D.load_cache = load
sys.exit(S.main(Path(sys.argv[1]), Path(sys.argv[2])))
'''


def run(model_py: Path) -> list[float]:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "exp"
        d.mkdir()
        shutil.copy(model_py, d / "model.py")
        (d / "config.json").write_text(json.dumps({"experiment": "exp_999", "canary": "CANARY-exp_999-00000000"}))
        proc = subprocess.run([config.PYTHON_EXE, "-c", RUNNER, str(d), str(config.CACHE_DIR)], cwd=config.PROJECT_ROOT,
                              env=dict(__import__("os").environ),
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:])
        return json.loads((d / "sim_metrics.json").read_text())["turn_returns"]


def main(paths: list[str]) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(2) as ex:
        results = dict(zip(paths, ex.map(lambda p: run(Path(p)), paths)))
    base = results[paths[0]]
    for p, r in results.items():
        d = [a - b for a, b in zip(r, base)]
        m = sum(d) / len(d)
        se = statistics.stdev(d) / math.sqrt(len(d)) if any(d) else 0.0
        print(f"{p}: mean return {sum(r) / len(r):.3f}  delta vs first {m:+.3f} +- {se:.3f} return")


if __name__ == "__main__":
    main(sys.argv[1:])
