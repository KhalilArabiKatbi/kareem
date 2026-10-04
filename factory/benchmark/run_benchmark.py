"""Final benchmark: every model generation and every baseline on scenarios no run ever saw.

    python -m factory.benchmark.run_benchmark

Sets (seeds disjoint from training, on-policy, DAgger, development, evaluation and confirmation):
    main      1,050 in-distribution scenarios (150 per family)
    heavy     210 scenarios: context grows 1.5x faster, tool outputs 1.5x larger
    drift     210 scenarios: instruction salience decays 2x faster
    buggy     210 scenarios: +0.10 sub-task error rate, stickier error loops
    recall    210 scenarios: 2x more recall demands, every compaction loses facts heavily
Policies:
    baselines: do-nothing, heuristic, teacher v1, teacher v2, clairvoyant (sees the true future)
    learned:   the best model of each factory generation, each retrained with 3 seeds through the
               exact simulator protocol (exported weights), evaluated closed-loop.
Statistics: per-scenario paired differences, 95% confidence intervals (1.96 x SE over scenarios,
learned models averaged over seeds), seed spread, success / overflow / intervention rates, per-family
returns, decision latency. Output: specs/optimizer/benchmark_results.json and benchmark.md
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from factory import config
from factory.simulator import env as E

ARCH = config.EXPERIMENTS_DIR / "_archived"
MODELS = {
    "v4.1 best": ARCH / "20260928-213258" / "exp_008" / "model.py",
    "v4.2 best": ARCH / "20260929-043805" / "exp_007" / "model.py",
    "v4.3 best": ARCH / "20260929-104645" / "exp_006" / "model.py",
    "v4.4 best": ARCH / "20260929-115104" / "exp_005" / "model.py",
    "v4.5 reference recipe (it. 0)": config.EXPERIMENTS_DIR / "exp_000" / "model.py",
    "v4.5 selected best (it. 3)": config.EXPERIMENTS_DIR / "exp_003" / "model.py",
    "v4.5 confirmed best (it. 10)": config.EXPERIMENTS_DIR / "exp_010" / "model.py",
}
SEEDS = (1234, 2234, 3234)
BASELINES = ("noop", "heuristic", "teacher_v1", "teacher_v2", "clairvoyant")
OUT_JSON = config.PLAN_DIR / "benchmark_results.json"
OUT_MD = config.PLAN_DIR / "benchmark.md"
WORK = config.CACHE_DIR / "benchmark"


def _stress(p: dict, kind: str) -> dict:
    p = dict(p)
    if kind == "heavy":
        p["growth"] *= 1.5
        p["tool_size"] *= 1.5
    elif kind == "drift":
        p["sal_decay"] *= 2.0
    elif kind == "buggy":
        p["err"] = min(0.45, p["err"] + 0.10)
        p["loopiness"] = min(0.9, p["loopiness"] + 0.25)
    elif kind == "recall":
        p["recall_p"] = min(0.6, p["recall_p"] * 2.0)
        p["fact_density"] = 1.0
    return p


def scenario_specs() -> dict[str, list[tuple]]:
    sets = {"main": [(f"B-MAIN-{k:04d}", E.FAMILIES[k % 7], 200_000 + k, None) for k in range(1050)]}
    for j, kind in enumerate(("heavy", "drift", "buggy", "recall")):
        rows = []
        for k in range(210):
            fam, seed = E.FAMILIES[k % 7], 300_000 + 10_000 * j + k
            base = E.Scenario("x", fam, seed)
            rows.append((f"B-{kind.upper()}-{k:03d}", fam, seed, _stress(base.p, kind)))
        sets[kind] = rows
    return sets


def build(spec) -> E.Scenario:
    sid, fam, seed, params = spec
    return E.Scenario(sid, fam, seed, params=params)


def _baseline_job(spec):
    sc = build(spec)
    out = {}
    pols = {
        "noop": lambda h, s: 0,
        "heuristic": lambda h, s: E.heuristic_policy(h),
        "teacher_v1": lambda h, s: E.oracle_action(s, sc, "v1"),
        "teacher_v2": lambda h, s: E.oracle_action(s, sc, "v2"),
        "clairvoyant": lambda h, s: E.clairvoyant_action(s, sc),
    }
    for name, pol in pols.items():
        st, _, _ = E.run_episode(sc, pol)
        out[name] = _episode_stats(st)
    return out


def _episode_stats(st: E.EnvState) -> dict:
    return {"ret": st.reward, "success": int(st.progress >= E.TARGET), "overflow": int(st.overflows > 0),
            "interventions": sum(st.action_counts[1:]), "actions": list(st.action_counts)}


def export(name: str, model_py: Path, seed: int) -> Path:
    import hashlib
    tag = hashlib.sha256(f"{name}|{model_py}".encode()).hexdigest()[:8]  # stable across processes
    out = WORK / "policies" / f"{name.split(' (')[0].replace(' ', '_').replace('.', '')}_{tag}_{seed}"
    if (out / "intervention_policy.pt").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "exp"
        work.mkdir()
        shutil.copy(model_py, work / "model.py")
        (work / "config.json").write_text(json.dumps({"experiment": "exp_999", "canary": "CANARY-exp_999-00000000"}))
        env = dict(os.environ, FACTORY_SIM_SEED=str(seed))
        proc = subprocess.run([config.PYTHON_EXE, "-m", "factory.simulator.sim_runner", str(work), str(config.CACHE_DIR),
                               "--export", str(out)], cwd=config.PROJECT_ROOT, env=env, capture_output=True, text=True,
                              timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"export failed for {name} seed {seed}: {proc.stderr[-1500:]}")
    shutil.copy(model_py, out / "policy_model.py")
    shutil.copy(config.TEMPLATES_DIR / "policy_runtime.py", out / "policy_runtime.py")
    return out


def _export_job(args):
    return args[0], args[2], str(export(*args))


def _learned_job(args):
    name, seed, pdir, specs_by_set = args
    import torch
    torch.set_num_threads(1)
    sys.dont_write_bytecode = True
    sys.path.insert(0, pdir)
    import importlib
    rt = importlib.import_module("policy_runtime")
    importlib.reload(rt)
    pol = rt.InterventionPolicy.load(pdir)
    sys.path.pop(0)
    results = {}
    for set_name, specs in specs_by_set.items():
        scs = [build(s) for s in specs]
        states = [E.EnvState() for _ in scs]
        hists = [[] for _ in scs]
        for _t in range(E.HORIZON):
            for i, sc in enumerate(scs):
                hists[i].append(E.observe(states[i], sc))
            acts = pol.decide_batch(hists)
            for i, sc in enumerate(scs):
                E.step(states[i], rt.ACTIONS.index(acts[i]), sc)
        results[set_name] = [_episode_stats(s) for s in states]
    # single-decision latency on realistic histories
    h = hists[0]
    t0 = time.perf_counter()
    for k in range(1, 201):
        pol.decide(h[: 1 + k % len(h)])
    latency_ms = (time.perf_counter() - t0) / 200 * 1000
    return name, seed, results, latency_ms


def ci(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1)) if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)


def anchored(r, noop, heur, clair):
    if r <= noop:
        return 0.0
    if r <= heur:
        return 50.0 * (r - noop) / (heur - noop)
    return min(100.0, 50.0 + 50.0 * (r - heur) / (clair - heur))


def main() -> None:
    t_start = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    sets = scenario_specs()
    all_specs = [s for v in sets.values() for s in v]

    # 1. baselines (cached)
    base_path = WORK / "baselines.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
    else:
        with ProcessPoolExecutor(7) as ex:
            res = list(ex.map(_baseline_job, all_specs, chunksize=6))
        base = {spec[0]: r for spec, r in zip(all_specs, res)}
        base_path.write_text(json.dumps(base))
    print(f"[bench] baselines ready ({time.time() - t_start:.0f}s)", flush=True)

    # 2. export every learned model with 3 training seeds (exact simulator protocol)
    jobs = [(name, path, seed) for name, path in MODELS.items() for seed in SEEDS]
    with ProcessPoolExecutor(3) as ex:
        exported = list(ex.map(_export_job, jobs))
    print(f"[bench] {len(exported)} trained policies exported ({time.time() - t_start:.0f}s)", flush=True)

    # 3. evaluate learned policies on every set
    with ProcessPoolExecutor(6) as ex:
        learned = list(ex.map(_learned_job, [(n, s, d, sets) for n, s, d in exported]))
    print(f"[bench] learned policies evaluated ({time.time() - t_start:.0f}s)", flush=True)

    # 4. aggregate
    report = {"sets": {k: len(v) for k, v in sets.items()}, "seeds": list(SEEDS), "policies": {}, "latency_ms": {}}
    per_policy = {}  # name -> set -> list (per scenario) of stats averaged over seeds
    for b in BASELINES:
        per_policy[b] = {sn: [base[s[0]][b] for s in specs] for sn, specs in sets.items()}
    seed_runs: dict[str, list] = {}
    for name, seed, res, lat in learned:
        seed_runs.setdefault(name, []).append(res)
        report["latency_ms"].setdefault(name, []).append(round(lat, 3))
    for name, runs in seed_runs.items():
        per_policy[name] = {}
        for sn in sets:
            per_policy[name][sn] = [{
                "ret": sum(r[sn][i]["ret"] for r in runs) / len(runs),
                "success": sum(r[sn][i]["success"] for r in runs) / len(runs),
                "overflow": sum(r[sn][i]["overflow"] for r in runs) / len(runs),
                "interventions": sum(r[sn][i]["interventions"] for r in runs) / len(runs),
                "actions": [sum(r[sn][i]["actions"][a] for r in runs) / len(runs) for a in range(6)],
            } for i in range(len(sets[sn]))]

    for sn, specs in sets.items():
        mean = lambda pol, key="ret": sum(x[key] for x in per_policy[pol][sn]) / len(specs)  # noqa: E731
        noop, heur, clair = mean("noop"), mean("heuristic"), mean("clairvoyant")
        ref = [x["ret"] for x in per_policy["v4.2 best"][sn]]
        heur_r = [x["ret"] for x in per_policy["heuristic"][sn]]
        for pol in per_policy:
            rets = [x["ret"] for x in per_policy[pol][sn]]
            m, half = ci(rets)
            dh, hh = ci([a - b for a, b in zip(rets, heur_r)])
            d42, h42 = ci([a - b for a, b in zip(rets, ref)])
            fam = {}
            for spec, x in zip(specs, per_policy[pol][sn]):
                fam.setdefault(spec[1], []).append(x["ret"])
            acts = [sum(x["actions"][a] for x in per_policy[pol][sn]) for a in range(6)]
            entry = {
                "mean_return": round(m, 3), "ci95": round(half, 3),
                "sim_score": round(anchored(m, noop, heur, clair), 2),
                "vs_heuristic": [round(dh, 3), round(hh, 3)], "vs_v42_best": [round(d42, 3), round(h42, 3)],
                "success_rate": round(mean(pol, "success"), 4), "overflow_rate": round(mean(pol, "overflow"), 4),
                "interventions_per_episode": round(mean(pol, "interventions"), 2),
                "action_share": {E.ACTIONS[a]: round(acts[a] / max(1, sum(acts)), 4) for a in range(6)},
                "family_return": {k: round(sum(v) / len(v), 2) for k, v in fam.items()},
            }
            if pol in seed_runs:
                per_seed = [sum(r[sn][i]["ret"] for i in range(len(specs))) / len(specs) for r in seed_runs[pol]]
                mu = sum(per_seed) / len(per_seed)
                entry["seed_means"] = [round(v, 3) for v in per_seed]
                entry["seed_sd_return"] = round(math.sqrt(sum((v - mu) ** 2 for v in per_seed) / (len(per_seed) - 1)), 3)
            report["policies"].setdefault(pol, {})[sn] = entry
    report["runtime_seconds"] = round(time.time() - t_start)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    write_markdown(report)
    print(f"[bench] done in {report['runtime_seconds']}s -> {OUT_JSON}", flush=True)


def write_markdown(rep: dict) -> None:
    order = ["noop", "heuristic", "teacher_v1", "teacher_v2", *MODELS.keys(), "clairvoyant"]
    L = ["# Benchmark: every model generation on unseen scenarios", "",
         f"Sets: {rep['sets']}; learned models retrained with seeds {rep['seeds']} and averaged per scenario.",
         "sim_score is anchored per set: do-nothing 0, heuristic 50, clairvoyant 100. CI = 95% interval over scenarios.", ""]
    for sn in rep["sets"]:
        L += [f"## {sn}", "", "| Policy | sim_score | Mean return (95% CI) | vs heuristic | vs v4.2 best | Success | Overflow | Interventions/ep | Seed SD |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for pol in order:
            e = rep["policies"][pol][sn]
            L.append(f"| {pol} | {e['sim_score']:.1f} | {e['mean_return']:.2f} +- {e['ci95']:.2f} | "
                     f"{e['vs_heuristic'][0]:+.2f} +- {e['vs_heuristic'][1]:.2f} | {e['vs_v42_best'][0]:+.2f} +- {e['vs_v42_best'][1]:.2f} | "
                     f"{e['success_rate']:.3f} | {e['overflow_rate']:.3f} | {e['interventions_per_episode']:.1f} | "
                     f"{e.get('seed_sd_return', '-')} |")
        L.append("")
    L += ["## Decision latency (single decision, ms)", ""]
    for pol, v in rep["latency_ms"].items():
        L.append(f"- {pol}: {sum(v) / len(v):.2f} ms")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
