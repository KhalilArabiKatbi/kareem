"""Run the plateau probes from consultation 2 and write specs/optimizer/probe_results.json.

    python -m factory.probes.run_probes [stage ...]      stages: base, targets, teachers, dagger, t4
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from factory import config
from factory.probes import lab
from factory.simulator import dataset as D

OUT = config.PLAN_DIR / "probe_results.json"
SEEDS = (0, 1, 2)


def _job(args):
    name, target, tau, seed, x, q, recipe = args
    torch.set_num_threads(1)
    mod = lab.load_recipe(Path(recipe))
    model, mu, sd = lab.train_student(mod, x, q, seed, target=target, tau=tau)
    rets = lab.evaluate(mod, model, mu, sd, lab.dev_scenarios())
    return name, seed, rets, {k: v.clone() for k, v in model.state_dict().items()}, mu, sd


def run_jobs(jobs, workers=7):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_job, jobs))


def summarise(results: dict, base_key: str, scale: float) -> dict:
    scs = lab.dev_scenarios()
    mean_of = lambda runs: [sum(r[i] for r in runs) / len(runs) for i in range(len(runs[0]))]  # noqa: E731
    base = mean_of(results[base_key])
    out = {}
    for name, runs in results.items():
        m = mean_of(runs)
        d, se = lab.paired(m, base)
        out[name] = {
            "seeds": len(runs),
            "mean_return": round(sum(m) / len(m), 3),
            "delta_return_vs_base": round(d, 3),
            "se_return": round(se, 3),
            "delta_sim": round(d * scale, 2),
            "t": round(d / se, 2) if se else None,
            "per_seed_mean_return": [round(sum(r) / len(r), 3) for r in runs],
            "family": lab.family_breakdown(m, scs),
        }
    return out


def main(stages):
    torch.set_num_threads(1)
    cache = D.load_cache(config.CACHE_DIR)
    base_b = {k: sum(b[k] for b in cache["baselines"]) / len(cache["baselines"]) for k in ("heuristic", "clairvoyant")}
    scale = 50.0 / (base_b["clairvoyant"] - base_b["heuristic"])
    eps = cache["train"]
    recipe = str(lab.RECIPE)
    mod = lab.load_recipe()
    report = json.loads(OUT.read_text()) if OUT.exists() else {}
    report["scale_sim_per_return"] = round(scale, 3)
    results: dict[str, list] = {}

    t0 = time.time()
    x, q_v2 = lab.build_xq(mod, eps, [ep["q"]["v2"] for ep in eps])
    print(f"[probe] features {tuple(x.shape)} in {time.time() - t0:.1f}s", flush=True)

    # base (P-F: 5 seeds) + target variants (P-G)
    jobs = [("base_v2_soft", "soft", None, s, x, q_v2, recipe) for s in range(5)]
    if "targets" in stages:
        for tau in (0.5, 2.0, 4.0):
            jobs += [(f"v2_soft_tau{tau}", "soft", tau, s, x, q_v2, recipe) for s in SEEDS]
        jobs += [("v2_advantage", "advantage", None, s, x, q_v2, recipe) for s in SEEDS]
        jobs += [("v2_hard", "hard", None, s, x, q_v2, recipe) for s in SEEDS]
        q_v1 = lab.build_xq(mod, eps, [ep["q"]["v1"] for ep in eps])[1]
        jobs += [("v1_soft", "soft", None, s, x, q_v1, recipe) for s in SEEDS]
    raw = run_jobs(jobs)
    states = {}
    for name, seed, rets, sd_, mu, sdv in raw:
        results.setdefault(name, []).append(rets)
        if name == "base_v2_soft":
            states[seed] = (sd_, mu, sdv)
    # bundle of the seed-0 base student (rollout policy for T4, behaviour policy for DAgger)
    bundle = lab.PROBE_CACHE / "base_student.pt"
    lab.PROBE_CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"recipe": recipe, "state_dict": states[0][0], "mu": states[0][1], "sd": states[0][2]}, bundle)
    if "targets" in stages:  # 5-seed logit ensemble of the base recipe
        members = []
        for s in range(5):
            torch.manual_seed(0)
            m = mod.build_model(x.shape[1], 6)
            m.load_state_dict(states[s][0])
            members.append(m)
        ens = lab.Ensemble(members).eval()
        results["v2_soft_ensemble5"] = [lab.evaluate(mod, ens, states[0][1], states[0][2], lab.dev_scenarios())]

    if "teachers" in stages:
        for teacher in ("v2f32", "t2"):
            qs = lab.label(eps, teacher)
            qt = lab.build_xq(mod, eps, qs)[1]
            for name, seed, rets, *_ in run_jobs([(f"teacher_{teacher}", "soft", None, s, x, qt, recipe) for s in SEEDS]):
                results.setdefault(name, []).append(rets)
            if teacher == "v2f32":  # P-C: sampling noise of v2@8 against v2@32
                v8 = q_v2.argmax(1)
                v32 = qt.argmax(1)
                report["P-C"] = {"agreement_v2at8_vs_v2at32": round(float((v8 == v32).float().mean()), 4),
                                 "regret_of_v2at8_under_v2at32": round(float((qt.max(1).values - qt.gather(1, v8[:, None]).squeeze(1)).mean()), 4)}

    if "dagger" in stages:
        extra = lab.onpolicy_data(bundle)
        x2, q2 = lab.build_xq(mod, extra, [e["q"]["v2"] for e in extra])
        xa, qa = torch.cat([x, x2]), torch.cat([q_v2, q2])
        for name, seed, rets, *_ in run_jobs([("dagger_v2_soft", "soft", None, s, xa, qa, recipe) for s in SEEDS]):
            results.setdefault(name, []).append(rets)

    if "t4" in stages:
        qs = lab.label(eps, "t4", {"bundle": str(bundle), "futures": 16})
        qt = lab.build_xq(mod, eps, qs)[1]
        for name, seed, rets, *_ in run_jobs([("teacher_t4_student_rollout", "soft", None, s, x, qt, recipe) for s in SEEDS]):
            results.setdefault(name, []).append(rets)
        results.setdefault("t4_hard", [])
        for name, seed, rets, *_ in run_jobs([("t4_hard", "hard", None, s, x, qt, recipe) for s in SEEDS]):
            results["t4_hard"].append(rets)

    if "combos" in stages:
        extra = lab.onpolicy_data(bundle)
        x2, q2 = lab.build_xq(mod, extra, [e["q"]["v2"] for e in extra])
        q4 = lab.build_xq(mod, eps, lab.label(eps, "t4", {"bundle": str(bundle), "futures": 16}))[1]
        q4d = lab.build_xq(mod, extra, lab.label(extra, "t4", {"bundle": str(bundle), "futures": 16}), split="all")[1]
        xa = torch.cat([x, x2])
        combos = {
            "combo_dagger_tau05": (xa, torch.cat([q_v2, q2]), 0.5),
            "combo_dagger_t4": (xa, torch.cat([q4, q4d]), None),
            "combo_dagger_t4_tau05": (xa, torch.cat([q4, q4d]), 0.5),
        }
        member_states = {}
        for cname, (cx, cq, tau) in combos.items():
            for name, seed, rets, sdict, mu, sdv in run_jobs([(cname, "soft", tau, s, cx, cq, recipe) for s in SEEDS]):
                results.setdefault(name, []).append(rets)
                member_states.setdefault(name, []).append((sdict, mu, sdv))
        for cname, members_ in member_states.items():  # 3-seed ensembles of each combo
            nets = []
            for sdict, _mu, _sd in members_:
                torch.manual_seed(0)
                net = mod.build_model(x.shape[1], 6)
                net.load_state_dict(sdict)
                nets.append(net)
            ens = lab.Ensemble(nets).eval()
            mu, sdv = members_[0][1], members_[0][2]
            results[f"{cname}_ens3"] = [lab.evaluate(mod, ens, mu, sdv, lab.dev_scenarios())]

    summary = summarise(results, "base_v2_soft", scale)
    base_seeds = summary["base_v2_soft"]["per_seed_mean_return"]
    mean = sum(base_seeds) / len(base_seeds)
    sd = (sum((v - mean) ** 2 for v in base_seeds) / (len(base_seeds) - 1)) ** 0.5
    report["P-F_seed_sd_return"] = round(sd, 3)
    report["P-F_seed_sd_sim"] = round(sd * scale, 2)
    report.setdefault("runs", {}).update(summary)
    report["dev_set"] = f"{lab.N_DEV} scenarios, seeds 70000+ (disjoint from training and evaluation)"
    OUT.write_text(json.dumps(report, indent=2))
    for name, s in summary.items():
        print(f"{name:30s} ret {s['mean_return']:7.3f}  d {s['delta_return_vs_base']:+.3f} +- {s['se_return']:.3f} "
              f"(sim {s['delta_sim']:+.2f}, t {s['t']})", flush=True)
    print("P-F seed sd:", report["P-F_seed_sd_return"], "return =", report["P-F_seed_sd_sim"], "sim")
    if "P-C" in report:
        print("P-C:", report["P-C"])


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"targets"})
