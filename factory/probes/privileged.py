"""Information-ceiling probe (P-B): how much would a student gain if it could SEE what the teacher sees?

    python -m factory.probes.privileged <recipe model.py>

The recipe (its featurize + build_model + TRAIN_CONFIG lr/epochs/batch/tau/teacher) is trained on
behaviour states with extra privileged inputs that no deployable policy has, then evaluated
closed-loop on the 280-scenario development set:
  obs          observation features only (the deployable baseline)
  +scenario    + family one-hot and the scenario's true parameters (what kind of session is this?)
  +hidden      + true hidden state (instruction salience, fact integrity, true redundancy / stale
                 fractions, warm-up, health)
  +both        + both
The gap between `+both` and `obs` bounds what better belief-state estimation could ever add.
Results: specs/optimizer/privileged_results.json
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from factory import config
from factory.probes import lab
from factory.simulator import dataset as D
from factory.simulator import env as E

SEEDS = (0, 1, 2)
VARIANTS = ("obs", "+scenario", "+hidden", "+both")
OUT = config.PLAN_DIR / "privileged_results.json"
_PMAX = {k: max(abs(v[i]) for v in E.FAMILY_PARAMS.values()) or 1.0 for i, k in enumerate(E.PARAM_KEYS)}


def privileged(s: E.EnvState, sc: E.Scenario, variant: str) -> list[float]:
    out = []
    if variant in ("+scenario", "+both"):
        out += [1.0 if sc.family == f else 0.0 for f in E.FAMILIES]
        out += [sc.p[k] / _PMAX[k] for k in E.PARAM_KEYS]
    if variant in ("+hidden", "+both"):
        content = max(1e-6, s.total - E.SYSTEM_TOKENS)
        out += [s.salience, s.integrity, s.red / content, s.stale / content, s.warmup / 2.0, E.health(s)]
    return out


def _job(args):
    recipe, variant, seed = args
    torch.set_num_threads(1)
    mod = lab.load_recipe(Path(recipe))
    tc = dict(getattr(mod, "TRAIN_CONFIG", {}))
    teacher = str(tc.get("teacher", "v2"))
    cache = D.load_cache(config.CACHE_DIR)
    rows, qs = [], []
    for ei, ep in enumerate(cache["train"]):
        if ei % 7 == 0:
            continue
        sc = E.Scenario(ep["sid"], ep["family"], ep["seed"])
        states = lab.replay_states(ep)
        H = ep["history"]
        for t in range(len(H)):
            rows.append([float(v) for v in mod.featurize(H[: t + 1])] + privileged(states[t], sc, variant))
            qs.append(ep["q"][teacher][t])
    x, q = torch.tensor(rows, dtype=torch.float32), torch.tensor(qs, dtype=torch.float32)
    model, mu, sd = lab.train_student(mod, x, q, seed, target="soft",
                                      tau=float(tc.get("target_temperature", 0.5) or 0.5))
    # closed-loop on the development set with the same privileged inputs
    scs = lab.dev_scenarios()
    st = [E.EnvState() for _ in scs]
    hist = [[] for _ in scs]
    for _t in range(E.HORIZON):
        feats = []
        for i, sc in enumerate(scs):
            hist[i].append(E.observe(st[i], sc))
            feats.append([float(v) for v in mod.featurize(hist[i])] + privileged(st[i], sc, variant))
        xb = (torch.tensor(feats, dtype=torch.float32) - mu) / sd
        with torch.no_grad():
            acts = model(xb).argmax(1).tolist()
        for i, sc in enumerate(scs):
            E.step(st[i], acts[i], sc)
    return variant, seed, [s.reward for s in st]


def _baseline(args):
    sid, fam, seed = args
    sc = E.Scenario(sid, fam, seed)
    out = {}
    for name, pol in (("noop", lambda h, s: 0), ("heuristic", lambda h, s: E.heuristic_policy(h)),
                      ("clairvoyant", lambda h, s: E.clairvoyant_action(s, sc))):
        out[name] = E.run_episode(sc, pol)[0].reward
    return out


def main(recipe: str) -> None:
    scs = lab.dev_scenarios()
    with ProcessPoolExecutor(6) as ex:
        base = list(ex.map(_baseline, [(sc.sid, sc.family, sc.seed) for sc in scs], chunksize=8))
        runs = list(ex.map(_job, [(recipe, v, s) for v in VARIANTS for s in SEEDS]))
    anchors = {k: sum(b[k] for b in base) / len(base) for k in ("noop", "heuristic", "clairvoyant")}
    per = {}
    for variant, _seed, rets in runs:
        per.setdefault(variant, []).append(rets)
    mean_of = lambda rs: [sum(r[i] for r in rs) / len(rs) for i in range(len(rs[0]))]  # noqa: E731
    obs = mean_of(per["obs"])
    report = {"recipe": recipe, "dev_scenarios": len(scs), "anchors": anchors, "variants": {}}
    for v in VARIANTS:
        m = mean_of(per[v])
        mr = sum(m) / len(m)
        d = [a - b for a, b in zip(m, obs)]
        md = sum(d) / len(d)
        se = statistics.stdev(d) / math.sqrt(len(d)) if any(d) else 0.0
        sim = D.anchored_score(mr, anchors["noop"], anchors["heuristic"], anchors["clairvoyant"])
        fam = lab.family_breakdown(m, scs)
        report["variants"][v] = {"mean_return": round(mr, 3), "sim_score": round(sim, 2),
                                 "delta_vs_obs": round(md, 3), "se": round(se, 3), "family": fam}
        print(f"{v:10s} return {mr:7.3f}  sim {sim:6.2f}  delta vs obs {md:+.3f} +- {se:.3f}")
    print("anchors on the dev set:", {k: round(v, 3) for k, v in anchors.items()})
    OUT.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
