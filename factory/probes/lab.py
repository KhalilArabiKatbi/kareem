"""Offline probe lab (no LLM). Measures what limits the student on a DEV set that is disjoint from
both the training scenarios and the 200 evaluation scenarios, so selecting a protocol here does not
overfit the evaluation set.

Every probe trains a fixed "recipe" (a model.py) on some target, evaluates it closed-loop on the dev
set, and compares per-scenario returns paired against the baseline recipe (teacher v2, soft).

    python -m factory.probes.lab <probe> [...]
"""
from __future__ import annotations

import importlib.util
import json
import math
import pickle
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from factory import config
from factory.simulator import dataset as D
from factory.simulator import env as E

RECIPE = config.EXPERIMENTS_DIR / "exp_007" / "model.py"
PROBE_CACHE = config.CACHE_DIR / "probes"
N_DEV = 280
SCALE = None  # sim points per return point, filled from baselines


def dev_scenarios() -> list[E.Scenario]:
    return [E.Scenario(f"S-DEV-{k:03d}", E.FAMILIES[k % len(E.FAMILIES)], 70_000 + k) for k in range(N_DEV)]


def load_recipe(path: Path = RECIPE):
    spec = importlib.util.spec_from_file_location(f"recipe_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- replay: recover hidden states of cached training episodes -------------------------------
def replay_states(ep: dict) -> list[E.EnvState]:
    sc = E.Scenario(ep["sid"], ep["family"], ep["seed"])
    s = E.EnvState()
    H = ep["history"]
    out = []
    for t in range(len(H)):
        out.append(s.clone())
        if t + 1 < len(H):
            E.step(s, H[t + 1]["last_action"], sc)
    assert E.observe(out[-1], sc) == H[-1], f"replay mismatch in {ep['sid']}"
    return out


# ---- generic teachers --------------------------------------------------------------------------
def plan_values(s: E.EnvState, sc: E.Scenario, futures: int, depth: int | None, gamma: float,
                two_step: bool, bootstrap: bool = True) -> list[float]:
    """Per-first-action value: mean over imagined futures; depth None = plan to the episode end."""
    futs = E.imagined_futures(s, sc, futures)

    def value(first, fut):
        c = s.clone()
        v, g = 0.0, 1.0
        for a in first:
            if c.done:
                break
            v += g * E.step(c, a, fut)
            g *= gamma
        steps = (E.HORIZON if depth is None else depth) - len(first)
        for _ in range(max(0, steps)):
            if c.done:
                break
            v += g * E.step(c, E.hidden_rollout_policy(c, fut), fut)
            g *= gamma
        if bootstrap and not c.done:
            v += g * 0.7 * E.health(c) * min(E.HORIZON - c.t, 10)
        return v

    vals = []
    for a0 in range(E.N_ACTIONS):
        if two_step:
            vals.append(max(sum(value((a0, a1), f) for f in futs) / len(futs) for a1 in range(E.N_ACTIONS)))
        else:
            vals.append(sum(value((a0,), f) for f in futs) / len(futs))
    return vals


TEACHER_SPECS = {
    "v2f32": dict(futures=32, depth=10, gamma=0.95, two_step=True),
    "t2": dict(futures=16, depth=None, gamma=1.0, two_step=True, bootstrap=False),
}


# ---- student-rollout teacher (T4): one improvement step over a trained student ------------------
class StudentPolicy:
    def __init__(self, bundle_path: Path):
        b = torch.load(bundle_path, weights_only=False)
        self.mod = load_recipe(Path(b["recipe"]))
        self.nf = len(self.mod.FEATURE_NAMES)
        torch.manual_seed(0)
        self.model = self.mod.build_model(self.nf, E.N_ACTIONS)
        self.model.load_state_dict(b["state_dict"])
        self.model.eval()
        self.mu, self.sd = b["mu"], b["sd"]

    def act_batch(self, hists: list[list[dict]]) -> list[int]:
        rows = [[float(v) for v in self.mod.featurize(h)] for h in hists]
        x = (torch.tensor(rows, dtype=torch.float32) - self.mu) / self.sd
        with torch.no_grad():
            return self.model(x).argmax(1).tolist()


def student_values(s, sc, hist, pol: StudentPolicy, futures: int) -> list[float]:
    """Q(a0) = r(a0) + return of the student policy afterwards, to the episode end, mean over futures."""
    futs = E.imagined_futures(s, sc, futures)
    roll = []
    for a0 in range(E.N_ACTIONS):
        for f in futs:
            c = s.clone()
            r = E.step(c, a0, f)
            roll.append([a0, f, c, list(hist), r])
    while True:
        live = [x for x in roll if not x[2].done]
        if not live:
            break
        for x in live:
            x[3].append(E.observe(x[2], x[1]))
        acts = pol.act_batch([x[3] for x in live])
        for x, a in zip(live, acts):
            x[4] += E.step(x[2], a, x[1])
    vals = [0.0] * E.N_ACTIONS
    for a0, _f, _c, _h, r in roll:
        vals[a0] += r / len(futs)
    return vals


# ---- labelling jobs (process pool) -----------------------------------------------------------------
def _label_episode(args):
    ep, teacher, extra = args
    torch.set_num_threads(1)
    states = replay_states(ep)
    sc = E.Scenario(ep["sid"], ep["family"], ep["seed"])
    if teacher == "t4":
        pol = StudentPolicy(Path(extra["bundle"]))
        return [student_values(states[t], sc, ep["history"][: t + 1], pol, extra.get("futures", 16))
                for t in range(len(states))]
    spec = TEACHER_SPECS[teacher]
    return [plan_values(st, sc, **spec) for st in states]


def label(episodes: list[dict], teacher: str, extra: dict | None = None, workers: int = 7) -> list[list[list[float]]]:
    PROBE_CACHE.mkdir(parents=True, exist_ok=True)
    import hashlib
    ident = json.dumps([[ep["sid"], ep["seed"], len(ep["history"])] for ep in episodes] + [extra or {}], sort_keys=True)
    key = f"{teacher}_{hashlib.sha256(ident.encode()).hexdigest()[:16]}"
    path = PROBE_CACHE / f"labels_{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_label_episode, [(ep, teacher, extra or {}) for ep in episodes], chunksize=2))
    path.write_bytes(pickle.dumps(out))
    print(f"[label] {teacher}: {sum(len(x) for x in out)} states in {time.time() - t0:.0f}s", flush=True)
    return out


# ---- on-policy data (DAgger) ---------------------------------------------------------------------
def _onpolicy_episode(args):
    sid, family, seed, bundle = args
    torch.set_num_threads(1)
    pol = StudentPolicy(Path(bundle))
    sc = E.Scenario(sid, family, seed)
    s = E.EnvState()
    hist, qs = [], []
    while not s.done:
        hist.append(E.observe(s, sc))
        qs.append(E.oracle_values(s, sc, "v2"))
        a = pol.act_batch([hist])[0]
        E.step(s, a, sc)
    return {"sid": sid, "family": family, "seed": seed, "history": hist, "q": {"v2": qs}}


def onpolicy_data(bundle: Path, workers: int = 7) -> list[dict]:
    path = PROBE_CACHE / "dagger_v2.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    scs = E.train_scenarios(D.N_TRAIN_PER_FAMILY)
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_onpolicy_episode, [(sc.sid, sc.family, sc.seed + 500, str(bundle)) for sc in scs], chunksize=2))
    path.write_bytes(pickle.dumps(out))
    return out


# ---- training + evaluation -----------------------------------------------------------------------
def build_xq(mod, episodes, qsets, split="train"):
    """Features and teacher values for the training split (every 7th episode is validation)."""
    xs, qs = [], []
    for ei, (ep, q) in enumerate(zip(episodes, qsets)):
        if (ei % 7 == 0) != (split == "val"):
            continue
        H = ep["history"]
        for t in range(len(H)):
            xs.append([float(v) for v in mod.featurize(H[: t + 1])])
            qs.append(q[t])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(qs, dtype=torch.float32)


def train_student(mod, x, q, seed, target="soft", tau=None, epochs=None):
    torch.manual_seed(1234 + seed)
    tc = dict(getattr(mod, "TRAIN_CONFIG", {}))
    tau = tau if tau is not None else float(tc.get("target_temperature", 1.0))
    mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    xn = (x - mu) / sd
    model = mod.build_model(x.shape[1], E.N_ACTIONS)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc.get("lr", 1e-3)),
                            weight_decay=float(tc.get("weight_decay", 0.0)))
    gen = torch.Generator().manual_seed(1234 + seed)
    bs = int(tc.get("batch_size", 256))
    ep_n = epochs or int(tc.get("epochs", 30))
    clip = float(tc.get("grad_clip", 0.0))
    for _ in range(ep_n):
        model.train()
        perm = torch.randperm(xn.shape[0], generator=gen)
        for i in range(0, xn.shape[0], bs):
            idx = perm[i: i + bs]
            logits = model(xn[idx])
            qb = q[idx]
            if target == "soft":
                loss = -(torch.softmax(qb / tau, 1) * torch.log_softmax(logits, 1)).sum(1).mean()
            elif target == "hard":
                loss = torch.nn.functional.cross_entropy(logits, qb.argmax(1))
            elif target == "advantage":
                loss = ((logits - logits.mean(1, keepdim=True)) - (qb - qb.mean(1, keepdim=True))).pow(2).mean()
            else:
                raise ValueError(target)
            opt.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    model.eval()
    return model, mu, sd


class Ensemble(torch.nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members]).mean(0)


def evaluate(mod, model, mu, sd, scenarios) -> list[float]:
    from factory.simulator.sim_runner import run_lockstep
    states, _, _ = run_lockstep(mod, model, mu, sd, len(mod.FEATURE_NAMES), scenarios)
    return [s.reward for s in states]


def paired(a: list[float], b: list[float]) -> tuple[float, float]:
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / len(d)
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
    return m, se


def family_breakdown(returns: list[float], scenarios) -> dict:
    fam = {}
    for r, sc in zip(returns, scenarios):
        fam.setdefault(sc.family, []).append(r)
    return {k: round(sum(v) / len(v), 2) for k, v in fam.items()}
