"""Cached synthetic training data and evaluation baselines.

The cache depends only on the environment source code, never on a model, so it is built once at
preflight (outside the 60 s simulator budget) and reused by every experiment. Building is spread
over worker processes; every episode is computed independently from its own seed, so the result
is identical whatever the process count or scheduling.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import env as E
from . import reference_policy as R

CACHE_VERSION = 4
ONPOLICY_SEED_OFFSET = 500
N_TRAIN_PER_FAMILY = 36
TEACHER_NAMES = tuple(E.TEACHERS)


def env_fingerprint() -> str:
    src = Path(E.__file__).read_bytes() + Path(__file__).read_bytes() + Path(R.__file__).read_bytes()
    return hashlib.sha256(src).hexdigest()[:16]


def cache_path(cache_dir: Path) -> Path:
    return cache_dir / f"sim_cache_v{CACHE_VERSION}_{env_fingerprint()}.pkl"


def _train_episode(args):
    """Roll out a behaviour policy; store every teacher's Q-vector for each visited state."""
    i, sid, family, seed = args
    sc = E.Scenario(sid, family, seed)
    rng = random.Random(7_000_000 + seed)
    mode = i % 3
    eps = (0.2, 0.5, 0.3)[mode]
    s = E.EnvState()
    history = []
    q = {t: [] for t in TEACHER_NAMES}
    while not s.done:
        history.append(E.observe(s, sc))
        for t in TEACHER_NAMES:
            q[t].append(E.oracle_values(s, sc, t))
        label_v1 = E.argmax_first(q["v1"][-1])
        if rng.random() < eps:
            a = rng.randrange(E.N_ACTIONS)
        elif mode == 2:
            a = E.heuristic_policy(history)
        else:
            a = label_v1
        E.step(s, a, sc)
    return {
        "sid": sid, "family": family, "seed": seed, "history": history,
        "q": q, "labels": {t: [E.argmax_first(v) for v in q[t]] for t in TEACHER_NAMES},
    }


def _onpolicy_episode(args):
    """Roll out the harness reference student; store every teacher's Q-vector for the visited states."""
    sid, family, seed, bundle = args
    sc = E.Scenario(sid, family, seed)
    pol = R.Policy(bundle)
    s = E.EnvState()
    history = []
    q = {t: [] for t in TEACHER_NAMES}
    while not s.done:
        history.append(E.observe(s, sc))
        for t in TEACHER_NAMES:
            q[t].append(E.oracle_values(s, sc, t))
        E.step(s, pol.act_batch([history])[0], sc)
    return {
        "sid": sid, "family": family, "seed": seed, "history": history,
        "q": q, "labels": {t: [E.argmax_first(v) for v in q[t]] for t in TEACHER_NAMES},
    }


def _eval_scenario(args):
    sid, family, seed = args
    sc = E.Scenario(sid, family, seed)
    out = {"sid": sid, "family": family, "seed": seed}
    policies = {
        "noop": lambda h, s: 0,
        "heuristic": lambda h, s: E.heuristic_policy(h),
        "oracle": lambda h, s: E.oracle_action(s, sc, "v1"),
        "teacher_v2": lambda h, s: E.oracle_action(s, sc, "v2"),
        "clairvoyant": lambda h, s: E.clairvoyant_action(s, sc),
    }
    for name, pol in policies.items():
        st, _, _ = E.run_episode(sc, pol)
        out[name] = st.reward
        out[f"{name}_success"] = int(st.progress >= E.TARGET)
    return out


def build_cache(cache_dir: Path, workers: int | None = None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir)
    if path.exists():
        return path
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    train_args = [(i, sc.sid, sc.family, sc.seed) for i, sc in enumerate(E.train_scenarios(N_TRAIN_PER_FAMILY))]
    eval_args = [(sc.sid, sc.family, sc.seed) for sc in E.eval_scenarios()]
    conf_args = [(sc.sid, sc.family, sc.seed) for turn in E.confirmation_turns() for sc in turn]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        train = list(ex.map(_train_episode, train_args, chunksize=4))
        evals = list(ex.map(_eval_scenario, eval_args, chunksize=4))
        confirmation = list(ex.map(_eval_scenario, conf_args, chunksize=4))
    # DAgger data: states visited by a deterministic, harness-owned reference student
    bundle = R.train(train)
    on_args = [(sc.sid.replace("S-TR", "S-OP"), sc.family, sc.seed + ONPOLICY_SEED_OFFSET, bundle)
               for sc in E.train_scenarios(N_TRAIN_PER_FAMILY)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        onpolicy = list(ex.map(_onpolicy_episode, on_args, chunksize=4))
    payload = {"version": CACHE_VERSION, "fingerprint": env_fingerprint(), "train": train, "onpolicy": onpolicy,
               "baselines": evals, "confirmation": confirmation, "teachers": TEACHER_NAMES}
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def load_cache(cache_dir: Path) -> dict:
    path = cache_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"simulator cache missing: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def anchored_score(model_return: float, noop: float, heuristic: float, clairvoyant: float) -> float:
    """Piecewise-linear score on mean return: NOOP=0, heuristic=50, clairvoyant bound=100."""
    if model_return <= noop:
        return 0.0
    if model_return <= heuristic:
        return 50.0 * (model_return - noop) / max(1e-6, heuristic - noop)
    return min(100.0, 50.0 + 50.0 * (model_return - heuristic) / max(1e-6, clairvoyant - heuristic))
