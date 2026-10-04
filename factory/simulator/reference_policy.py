"""Harness-owned reference student.

A fixed, deterministic policy (reference featurizer + small MLP trained on teacher-v2 soft targets).
The cache builder uses it for two things that need a *learned* observation-only policy:
  - on-policy training states (DAgger): the reference student is rolled out on training seeds and
    the states it visits are labelled by the teachers;
  - teacher "t4": one improvement step over the reference student (it is the rollout policy).
It lives in harness source code, so the cache still depends only on source code, never on a file an
LLM wrote.
"""
from __future__ import annotations

import math

import torch

from . import env as E

FEATURE_NAMES = [
    "util", "headroom", "tool_frac", "redundancy", "stale", "confidence", "phase", "since_instr",
    "since_compact", "error_streak", "last_error", "recall_miss", "topic_shift", "util_delta",
    "growth_ema", "steps_to_overflow", "pressure", "salience_est", "integrity_est", "recall_intensity",
    "loop_persistence", "streak_pressure", "tool_rate", "shift_rate", "cliff_margin", "required_rate",
    "recent_success", "last_a0", "last_a1", "last_a2", "last_a3", "last_a4", "last_a5",
]
SEED = 4321
EPOCHS = 30


def featurize(history, lam=0.04):
    """Same construction as the reference featurizer in ai_docs/feature_catalog.md."""
    o = history[-1]
    n = len(history)
    util = o["utilization"]
    prev_util = history[-2]["utilization"] if n > 1 else util
    ema = 0.0
    for h in history[-8:]:
        ema = 0.7 * ema + 0.3 * h["tokens_added_last"]
    sal, integ, errs, err_after_err, prev_err = 1.0, 1.0, 0, 0, 0
    ri = sum(h["recall_miss"] for h in history) / max(1, o["step"])
    tools = shifts = 0
    prev_tool = 0.0
    for h in history[1:]:
        a = h["last_action"]
        sal = 1.0 if a == 3 else sal * math.exp(-lam * (0.5 + h["utilization"]))
        if h["topic_shift"]:
            sal *= 0.9
        if a == 1:
            integ *= 1 - 0.35 * min(1.0, 3 * ri)
        elif a == 2:
            integ *= 1 - 0.12 * min(1.0, 3 * ri)
        elif a == 4:
            integ *= 1 - 0.5 * min(1.0, 3 * ri)
        elif a == 5:
            integ = min(1.0, integ + 0.6)
        if h["last_error"]:
            errs += 1
            err_after_err += prev_err
        prev_err = h["last_error"]
        tools += int(h["tool_tokens"] > prev_tool + 0.5)
        prev_tool = h["tool_tokens"]
        shifts += h["topic_shift"]
    steps = max(1, o["step"])
    loop_p = err_after_err / max(1, errs)
    recent = history[-6:]
    rs = sum(h["last_success"] for h in recent) / len(recent)
    remaining = o["horizon"] - o["step"]
    la = [0.0] * 6
    la[o["last_action"]] = 1.0
    return [
        util, 1 - util, o["tool_fraction"], o["redundancy_est"], o["stale_est"], o["confidence"],
        o["step"] / o["horizon"], o["steps_since_instruction"] / 40, o["steps_since_compaction"] / 40,
        min(o["error_streak"], 6) / 6, o["last_error"], o["recall_miss"], o["topic_shift"], util - prev_util,
        ema / 10, min(40.0, (o["capacity"] - o["tokens_used"]) / max(0.5, ema)) / 40,
        max(0.0, util - 0.55) / 0.45, sal, integ, ri, loop_p, min(o["error_streak"], 5) / 5 * loop_p,
        tools / steps, shifts / steps, (o["progress"] + rs * remaining - 24) / 4,
        (24 - o["progress"]) / max(1, remaining), rs,
    ] + la


def _net(d: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, E.N_ACTIONS))


def train(train_episodes: list[dict]) -> dict:
    """Deterministic training on teacher-v2 soft targets (tau 1). Returns a picklable bundle."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    xs, qs = [], []
    for ei, ep in enumerate(train_episodes):
        if ei % 7 == 0:
            continue
        H = ep["history"]
        for t in range(len(H)):
            xs.append(featurize(H[: t + 1]))
            qs.append(ep["q"]["v2"][t])
    x = torch.tensor(xs, dtype=torch.float32)
    q = torch.tensor(qs, dtype=torch.float32)
    mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    xn = (x - mu) / sd
    torch.manual_seed(SEED)
    net = _net(x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.01)
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        perm = torch.randperm(xn.shape[0], generator=gen)
        for i in range(0, xn.shape[0], 256):
            idx = perm[i: i + 256]
            loss = -(torch.softmax(q[idx], 1) * torch.log_softmax(net(xn[idx]), 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    return {"state_dict": {k: v.clone() for k, v in net.state_dict().items()}, "mu": mu, "sd": sd}


class Policy:
    def __init__(self, bundle: dict):
        torch.set_num_threads(1)
        self.net = _net(len(FEATURE_NAMES))
        self.net.load_state_dict(bundle["state_dict"])
        self.net.eval()
        self.mu, self.sd = bundle["mu"], bundle["sd"]

    def act_batch(self, hists: list[list[dict]]) -> list[int]:
        x = (torch.tensor([featurize(h) for h in hists], dtype=torch.float32) - self.mu) / self.sd
        with torch.no_grad():
            return self.net(x).argmax(1).tolist()


def improvement_values(s: E.EnvState, sc: E.Scenario, history: list[dict], pol: Policy, futures: int = 16) -> list[float]:
    """Teacher t4: Q(a0) = r(a0) + return of the reference student to the episode end, mean over
    imagined futures (all rollouts advance in lockstep with one batched forward per step)."""
    futs = E.imagined_futures(s, sc, futures)
    roll = []
    for a0 in range(E.N_ACTIONS):
        for f in futs:
            c = s.clone()
            r = E.step(c, a0, f)
            roll.append([a0, f, c, list(history), r])
    while True:
        live = [x for x in roll if not x[2].done]
        if not live:
            break
        for x in live:
            x[3].append(E.observe(x[2], x[1]))
        for x, a in zip(live, pol.act_batch([x[3] for x in live])):
            x[4] += E.step(x[2], a, x[1])
    vals = [0.0] * E.N_ACTIONS
    for a0, _f, _c, _h, r in roll:
        vals[a0] += r / futures
    return vals
