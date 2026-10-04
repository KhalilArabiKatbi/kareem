"""Deterministic mock environment for autonomous-agent context health.

A scenario is one simulated agent session of HORIZON steps. At every step the intervention policy
(the Worker's MLP) sees the observation history and picks one of six interventions. Hidden
dynamics (salience, fact integrity, true redundancy/staleness) decide whether the base agent
succeeds at its sub-task, how fast the context grows, and whether it overflows.

All exogenous randomness is pre-generated per scenario ("common random numbers"), so the
transition function is a pure function of (state, action, schedule). Same inputs, same outputs.
"""
from __future__ import annotations

import math
import random

ACTIONS = (
    "NOOP",
    "COMPACT",
    "PRUNE_TOOLS",
    "REINJECT_INSTRUCTIONS",
    "CHECKPOINT_RESET",
    "RETRIEVE_MEMORY",
)
ACTION_ABBR = ("N", "C", "P", "I", "X", "M")
N_ACTIONS = len(ACTIONS)
ACTION_COST = (0.0, 0.8, 0.25, 0.35, 2.0, 0.35)

CAPACITY = 128.0            # context window, thousands of tokens
SYSTEM_TOKENS = 4.0         # fixed system prompt + tool schemas
HORIZON = 40                # steps per scenario
TARGET = 24                 # successful sub-tasks needed for task success
TERMINAL_BONUS = 10.0
OVERFLOW_PENALTY = 6.0
RECALL_MISS_PENALTY = 0.5

PARAM_KEYS = (
    "growth", "growth_sd", "tool_p", "tool_size", "err", "shift_p", "recall_p",
    "sal_decay", "red_rate", "stale_rate", "fact_density", "retry_red", "loopiness",
)
FAMILY_PARAMS = {
    "steady_dialogue":     (2.6, 0.8, 0.20, 3.0, 0.05, 0.05, 0.06, 0.020, 0.10, 0.10, 0.3, 1.0, 0.15),
    "tool_flood":          (1.4, 0.5, 0.70, 7.0, 0.06, 0.05, 0.04, 0.025, 0.05, 0.25, 0.3, 1.0, 0.15),
    "debug_loop":          (2.2, 0.6, 0.50, 4.0, 0.18, 0.02, 0.05, 0.030, 0.20, 0.12, 0.3, 2.5, 0.70),
    "long_horizon_recall": (3.0, 0.8, 0.30, 4.0, 0.05, 0.03, 0.28, 0.020, 0.08, 0.08, 1.0, 1.0, 0.15),
    "topic_hopping":       (3.0, 0.9, 0.30, 4.0, 0.06, 0.25, 0.03, 0.030, 0.08, 0.10, 0.3, 1.0, 0.20),
    "instruction_drift":   (2.4, 0.7, 0.30, 3.0, 0.06, 0.05, 0.05, 0.085, 0.08, 0.10, 0.3, 1.0, 0.15),
}
# Hidden per-scenario jitter (log-normal sd) so that rates must be inferred from history.
JITTER = {"growth": 0.20, "tool_size": 0.25, "sal_decay": 0.40, "recall_p": 0.30, "retry_red": 0.30}
BASE_FAMILIES = tuple(FAMILY_PARAMS)
FAMILIES = BASE_FAMILIES + ("mixed_chaos",)


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class Scenario:
    """Family parameters plus the pre-generated exogenous event schedule."""

    __slots__ = ("sid", "family", "seed", "p", "g_noise", "tool_event", "tool_noise", "shift",
                 "recall", "u_success", "u_recall", "n_red", "n_stale", "n_conf", "n_lat")

    def __init__(self, sid: str, family: str, seed: int, params: dict | None = None):
        rng = random.Random(seed)
        self.sid, self.family, self.seed = sid, family, seed
        if params is not None:
            vals = tuple(params[k] for k in PARAM_KEYS)
        elif family == "mixed_chaos":
            a, b = rng.sample(BASE_FAMILIES, 2)
            w = rng.random()
            vals = tuple(w * x + (1 - w) * y for x, y in zip(FAMILY_PARAMS[a], FAMILY_PARAMS[b]))
        else:
            vals = FAMILY_PARAMS[family]
        self.p = dict(zip(PARAM_KEYS, vals))
        if params is None:
            for k, sd in JITTER.items():
                self.p[k] *= math.exp(rng.gauss(0, sd))
            self.p["err"] = _clip(self.p["err"] + rng.gauss(0, 0.03), 0.01, 0.35)
        H, p = HORIZON, self.p
        self.g_noise = [rng.gauss(0, 1) for _ in range(H)]
        self.tool_event = [rng.random() < p["tool_p"] for _ in range(H)]
        self.tool_noise = [rng.gauss(0, 1) for _ in range(H)]
        self.shift = [rng.random() < p["shift_p"] for _ in range(H)]
        self.recall = [t >= 8 and rng.random() < p["recall_p"] for t in range(H)]
        self.u_success = [rng.random() for _ in range(H)]
        self.u_recall = [rng.random() for _ in range(H)]
        self.n_red = [rng.gauss(0, 1) for _ in range(H + 1)]
        self.n_stale = [rng.gauss(0, 1) for _ in range(H + 1)]
        self.n_conf = [rng.gauss(0, 1) for _ in range(H + 1)]
        self.n_lat = [rng.gauss(0, 1) for _ in range(H + 1)]


class EnvState:
    __slots__ = ("t", "fresh", "tool", "red", "stale", "salience", "integrity", "error_streak",
                 "warmup", "prune_penalty", "progress", "since_instr", "since_compact",
                 "last_action", "last_success", "last_error", "recall_miss", "topic_shift",
                 "added_last", "overflows", "reward", "health_sum", "action_counts")

    def __init__(self):
        self.t = 0
        self.fresh = 2.0
        self.tool = 0.0
        self.red = 0.0
        self.stale = 0.0
        self.salience = 1.0
        self.integrity = 1.0
        self.error_streak = 0
        self.warmup = 0
        self.prune_penalty = 0
        self.progress = 0
        self.since_instr = 0
        self.since_compact = 0
        self.last_action = 0
        self.last_success = 1
        self.last_error = 0
        self.recall_miss = 0
        self.topic_shift = 0
        self.added_last = 0.0
        self.overflows = 0
        self.reward = 0.0
        self.health_sum = 0.0
        self.action_counts = [0] * N_ACTIONS

    def clone(self) -> "EnvState":
        c = EnvState.__new__(EnvState)
        for k in EnvState.__slots__:
            setattr(c, k, getattr(self, k))
        c.action_counts = list(self.action_counts)
        return c

    @property
    def total(self) -> float:
        return SYSTEM_TOKENS + self.fresh + self.tool + self.red + self.stale

    @property
    def done(self) -> bool:
        return self.t >= HORIZON


def health(s: EnvState) -> float:
    tot = s.total
    content = max(1e-6, tot - SYSTEM_TOKENS)
    p_util = min(1.0, max(0.0, (tot / CAPACITY - 0.55) / 0.45))
    h = (
        1.0
        - 0.55 * p_util ** 1.5
        - 0.60 * (s.red / content)
        - 0.45 * (s.stale / content)
        - 0.45 * (1.0 - s.salience)
    )
    return _clip(h, 0.03, 1.0)


def step(s: EnvState, a: int, sc: Scenario) -> float:
    """Apply intervention `a`, then let the base agent work one step. Returns the step reward."""
    p = sc.p
    t = s.t
    r = -ACTION_COST[a]
    density = p["fact_density"]
    prune_hit = 0

    # 1. intervention
    if a == 1:  # COMPACT: summarise the conversation
        s.fresh = s.fresh * 0.45 + 1.5
        s.tool *= 0.35
        s.red *= 0.10
        s.stale *= 0.25
        s.integrity *= 1.0 - 0.35 * density
        s.salience = max(0.0, s.salience - 0.05)
        s.since_compact = 0
        s.error_streak = 0  # the retry transcript is summarised away; the loop is broken
    elif a == 2:  # PRUNE_TOOLS: drop tool outputs
        prune_hit = 1 if (t > 0 and sc.tool_event[t - 1]) else 0
        s.stale *= 0.5
        s.tool *= 0.2
        s.integrity *= 1.0 - 0.12 * density
    elif a == 3:  # REINJECT_INSTRUCTIONS
        s.salience = 1.0
        s.fresh += 2.0
        s.since_instr = 0
        s.error_streak //= 2
    elif a == 4:  # CHECKPOINT_RESET: fresh context seeded by a checkpoint summary
        s.fresh, s.tool, s.red, s.stale = 3.0, 0.0, 0.0, 0.0
        s.salience = 1.0
        s.integrity *= 1.0 - 0.5 * density
        s.warmup = 2
        s.since_instr = 0
        s.since_compact = 0
        s.error_streak = 0
    elif a == 5:  # RETRIEVE_MEMORY
        s.integrity = min(1.0, s.integrity + 0.6)
        s.fresh += 3.0

    # 2. the base agent works
    h = health(s)
    p_succ = h * (1.0 - p["err"]) * (1.0 - p["loopiness"] * min(s.error_streak, 5) / 5.0)
    if s.warmup > 0:
        p_succ *= 0.55
    if prune_hit:
        p_succ *= 0.7
    success = sc.u_success[t] < p_succ
    recall_miss = 0
    if sc.recall[t] and sc.u_recall[t] > s.integrity:
        success = False
        recall_miss = 1
        r -= RECALL_MISS_PENALTY

    # 3. context growth
    added = max(0.3, p["growth"] + p["growth_sd"] * sc.g_noise[t])
    s.fresh += added
    red_add = p["red_rate"] * added
    s.red += red_add
    added += red_add
    if sc.tool_event[t]:
        tool_add = max(0.5, p["tool_size"] * (1.0 + 0.35 * sc.tool_noise[t]))
        s.tool += tool_add
        added += tool_add
    if success:
        s.error_streak = 0
        s.progress += 1
        r += 1.0
    else:
        extra = p["retry_red"] * (1.0 + 0.5 * min(s.error_streak, 4))
        s.red += extra
        added += extra
        s.error_streak += 1

    moved = p["stale_rate"] * s.tool
    s.tool -= moved
    s.stale += moved
    if sc.shift[t]:
        m1, m2 = 0.6 * s.fresh, 0.5 * s.tool
        s.fresh -= m1
        s.tool -= m2
        s.stale += m1 + m2
        s.salience = max(0.0, s.salience - 0.10)
    s.salience = max(0.0, s.salience - p["sal_decay"] * (0.5 + s.total / CAPACITY))

    # 4. overflow: the host truncates the oldest context
    if s.total > CAPACITY:
        k = (0.65 * CAPACITY - SYSTEM_TOKENS) / (s.total - SYSTEM_TOKENS)
        s.fresh *= k
        s.tool *= k
        s.red *= k
        s.stale *= k
        s.salience = max(0.0, s.salience - 0.5)
        s.integrity *= 0.5
        s.overflows += 1
        r -= OVERFLOW_PENALTY

    # 5. bookkeeping
    s.warmup = max(0, s.warmup - 1)
    s.since_instr += 1
    s.since_compact += 1
    s.last_action = a
    s.last_success = 1 if success else 0
    s.last_error = 0 if success else 1
    s.recall_miss = recall_miss
    s.topic_shift = 1 if sc.shift[t] else 0
    s.added_last = added
    s.health_sum += h
    s.action_counts[a] += 1
    s.t += 1
    if s.t == HORIZON and s.progress >= TARGET:
        r += TERMINAL_BONUS
    s.reward += r
    return r


def observe(s: EnvState, sc: Scenario) -> dict:
    """What the intervention policy is allowed to see. Hidden state stays hidden."""
    t = s.t
    tot = s.total
    content = max(1e-6, tot - SYSTEM_TOKENS)
    return {
        "step": t,
        "horizon": HORIZON,
        "tokens_used": round(tot, 4),
        "capacity": CAPACITY,
        "utilization": round(tot / CAPACITY, 5),
        "tokens_added_last": round(s.added_last, 4),
        "tool_tokens": round(s.tool, 4),
        "tool_fraction": round(s.tool / content, 5),
        "redundancy_est": round(_clip(s.red / content + 0.04 * sc.n_red[t], 0.0, 1.0), 5),
        "stale_est": round(_clip(s.stale / content + 0.07 * sc.n_stale[t], 0.0, 1.0), 5),
        "steps_since_instruction": s.since_instr,
        "steps_since_compaction": s.since_compact,
        "last_error": s.last_error,
        "error_streak": s.error_streak,
        "topic_shift": s.topic_shift,
        "recall_miss": s.recall_miss,
        "last_action": s.last_action,
        "last_success": s.last_success,
        "confidence": round(_clip(health(s) + 0.10 * sc.n_conf[t], 0.0, 1.0), 5),
        "latency_ms": round(350.0 + 9.0 * tot + 40.0 * sc.n_lat[t], 3),
        "progress": s.progress,
    }


OBS_KEYS = tuple(observe(EnvState(), Scenario("probe", "steady_dialogue", 0)).keys())


# ---- reference policies ------------------------------------------------------------------
def hidden_rollout_policy(s: EnvState, sc: Scenario) -> int:
    """Rollout policy for the oracle. Uses hidden state; never shown to the MLP."""
    util = s.total / CAPACITY
    content = max(1e-6, s.total - SYSTEM_TOKENS)
    if util > 0.85:
        return 1
    if s.integrity < 0.55 and sc.p["recall_p"] > 0.1:
        return 5
    if s.salience < 0.45:
        return 3
    if s.tool / content > 0.45 and util > 0.5:
        return 2
    return 0


ORACLE_DEPTH = 8
ORACLE_GAMMA = 0.95
ORACLE_FUTURES = 6

# Imitation teachers. All plan on hidden state over IMAGINED futures (never the true one).
#   v1: one-step improvement over the rollout policy (6 futures, depth 8)
#   v2: two-step search (best follow-up action per first action; 8 futures, depth 10)
#       measured closed-loop: about +1.0 return over v1 on the evaluation scenarios
TEACHERS = {
    "v1": {"two_step": False, "futures": 6, "depth": 8},
    "v2": {"two_step": True, "futures": 8, "depth": 10},
}


def imagined_futures(s: EnvState, sc: Scenario, n: int = ORACLE_FUTURES) -> list[Scenario]:
    """Alternative event schedules drawn from the scenario's own parameters.

    The oracle knows the hidden state and the family parameters, but it must not peek at the
    actual future random draws; otherwise its labels would be unlearnable from observations.
    """
    return [
        Scenario(sc.sid, sc.family, (sc.seed * 7919 + s.t * 104729 + k * 15485863) & 0x7FFFFFFF, params=sc.p)
        for k in range(n)
    ]


def _plan_value(s: EnvState, first: tuple, fut: Scenario, depth: int) -> float:
    c = s.clone()
    v, g = 0.0, 1.0
    for k, a in enumerate(first):
        if c.done:
            break
        v += g * step(c, a, fut)
        g *= ORACLE_GAMMA
    for _ in range(depth - len(first)):
        if c.done:
            break
        v += g * step(c, hidden_rollout_policy(c, fut), fut)
        g *= ORACLE_GAMMA
    if not c.done:
        v += g * 0.7 * health(c) * min(HORIZON - c.t, 10)
    return v


def oracle_values(s: EnvState, sc: Scenario, teacher: str = "v1") -> list[float]:
    """Expected discounted value of each first action (mean over imagined futures)."""
    cfg = TEACHERS[teacher]
    futures = imagined_futures(s, sc, cfg["futures"])
    n = len(futures)
    vals = []
    for a0 in range(N_ACTIONS):
        if cfg["two_step"]:
            vals.append(max(sum(_plan_value(s, (a0, a1), f, cfg["depth"]) for f in futures) / n
                            for a1 in range(N_ACTIONS)))
        else:
            vals.append(sum(_plan_value(s, (a0,), f, cfg["depth"]) for f in futures) / n)
    return vals


def argmax_first(vals: list[float]) -> int:
    best_a, best_v = 0, -math.inf
    for a, v in enumerate(vals):
        if v > best_v + 1e-9:  # ties keep the lower action index
            best_a, best_v = a, v
    return best_a


def oracle_action(s: EnvState, sc: Scenario, teacher: str = "v1") -> int:
    """Teacher action: argmax of oracle_values (common random numbers across actions)."""
    return argmax_first(oracle_values(s, sc, teacher))


CLAIRVOYANT_DEPTH = 12


def clairvoyant_action(s: EnvState, sc: Scenario) -> int:
    """Upper-bound reference: two-step exhaustive search over the TRUE future schedule.

    It sees information no observation-based policy can have, so it is only used as the top
    score anchor, never as a training label.
    """
    vals = [-math.inf] * N_ACTIONS
    for a0 in range(N_ACTIONS):
        for a1 in range(N_ACTIONS):
            c = s.clone()
            v = step(c, a0, sc)
            g = ORACLE_GAMMA
            if not c.done:
                v += g * step(c, a1, sc)
                g *= ORACLE_GAMMA
            for _ in range(CLAIRVOYANT_DEPTH - 2):
                if c.done:
                    break
                v += g * step(c, hidden_rollout_policy(c, sc), sc)
                g *= ORACLE_GAMMA
            if not c.done:
                v += g * 0.7 * health(c) * min(HORIZON - c.t, 10)
            if v > vals[a0]:
                vals[a0] = v
    best = max(vals)
    return vals.index(best)


def heuristic_policy(history: list[dict]) -> int:
    """Observation-only threshold rules: the baseline a human would write."""
    o = history[-1]
    if o["utilization"] > 0.85 or o["error_streak"] >= 4:
        return 1
    if o["recall_miss"]:
        return 5
    if o["steps_since_instruction"] >= 12:
        return 3
    if o["tool_fraction"] > 0.45 and o["utilization"] > 0.5:
        return 2
    return 0


def noop_policy(history: list[dict]) -> int:
    return 0


def run_episode(sc: Scenario, policy, record: bool = False):
    """Run one scenario closed-loop. `policy(history, state) -> action`."""
    s = EnvState()
    history: list[dict] = []
    trace = []
    while not s.done:
        history.append(observe(s, sc))
        a = int(policy(history, s))
        if not 0 <= a < N_ACTIONS:
            raise ValueError(f"policy returned invalid action {a}")
        if record:
            trace.append((s.clone(), a))
        step(s, a, sc)
    return s, history, trace


def train_scenarios(n_per_family: int = 36, base_seed: int = 10_000) -> list[Scenario]:
    out = []
    for fi, fam in enumerate(FAMILIES):
        for k in range(n_per_family):
            out.append(Scenario(f"S-TR-{fi}{k:03d}", fam, base_seed + 1000 * fi + k))
    return out


EVAL_TURNS = 50
SEEDS_PER_TURN = 4


def eval_turns(n_turns: int = EVAL_TURNS, seeds: int = SEEDS_PER_TURN, base_seed: int = 90_000) -> list[list[Scenario]]:
    """50 evaluation turns; each turn = `seeds` scenarios of one family (variance reduction)."""
    return [
        [Scenario(f"S-EV-{k + 1:03d}-{j}", FAMILIES[k % len(FAMILIES)], base_seed + seeds * k + j) for j in range(seeds)]
        for k in range(n_turns)
    ]


def eval_scenarios() -> list[Scenario]:
    return [sc for turn in eval_turns() for sc in turn]


CONFIRM_SEED = 95_000


def confirmation_turns() -> list[list[Scenario]]:
    """Hidden confirmation set: same layout as the evaluation turns, seeds disjoint from every other set.
    Never used for selection during a run; only to re-score the top models after the halt."""
    return [
        [Scenario(f"S-CF-{k + 1:03d}-{j}", FAMILIES[k % len(FAMILIES)], CONFIRM_SEED + SEEDS_PER_TURN * k + j)
         for j in range(SEEDS_PER_TURN)]
        for k in range(EVAL_TURNS)
    ]


def finetune_scenarios(n: int, base_seed: int = 50_000) -> list[Scenario]:
    """Scenarios for return-based fine-tuning: disjoint from evaluation seeds, balanced across families."""
    return [Scenario(f"S-FT-{k:03d}", FAMILIES[k % len(FAMILIES)], base_seed + k) for k in range(n)]
