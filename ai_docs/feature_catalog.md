# Candidate Feature Catalog

All features are computed inside `featurize(history)` from observation dicts (see
`simulator_contract.md`). `o = history[-1]`; `W` = window length (e.g. 3, 5, 8).

## A. Instantaneous telemetry
| name | formula |
|---|---|
| util | `o["utilization"]` |
| headroom | `1 - o["utilization"]` |
| tool_frac | `o["tool_fraction"]` |
| redundancy | `o["redundancy_est"]` |
| stale | `o["stale_est"]` |
| confidence | `o["confidence"]` |
| log_tokens | `log1p(o["tokens_used"])` |
| log_latency | `log(o["latency_ms"])` |
| phase | `o["step"] / o["horizon"]` |
| progress_rate | `o["progress"] / max(1, o["step"])` |
| progress_gap | `(24 - o["progress"]) / max(1, 40 - o["step"])` (pressure to reach target) |

## B. Counters and flags
| name | formula |
|---|---|
| since_instr | `o["steps_since_instruction"] / 40` |
| since_compact | `o["steps_since_compaction"] / 40` |
| error_streak | `min(o["error_streak"], 6) / 6` |
| last_error / recall_miss / topic_shift / last_success | raw flags |

## C. Temporal dynamics (windowed)
| name | formula |
|---|---|
| util_delta | `util_t - util_{t-1}` |
| util_slope_W | least-squares slope of utilisation over the last W steps |
| growth_ema | EMA of `tokens_added_last` (alpha 0.3) |
| steps_to_overflow | `(capacity - tokens_used) / max(0.5, growth_ema)` clipped to [0, 40], scaled /40 |
| error_rate_W | mean of `last_error` over the last W steps |
| recall_miss_count_W | sum of `recall_miss` over the last W steps |
| topic_shift_count_W | sum of `topic_shift` over the last W steps |
| stale_trend_W | slope of `stale_est` over W |
| confidence_trend_W | slope of `confidence` over W |

## D. Action history
| name | formula |
|---|---|
| last_action_onehot | 6-way one-hot of `o["last_action"]` |
| action_counts | counts of each action in history / step |
| steps_since_retrieve | steps since last RETRIEVE_MEMORY (/40) |
| fact_loss_proxy | `0.35*#COMPACT + 0.12*#PRUNE + 0.5*#RESET - 0.6*#RETRIEVE` since the last RETRIEVE (clipped ≥0) |
| salience_proxy | `exp(-0.04 * steps_since_instruction)` |

## E. Interactions
| name | formula |
|---|---|
| util_x_stale | `util * stale` |
| util_x_since_instr | `util * since_instr` |
| pressure | `max(0, util - 0.55) / 0.45` |

Feature sets can be organised as named bundles (e.g. `minimal` = A-subset, `temporal` = A+B+C,
`full` = A+B+C+D+E) or as independent toggles; that choice is itself a design dimension.

## F. Belief-state estimates (reconstruct what the teacher sees)

The teacher decides from hidden quantities; these features estimate them from telemetry.

| name | construction |
|---|---|
| salience_est | recursive filter: start 1.0; after a REINJECT step reset to 1.0; otherwise multiply by `exp(-lam * (0.5 + utilization))` per step (`lam` ~ 0.02-0.09 is a design choice); drop further after `topic_shift` |
| salience_decay_rate | slope of `confidence` since the last REINJECT (instruction drift shows up as a steeper decline) |
| recall_intensity | recall_miss events per step so far (sessions with frequent recall need facts preserved) |
| integrity_est | start 1.0; multiply by `(1 - a * recall_intensity_scaled)` after COMPACT, a smaller factor after PRUNE_TOOLS, a large one after CHECKPOINT_RESET; add ~0.6 (cap 1.0) after RETRIEVE_MEMORY |
| loop_persistence | empirical P(error at t \| error at t-1) over the history (debugging loops are sticky) |
| streak_pressure | `min(error_streak, 5) / 5 * loop_persistence` |
| tool_rate / shift_rate / growth_rate | frequency of tool bursts (jumps in tool_tokens), topic shifts, mean tokens_added_last: together a soft "family" signature |
| cliff_margin | `(progress + recent_success_rate * (horizon - step) - 24) / 4` (distance to the 24/40 success cliff) |
| required_rate | `(24 - progress) / max(1, horizon - step)` |
| steps_to_overflow | `(capacity - tokens_used) / max(0.5, growth_ema)` clipped to [0, 40] / 40 |

## G. Raw history windows (for in-model encoders)

Stack the last K observations (K = 4-16) of selected numeric fields into a K x F block (pad
by repeating the first observation when the history is shorter; normalise each field to about
[0, 1]; one-hot `last_action`). Flatten it into the feature vector (up to 1024 values in total) and
let the model reshape it inside `forward` for a GRU, 1-D convolution or small attention encoder,
optionally concatenated with the engineered features above.

## Reference featurizer (optional starting point)

A compact, fast (about 0.05 ms per call) featurizer combining A-F. Workers may reuse, extend or
replace it; using it keeps feature sets comparable across experiments.

```python
import math

REF_FEATURE_NAMES = [
    "util", "headroom", "tool_frac", "redundancy", "stale", "confidence", "phase", "since_instr",
    "since_compact", "error_streak", "last_error", "recall_miss", "topic_shift", "util_delta",
    "growth_ema", "steps_to_overflow", "pressure", "salience_est", "integrity_est", "recall_intensity",
    "loop_persistence", "streak_pressure", "tool_rate", "shift_rate", "cliff_margin", "required_rate",
    "recent_success", "last_a0", "last_a1", "last_a2", "last_a3", "last_a4", "last_a5",
]


def ref_featurize(history, lam=0.04):
    o = history[-1]
    n = len(history)
    util = o["utilization"]
    prev_util = history[-2]["utilization"] if n > 1 else util
    ema = 0.0
    for h in history[-8:]:
        ema = 0.7 * ema + 0.3 * h["tokens_added_last"]
    sal, integ, errs, err_after_err, prev_err = 1.0, 1.0, 0, 0, 0
    recalls = sum(h["recall_miss"] for h in history)
    ri = recalls / max(1, o["step"])
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
```

## Reference recipe (strongest measured model so far)

Measured on development scenarios disjoint from training and evaluation, through the real
simulator protocol: this recipe beats the earlier best recipe by about +0.30 return (+1.3 sim).
A variant with the same training signal but batch size ~1024, many more epochs and 126 self
episodes was about 0.27 return worse; the settings below matter.

- Features: the reference featurizer above (33 values) plus two extras (35 total):
  - `progress_to_threshold` = clip((24 - progress) / 24, -1, 1)
  - `error_streak_decay` = error_streak x (0.5 if last_action is COMPACT or REINJECT_INSTRUCTIONS else 1.0)
- Network: MLP 35 -> 64 -> ReLU -> 64 -> ReLU -> 6 (default PyTorch initialisation).
- Training:

```python
TRAIN_CONFIG = {
    "lr": 5e-4, "epochs": 30, "batch_size": 256, "optimizer": "adamw", "weight_decay": 0.01,
    "grad_clip": 1.0, "standardize": True,
    "teacher": "v2", "target": "soft", "target_temperature": 0.5,
    "data": "behaviour+self", "self_episodes": 84, "ensemble_seeds": 3,
}
```

Training much longer does not help: closed-loop return peaks within about 500-3,000 optimizer
steps and then declines (over-fitting to near-tie teacher targets; no delayed generalisation was
found up to 100k steps). `early_stop="val_regret"` keeps the checkpoint with the lowest validation
regret; it adds about +0.1 return to recipes that train too long and nothing to this one.
