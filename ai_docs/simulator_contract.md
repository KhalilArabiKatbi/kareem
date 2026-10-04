# Simulator Contract for `model.py`

The deterministic simulator imports `model.py`, trains it on targets from a selectable lookahead
teacher (on behaviour states, optionally plus on-policy states, optionally as a seed ensemble),
then runs it closed-loop on 50 evaluation turns x 4 seeds (200 scenarios). You are writing
`model.py` only. The objective is closed-loop return (sim_score), not label accuracy.

## Required module API

```python
FEATURE_NAMES: list[str]            # 1..1024 names, one per value returned by featurize()

def featurize(history: list[dict]) -> list[float]:
    """history = every observation of the current scenario up to and including the current step
    (history[-1] is 'now'). Return exactly len(FEATURE_NAMES) finite floats."""

def build_model(input_dim: int, n_actions: int) -> torch.nn.Module:
    """input_dim == len(FEATURE_NAMES); n_actions == 6. forward(x: FloatTensor[B, input_dim])
    must return logits FloatTensor[B, n_actions]."""
```

Optional:

```python
TRAIN_CONFIG: dict = {
    "lr": 1e-3,                 # clamped to [1e-5, 0.1]
    "epochs": 40,               # clamped to [1, 300]; then capped deterministically so that
                                # optimizer steps <= 10,000 and params x samples seen <= 1e11
    "batch_size": 256,          # clamped to [16, 2048]
    "optimizer": "adam",        # adam | adamw | sgd (momentum 0.9) | rmsprop
    "weight_decay": 0.0,        # clamped to [0, 0.1]
    "label_smoothing": 0.0,     # clamped to [0, 0.3]; ignored when loss_fn is defined
    "class_weighting": "none",  # none | balanced | sqrt_balanced; ignored when loss_fn is defined
    "grad_clip": 0.0,           # 0 disables; clamped to [0, 100]
    "standardize": True,        # simulator z-scores features using training-set mean/std
    "teacher": "v1",            # v1 = one-step lookahead teacher; v2 = two-step lookahead (stronger)
    "target": "hard",           # hard = argmax labels (cross-entropy, uses class_weighting/label_smoothing)
                                # soft = cross-entropy to softmax(q / target_temperature)
                                # regret = expected regret sum_a p(a) * (max q - q[a]); can collapse
    "target_temperature": 0.5,  # clamped to [0.01, 10]; used by target="soft" only
    "data": "behaviour",        # behaviour = ~8.6k states from behaviour policies (teacher/random/heuristic)
                                # behaviour+onpolicy = plus ~8.6k states visited by a fixed harness
                                #   reference student, labelled by the same teachers
                                # behaviour+self = per-experiment DAgger: train once, roll THIS model out
                                #   on self_episodes training-side scenarios, label the states it visits
                                #   with the chosen teacher, retrain on behaviour + self states
    "self_episodes": 84,        # 28..126; used by data="behaviour+self" (~40 states each; labelling
                                #   costs ~0.13 s per episode of the 60 s budget)
    "early_stop": "none",       # none | val_regret: keep, per ensemble member, the epoch checkpoint
                                #   with the lowest validation regret (teacher-value gap of the choice)
    "ensemble_seeds": 3,        # 1..5 copies of build_model trained with different seeds; logits averaged.
                                #   The compute budget is shared by the members.
}

# Either signature is accepted (detected by parameter count). When loss_fn is defined it replaces
# the built-in target. q = the chosen teacher's per-action values (FloatTensor[B, 6]); higher is
# better; differences between actions are typically 0.05-2.0 return units.
def loss_fn(logits, targets):        ...
def loss_fn(logits, targets, q):     ...
```

Settings that have no effect are reported as `NOTE | ignored: ...` lines and are left out of the
effective configuration: class_weighting and label_smoothing apply to `target="hard"` only,
target_temperature to `target="soft"` only, and a `loss_fn` replaces all built-in target settings.
`FINETUNE` is no longer supported (it did not improve held-out return) and is ignored.

Before simulating, the harness fingerprints the model (effective configuration, standardised
features, initial parameters and logits, loss). A model that is functionally identical to an
earlier experiment is not simulated again; its slot goes back to the Meta-Agent. A featurize that
makes more than max(3, 10%) of the features constant over the training states is rejected.
## Observation dict (one per step; keys are always present)

| key | type | meaning |
|---|---|---|
| `step` | int | 0-based step index (0..39) |
| `horizon` | int | always 40 |
| `tokens_used` | float | context tokens in thousands (includes 4k system tokens) |
| `capacity` | float | always 128.0 (thousands of tokens) |
| `utilization` | float | tokens_used / capacity; above 1.0 the host truncates (overflow) |
| `tokens_added_last` | float | tokens added by the previous step |
| `tool_tokens` | float | tool-output tokens currently in context |
| `tool_fraction` | float | tool tokens / non-system tokens |
| `redundancy_est` | float | noisy estimate of the redundant fraction (retries, repeats) |
| `stale_est` | float | noisy estimate of the stale fraction (old tool output, abandoned topics) |
| `steps_since_instruction` | int | steps since instructions were last re-injected |
| `steps_since_compaction` | int | steps since the last COMPACT or CHECKPOINT_RESET |
| `last_error` | 0/1 | the previous sub-task failed |
| `error_streak` | int | consecutive failed sub-tasks |
| `topic_shift` | 0/1 | the previous step changed topic |
| `recall_miss` | 0/1 | the previous step needed an early fact that was missing |
| `last_action` | int | intervention chosen at the previous step (0..5) |
| `last_success` | 0/1 | the previous sub-task succeeded |
| `confidence` | float | noisy self-reported confidence of the base agent (0..1) |
| `latency_ms` | float | noisy response latency (grows with context size) |
| `progress` | int | sub-tasks completed so far |

Hidden (never observed): true instruction salience, fact integrity, true redundancy/staleness, the
scenario family, and future events.

## Actions (index = logit position)

| idx | action | effect | cost |
|---|---|---|---|
| 0 | NOOP | nothing | 0.0 |
| 1 | COMPACT | summarise: shrinks context by about half, removes most redundancy and staleness, **clears the error streak**, loses facts (heavily in recall-heavy sessions), small salience loss | 0.8 |
| 2 | PRUNE_TOOLS | drops most tool output and some staleness; hurts the next step if the last step's tool output was still needed; small fact loss | 0.25 |
| 3 | REINJECT_INSTRUCTIONS | restores instruction salience fully, **halves the error streak**, adds about 2k tokens | 0.35 |
| 4 | CHECKPOINT_RESET | fresh context from a checkpoint: clears everything and the error streak, large fact loss, 2 warm-up steps at reduced success; COMPACT is cheaper and better in almost every state | 2.0 |
| 5 | RETRIEVE_MEMORY | restores a large part of lost facts, adds about 3k tokens | 0.35 |

Reward per step: +1 per successful sub-task, minus action cost, −0.5 per recall miss, −6 per
overflow; +10 at the end if at least 24/40 sub-tasks succeeded (a cliff: sessions near 24 matter
most). Success probability falls with context pressure, redundancy, staleness, lost salience and a
running error streak (error loops are self-reinforcing, most strongly in debugging sessions);
recall steps fail when the needed fact was lost.

## Scenario families (hidden label, 7 families)

`steady_dialogue`, `tool_flood`, `debug_loop`, `long_horizon_recall`, `topic_hopping`,
`instruction_drift`, `mixed_chaos` (random blend). Training uses different seeds from evaluation.

## Training procedure (performed by the simulator)

1. ~10k states from 252 training episodes (behaviour policies mix teacher, random and heuristic
   actions). Every state carries both teachers' per-action values q and argmax labels. Every 7th
   episode is validation. Both teachers plan on hidden state over imagined futures; their labels
   agree only ~59% of the time because many actions are near-ties, so validation accuracy tops
   out around 0.55-0.60 and does not predict return. `val_regret` (mean q-gap between the
   teacher's best action and the model's choice) is reported as well.
   With `data="behaviour+onpolicy"`, ~8.6k more training states come from rollouts of a trained,
   harness-owned reference student (on-policy / DAgger data), labelled by the same teachers.
2. Features are computed with your `featurize`, z-scored (unless `standardize` is False).
3. Mini-batch training with the optimiser from `TRAIN_CONFIG`; ensemble member k uses seed 1234 + k.
4. Evaluation: argmax of (averaged) logits at every step of 200 scenarios (50 turns x 4 seeds, one
   family per turn), closed-loop, batched across scenarios.

Measured effects on independent development scenarios (disjoint from training and evaluation,
paired): teacher v2 vs v1 about +2.8 sim; soft vs hard targets about +1.1 sim; target_temperature
0.5 vs 1.0 about +0.6 sim; 3-seed ensemble about +0.5 sim; `behaviour+onpolicy` about 0 through
the real protocol (DAgger data must come from the model being trained);
`behaviour+self` + temperature 0.5 + 3-seed ensemble about +1.3 sim over the best earlier recipe.
One retrain with a different seed moves sim_score by about 0.5 (standard deviation).

## Score anchors (sim_score)

Piecewise linear in mean episode return: do-nothing policy = 0, hand-written threshold heuristic
= 50, clairvoyant search over the true future = 100 (unreachable by any observation-based
policy). Reference points: teacher v1 is about 74, teacher v2 about 76; an observation-only policy
is estimated to top out around 83-87.

## Hard limits (violations fail the experiment)

- Allowed imports only: `torch`, `math`, `typing`, `collections`, `dataclasses`, `functools`,
  `itertools`, `numbers`, `statistics`, `numpy`, `enum`, `abc`, `__future__`.
- No file, network, process, or dynamic-code access (`open`, `exec`, `eval`, `__import__`, ...).
- Do not call `torch.manual_seed`, `torch.set_num_threads`, `.cuda()`, `torch.load/save`,
  `torch.compile`, `torch.hub`. CPU only. No `while True` loops.
- `featurize` must be fast: the ~10,000 training calls must finish within 12 s total, and
  evaluation calls it 8,000 more times, and `data="behaviour+onpolicy"` doubles the training calls
  (aim for < 0.1 ms/call).
- At most 5,000,000 parameters in total over all ensemble members. Whole simulation (import +
  featurize + train + eval) ≤ 60 s.
- `featurize` must handle `len(history) == 1` (first step) and never return NaN/Inf.
- The model must not keep state between forward calls; features carry all temporal context.

## Minimal valid example

```python
import torch
import torch.nn as nn

FEATURE_NAMES = ["utilization", "tool_fraction", "steps_since_instruction"]

def featurize(history):
    o = history[-1]
    return [o["utilization"], o["tool_fraction"], o["steps_since_instruction"] / 40.0]

class Net(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, n))

    def forward(self, x):
        return self.body(x)

def build_model(input_dim, n_actions):
    return Net(input_dim, n_actions)

TRAIN_CONFIG = {"lr": 1e-3, "epochs": 30, "batch_size": 256, "optimizer": "adam"}
```
