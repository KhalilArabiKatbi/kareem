# Intervention MLP for Autonomous-Agent Context Health: Initial Concepts

## Problem framing

A long-running autonomous agent accumulates context: dialogue, tool output, retries, abandoned
sub-topics. As the window fills, quality degrades long before the hard limit: instructions lose
salience, redundant and stale material dilutes attention, and early facts are forgotten or
truncated. An *intervention policy* watches cheap telemetry each step and decides whether to act:
compact, prune tool output, re-inject instructions, reset from a checkpoint, retrieve memory, or
do nothing. Every intervention has a cost, so the policy must act at the right moment, not often.

The intervention policy here is a small MLP: telemetry history → engineered features → logits over
six actions. It is trained on targets from a lookahead teacher that sees hidden state (on
behaviour states and optionally on states a trained student visits), then judged closed-loop,
where its own mistakes compound.

**The objective is closed-loop return (sim_score), not label accuracy.** Teacher labels are noisy
near-ties: the two teachers agree on only ~59% of states, validation accuracy saturates around
0.55-0.60, and accuracy no longer predicts return. Earlier searches found that simple students
already match or beat the one-step teacher; further progress has to come from a better training
signal, better estimates of the hidden state, or a better decision rule.

## Design axes worth exploring

1. **Training signal (the largest lever so far).**
   - Teacher choice: `v1` (one-step lookahead) or `v2` (two-step lookahead, stronger closed-loop).
   - Target type: hard argmax labels, soft targets softmax(q / tau) over the teacher's per-action
     values (keeps the information about near-ties and costly mistakes), expected regret, or a
     custom `loss_fn(logits, targets, q)`.
   - Training distribution: `data="behaviour+self"` is per-experiment DAgger: the model's own
     closed-loop states are labelled by the teacher and added to training, correcting the mismatch
     between behaviour-policy states and the states its own decisions lead to. It only pays off with
     variance reduction (ensembles) and a sharper target (tau about 0.5); states from a different
     policy (`behaviour+onpolicy`) do not help.
   - Seed ensembles (`ensemble_seeds`): averaging several independently initialised copies removes
     much of the seed-to-seed noise (about 0.5 sim per retrain).
   - Past searches found that students already match or beat the teachers; gains now come from the
     training distribution, the target temperature, variance reduction, and better estimates of
     the hidden state, not from tuning optimiser hyperparameters.
2. **Hidden-state estimation (belief features).** The teacher decides from instruction salience,
   fact integrity, error-loop state and distance to the success cliff; the MLP only sees
   telemetry. Features that reconstruct those quantities (see `feature_catalog.md`, section F)
   or in-model history encoders over raw windows (GRU, 1-D convolution, small attention over the
   last K observations; up to 1024 input features) target exactly what the student lacks.
3. **Decision rule.** The simulator takes argmax. A NOOP margin (act only when the best
   intervention beats NOOP by a margin), logit biases baked into `forward`, or an ensemble of
   several small nets averaged inside one module (reduces label-noise variance) change decisions
   without changing features. Past searches found "intervene less" correlated with return.
4. **Topology.** Depth (1–4 hidden layers), width (16–512), residual blocks, bottlenecks,
   mixture-of-experts style gating, separate heads for "act?" and "which action?".
5. **Activation, normalisation, regularisation.** ReLU/GELU/SiLU/Tanh; LayerNorm/BatchNorm/none;
   dropout, weight decay, label smoothing, input noise. Small, slightly under-fit nets have done
   well because they average out label noise.
6. **Optimisation.** Learning rate, optimiser, epochs, batch size, gradient clipping (all under a
   deterministic compute budget).

## Known pitfalls

- *Forcing rare actions*: CHECKPOINT_RESET and RETRIEVE_MEMORY are rare because they are rarely
  optimal. Class weighting, focal loss and positive logit biases toward rare actions cost about 7
  sim points in an earlier search.
- *Action collapse*: predicting NOOP everywhere overflows; the regret target can collapse to it.
- *Thrashing*: repeated COMPACT/RESET wastes cost and destroys facts.
- *Covariate shift*: training states come from behaviour policies; closed-loop states may differ.
- *Noise*: a paired difference of about 1 sim point between two configurations is within
  evaluation noise, and a single retrain moves the score by about 0.5; do not chase it.
- *Slow featurisation*: training (up to ~17k states) and evaluation (8k steps) call `featurize` many times;
  keep it O(window) and well under 0.1 ms per call.

## Evaluation signals

sim_score (mean closed-loop return versus do-nothing / heuristic / clairvoyant anchors), success
rate (≥24/40 sub-tasks), overflow rate, mean hidden health, NOOP share, per-family return versus
teacher v1 / teacher v2 / clairvoyant, and validation regret (q-gap of the chosen action).
