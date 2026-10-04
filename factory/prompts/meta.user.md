# Meta-Agent task — wave starting at iteration {{ITERATION}}

Mutate the search space (optional but encouraged) and choose exactly {{WAVE_SIZE}} coordinate(s) to
evaluate. The factory evaluates the coordinates of one wave **in parallel**, so none of them can
learn from another's result: make them a deliberate portfolio (for example one exploitation step
around the best configuration plus different exploratory bets), not near-duplicates.
{{REFILL_NOTE}}

## Objective

Maximise `sim_score`: mean closed-loop return of the trained policy over 50 evaluation turns x 4
seeds, piecewise linear with do-nothing = 0, hand-written heuristic = 50, clairvoyant bound = 100
(unreachable; an observation-only policy is estimated to top out around 83-87). The judge score
stays within +-3 of sim_score. Validation accuracy is NOT the objective: teacher labels are noisy
near-ties and accuracy no longer predicts return.

## Domain brief (fixed)

- The MLP maps an engineered feature vector, computed from the per-step observation history of a
  simulated agent session, to logits over 6 interventions: NOOP, COMPACT, PRUNE_TOOLS,
  REINJECT_INSTRUCTIONS, CHECKPOINT_RESET, RETRIEVE_MEMORY. CHECKPOINT_RESET is rarely optimal
  (COMPACT dominates it almost everywhere); forcing rare actions has cost return in the past.
- Observation fields: step, horizon, tokens_used, capacity, utilization, tokens_added_last,
  tool_tokens, tool_fraction, redundancy_est, stale_est, steps_since_instruction,
  steps_since_compaction, last_error, error_streak, topic_shift, recall_miss, last_action,
  last_success, confidence, latency_ms, progress. Hidden state that decides outcomes: instruction
  salience, fact integrity (lost by compaction/reset/pruning, restored by retrieval), error loops
  (COMPACT clears an error streak, REINJECT halves it), the 24/40 success cliff (+10 bonus).
- The simulator's training protocol exposes these levers (the Worker sets them in model.py):
  - `TRAIN_CONFIG["teacher"]`: `v1` (one-step lookahead teacher) or `v2` (two-step lookahead
    teacher, stronger closed-loop).
  - `TRAIN_CONFIG["target"]`: `hard` (argmax labels), `soft` (cross-entropy to softmax(Q/tau) of
    the teacher's per-action values; `target_temperature` tau), or `regret` (expected regret;
    can collapse to all-NOOP). A custom `loss_fn(logits, targets, q)` receives the per-action
    teacher values `q` [B, 6].
  - `TRAIN_CONFIG["data"]`: `behaviour`; `behaviour+onpolicy` (adds ~8.6k states visited by a fixed
    harness reference student; measured no gain through the real protocol); or `behaviour+self`
    (per-experiment DAgger: a first model is trained, rolled out on `self_episodes` training-side
    scenarios (28-126, default 84), the states IT visits are labelled by the chosen teacher, and the
    final model is retrained on behaviour + self states; costs ~15-25 s of the 60 s budget).
  - `TRAIN_CONFIG["ensemble_seeds"]`: 1-5 independently seeded copies whose logits are averaged
    (default 3; a single retrain moves sim_score by about 0.5).
  - `TRAIN_CONFIG["early_stop"]`: `none` or `val_regret` (keep the lowest-validation-regret epoch;
    about +0.1 return for recipes that over-train, nothing otherwise).
  - `ai_docs/feature_catalog.md` documents a **reference recipe**: the strongest model measured so
    far (35 features, [64, 64] MLP, lr 5e-4, 30 epochs, batch 256, AdamW wd 0.01, grad clip 1.0,
    teacher v2, soft tau 0.5, behaviour+self with 84 episodes, 3 seeds). Deviations such as batch
    ~1024 with many more epochs cost about 0.27 return. Longer training never helped (no delayed
    generalisation up to 100k steps).
  - Up to 1024 features, so raw observation-history windows can feed in-model encoders
    (GRU/1-D conv/attention over the last K steps), plus belief-state features that estimate the
    hidden variables; decision calibration (NOOP margin / logit bias) baked into forward().
  - Standard training knobs: lr, epochs (capped deterministically at 10,000 optimizer steps and
    1e11 parameter-samples, shared by ensemble members), batch_size, optimizer
    (adam|adamw|sgd|rmsprop), weight_decay, grad_clip, standardize; label_smoothing and
    class_weighting only act with target=hard, target_temperature only with target=soft.
- Measured on independent development sets through the real simulator (paired): teacher v2 over
  v1 about +2.8 sim; soft over hard about +1.1; tau 0.5 over 1.0 about +0.6; 3-seed ensemble alone
  about +0.5; `behaviour+onpolicy` about 0; `behaviour+self` + tau 0.5 + 3-seed ensemble about +1.3
  over the best v4.2 recipe (self-DAgger alone with 1 seed about 0). Return-based fine-tuning was
  removed (it never improved held-out return).
- `state_space.json` contains a Python-computed `analytics` block: metric rows per iteration
  (valid runs only feed the statistics), per-dimension marginals (median/max sim_score per value),
  Spearman correlations with sim_score, paired differences against the best iteration with standard
  errors, `lever_coverage` (which values of each simulator setting have actually taken effect),
  `duplicates`, `invalid_iterations`, seed spread and stopping-rule status.

## Strategy rules

1. Read `analytics` first. Treat differences below about 2 x `typical_paired_se_sim` (or the seed
   spread) as noise.
2. The simulator is deterministic: never propose replicates or controls of an evaluated coordinate.
   A coordinate whose values have no effect (see NOTE lines and `lever_coverage`) produces a
   functionally identical model; the harness detects it and asks you for a replacement.
3. When every marginal of the existing dimensions is within noise, stop tuning hyperparameters and
   make a structural move: a new dimension that changes the training signal (teacher, target,
   data distribution, ensembles), the information available (belief-state or history-encoder
   features), or the decision rule (calibration).
4. A change aimed at one scenario family is worth at most (that family's share of the 50 turns) x
   (its gap); prefer changes that act on every family.
5. Ignore iterations listed in `invalid_iterations` when judging a lever; re-test the lever instead.
6. Prune dimensions whose marginals are flat or harmful; do not re-add a pruned dimension unless
   something material changed.
7. Each wave is a portfolio: at least one candidate exploits the best coordinate with a single
   change, at least one tests a new structural idea.
8. Until the ledger contains a model at least as good, anchor exploitation on the documented
   reference recipe (make the dimensions able to express it exactly).

## Current state space (workspace/state_space.json)

`dimensions` are active; `pruned_dimensions` were removed; `ledger` lists every evaluated
coordinate with its judge score and sim_score (`status: poisoned` = failed 3 times, never propose
it again); `analytics` summarises the evidence.

```json
{{STATE_SPACE}}
```

## Best configuration so far (workspace/best_config.json)

```json
{{BEST_CONFIG}}
```

## Most recent judge verdicts (workspace/last_judge.json; one per iteration of the previous wave)

```json
{{LAST_JUDGE}}
```

## Harness rules (violations are rejected)

1. `mutations`: at most 6 ops, applied in order:
   - `{"op": "add_dimension", "name": "<snake_case>", "spec": {"type": "categorical", "values": [...], "description": "..."}}`
     or `"type": "integer" | "float"` with numeric `"min"` and `"max"` instead of `values`.
   - `{"op": "add_values", "name": "<categorical dim>", "values": [...]}`
   - `{"op": "prune_values", "name": "<categorical dim>", "values": [...]}` (must leave at least one value)
   - `{"op": "prune_dimension", "name": "<dim>", "reason": "..."}` (at least one dimension must remain)
2. Names match `^[a-z][a-z0-9_]{1,47}$`. Descriptions are 10-800 characters and must tell the
   Worker exactly how to implement each value. At most 32 active dimensions; categorical
   dimensions hold 1-16 unique values. A value is a string (<=80 chars), number, boolean, or a list
   of up to 8 numbers.
3. `candidates` holds exactly {{WAVE_SIZE}} objects `{"next_coordinate": {...}, "hypothesis": "..."}`.
   Mutations apply before any candidate. Each `next_coordinate` must assign **every** active
   dimension (after your mutations) a value inside its domain and contain no other keys. Every
   coordinate must differ from every coordinate in the ledger and from the other candidates.
4. `analysis` (<=2000 chars): what the evidence says. Each `hypothesis` (<=1000 chars): what that
   coordinate tests and what result would confirm or refute it.

