<!-- Consultation: claude-opus-5-5, --effort xhigh, read-only repo access, 2026-09-28; 741 s, 42 turns, $3.04. Prompt: specs/optimizer/consultation_prompt.md -->

# Plateau diagnosis: context-health factory (16 iterations, halted on the plateau rule)

The search stopped for a real reason. Every dimension in the state space changes how well a small MLP fits hard argmax labels from one fixed teacher, and the best students already match or beat that teacher (33.00 vs 32.34 return). The levers that could plausibly add more than 2 points (teacher quality, value targets, return-based training) can't be expressed through the contract. On top of that, the judge's feedback pointed the Meta toward boosting rare actions, and the ledger shows that costs about 7 points.

## 1. Root causes, ranked

**1. The contract caps the reachable score (primary cause).**
- **Only one training path exists.** `sim_runner.py:207-229` does minibatch cross-entropy, or `loss_fn(logits, targets)`, on cached argmax labels. `dataset.py:36` stores only `oracle_action`. The per-action values the teacher computes (`env.py:342-358`) are thrown away. The model gets no environment access, no returns and no on-policy data.
- **The teacher is used up.** The student beats it overall (32.997 vs 32.344) and matches or beats it in 5 of 7 families. Across the 7 unweighted runs, val_acc stays in 0.558–0.580 while sim_score spans 75.5–79.5, with no relationship. Run 7 has the highest val_acc (0.580) and train_acc (0.687) but scored 76.95; run 8 (0.5625) scored 79.52.
- **The teacher is biased, not just noisy.**
  - It averages only 6 imagined futures, depth 8 (`env.py:320-322`).
  - Its bootstrap `0.7·health·min(remaining,10)` (`env.py:355`) has no terminal-bonus term, so the +10 for reaching 24/40 only becomes visible once t ≥ 32.
  - `health()` ignores integrity (`env.py:141-152`).
  - The rollout policy never breaks error loops (`env.py:305-317`).
- **These biases show up exactly where the student is weakest.**
  - long_horizon_recall: the teacher itself barely beats the heuristic (36.28 vs 36.17). That family is "below heuristic" because of the teacher, not the student.
  - debug_loop: success is 0.71. Two missed terminal bonuses cost about 0.4 return, roughly 1.7 sim points.
- **Hard-label cross-entropy learns the wrong target.** It learns the action that is most often best, not the one that is best on average. Rare but valuable actions (e.g. RETRIEVE_MEMORY after a compaction when fact density is high) get systematically under-chosen. A global logit bias, which is what the Meta tried, can't fix a problem that depends on the state.

**2. The feedback loop steered the Meta the wrong way.**
- **The rubric rewards the wrong things.** `judge.user.md:11-15` penalises low entropy, "near-dead" actions and the train/val gap. The judges docked the best runs for "CHECKPOINT_RESET = 1–2 of 2000" and for a "refuted hypothesis" (exp_008 and exp_012 `judge.json`). Every `next_steps` list recommended class weighting, focal loss or rare-action biases.
- **The ledger says the opposite.**

  | class_weighting | runs | mean sim_score |
  |---|---|---|
  | none | 7 | 77.8 |
  | sqrt_balanced | 4 | 73.4 |
  | balanced | 5 | 71.0 |

  - 9 of 16 iterations used a weighted variant.
  - Two dimensions (`rare_action_logit_bias`, `checkpoint_reset_extra_bias`) chase an action the teacher essentially never labels. exp_013 had +2.8 total bias plus balanced weighting and still used CHECKPOINT_RESET 0 times.
  - CHECKPOINT_RESET is dominated by COMPACT in almost every state: it costs 2.0 vs 0.8, adds 2 warm-up steps at 0.55× success, and multiplies integrity by (1 − 0.5·density).
- **The real signal was "intervene less".**
  - All 7 runs with ≥ 1,180 NOOPs scored ≥ 75.5; all 9 with ≤ 1,061 scored ≤ 74.6.
  - Within the 9 weighted runs alone, Spearman ρ(NOOP count, sim_score) = 0.95.
- **The Meta noticed and still followed the judge.** Its own analysis (`exp_008/meta.json`) says the loss-shaping runs "all scored lower (70/70/72)", yet it proposed more of them. Separately, `meta.user.md:17-18` frames the task as imitation and anchors "~55% is a good validation accuracy".

**3. The stopping test can't resolve differences of about 2 points.**
- **Noise floor.** Runs 4, 8, 12 and 14 share the same core recipe and scored 77.60, 79.52, 77.94 and 79.40, a standard deviation of about 1.0 sim.
  - Per-scenario differences between exp_008 and exp_014 have SD 1.82 return, so a paired comparison has a standard error of 0.26 return, about 1.1 sim.
  - One scenario (S-EV-018) swung by 6.0 return. That single scenario drives most of the conclusion that "window 9 is refuted" for long_horizon_recall.
- **The judge compresses and deflates.** Judge minus sim_score averages −2.75, and −2.6 to −4.4 for the top four runs. Judge scores span 68–76 while sim_scores span 68.8–79.5. Because scores are integers, "> 2.0" in practice means an improvement of at least 3.
- **Wave geometry.** With K=4, each 10-iteration window contains only about 2.5 Meta decisions, and the "before" set already held wave 0's score of 76.
- **Not decisive in this run.** A test on raw sim_score would also have halted (79.52 − 77.74 = 1.78). It will hide real 2–3 point gains in future runs, though.

**4. The docs contain wrong facts.**
- `simulator_contract.md:98-99` says the oracle scores 90. It actually scores about 76.8, and the clairvoyant is the 100 anchor.
- `simulator_contract.md:76` gives REINJECT_INSTRUCTIONS a cost of 0.2 and +1.5k tokens. The code uses 0.35 and +2k (`env.py:26`, `env.py:180`).
- The contract never mentions that COMPACT resets `error_streak` and REINJECT halves it (`env.py:172`, `env.py:182`). Those are the mechanics that decide debug_loop.
- Workers chose between 24 and 45 features with no coordinate controlling it, which adds an uncontrolled confounder to the ledger.

## 2. Your hypotheses

| | Verdict | Evidence |
|---|---|---|
| H1 | **Confirmed, with a refinement** | The student beats the teacher (33.00 > 32.34) and matches or beats it in 5 of 7 families, and val_acc no longer predicts score. The refinement: the cap comes from the teacher's *bias* (myopia, blindness to the terminal bonus and to integrity, a weak rollout policy), not from its sampled argmax. Students already average out the sampling noise, which is why a small, underfit SGD net (2,726 params, train_acc 0.64) beats larger nets. |
| H2 | **Partly** | The per-action values are discarded (`dataset.py:36`, `env.py:357`). But the "~45% irreducible noise" mixes three things: sampling noise from 6 futures (fixable on the teacher side), hidden-state aliasing (unfixable for the student), and near-ties (harmless). Only stored Q-vectors can separate them. |
| H3 | **Confirmed, with a caveat** | `loss_fn` sees only `(logits, targets)`; no environment, returns or on-policy data. Caveat: several levers are expressible today and were never tried: NOOP-bias or act-margin calibration, ensembles inside one module, a raw-history tensor fed to a GRU or conv encoder (≤ 256 features, about 12 steps × 21 fields), and a fixed cost-matrix loss. |
| H4 | **Partly** | The window/wave geometry and the noise level are confirmed (paired SE ≈ 1.1 sim, replicate SD ≈ 1.0). The judge deflation is real (−2.75 on average) but did not decide this run: a sim-based test also halts at 1.78. Integer scores push the effective threshold to 3. |
| H5 | **Partly; misdirection is the bigger problem** | The ledger stores only the judge score (`state_space.py:241`), and the metrics digest has no per-family data (`run_factory.py:567-580`). Per-family numbers appear only in the judge's prose. But the Meta saw that weighting hurt and still pursued it, because the judge had made "use rare actions more" the goal. |

## 3. What the state space still needs to discover

Scale: above the heuristic, 1 return point = 4.18 sim points.

| Dimension family | Why it should move sim_score | What the simulator/contract must expose | Estimated gain (sim) |
|---|---|---|---|
| **A. Target type**: hard label / soft softmax(Q/τ) / expected-regret (cost-sensitive) / Q-regression | The model learns the best action on average given what it can see, per state. This fixes rare-but-valuable actions where it matters. | Cache a 6-value Q-vector per state for each teacher. Pass it as `loss_fn(logits, targets, q)`, detected by signature. Report `val_regret` on the TRAINING line. | +1 to +2 |
| **B. Teacher choice/quality** | Removes the bias: 32–64 futures, depth 12, a terminal-aware bootstrap (10·P(progress ≥ 24)), an integrity term (recall_p·remaining·(1−integrity)·1.5), and a rollout policy that breaks error loops. | Several label/Q sets in the cache, selected with `TRAIN_CONFIG["teacher"]`. | +2 to +5, if P0-1 finds at least 1 return of headroom |
| **C. Return-based fine-tuning** (CEM or evolution strategies) | Optimises return directly, including the 24/40 cliff and the cost of each intervention. | The harness runs batched rollouts in lockstep on **training scenarios only**, with an env-step budget (≤ 300k). It must supply a seeded generator, because `manual_seed` is banned (`code_guard.py:25`). The model declares which parameters to tune (output bias or last layer). | +1 to +3 |
| **D. Belief-state features / history encoders** | Estimates the teacher's hidden inputs: salience and integrity filters, loopiness, a family posterior, P(reaching 24). | New catalog entries; `MAX_FEATURES` 256 → 1024 (`sim_runner.py:33`) so raw K×21 history tensors fit. | +0.5 to +2 |
| **E. Decision calibration and ensembles** | "Fewer interventions" correlates with higher score, and ensembles average out label noise. | Nothing; already expressible. | +0.5 to +1.5 |
| **F. On-policy data (DAgger)** | Covariate shift looks modest: closed-loop agreement 0.49 vs val_acc 0.56, and 0 overflows. | Student rollouts within the budget, relabelled by the current teacher (estimated ~12 s for 72 episodes). | 0 to +1 |
| **G. Privileged auxiliary targets** | Pushes the representation toward the hidden state. | Cache hidden salience, integrity and family per step; allow an optional auxiliary head. | +0.5 to +1 |

**Ceiling.** The clairvoyant uses the realised future, including which steps will succeed (`env.py:365-390`), so 100 is unreachable by design. My estimate for a policy that only sees observations is 34.0–34.9 return, i.e. **sim 83–87**. Reaching 90 needs 35.4 return, which I think is unlikely. The gains above overlap, so they don't simply add up. P0-1 below replaces this guess with a measurement.

## 4. Prioritised fixes

**P0: cheap; do these before the next run**

- **P0-1 Measure how much a better teacher can gain (this gates B).**
  - What: an offline pure-Python run of teacher variants closed-loop on the 50 eval scenarios, reusing the baseline code in `dataset.build_cache`. Variants:
    - 32 and 64 futures;
    - depth 12;
    - terminal- and integrity-aware bootstrap;
    - loop-breaking rollout (COMPACT when error_streak ≥ 2).
  - Also relabel the 36 validation episodes with a 64-future teacher and measure agreement with the current labels. That splits H2's "noise" into sampling noise vs aliasing.
  - Proceed with B only if the best variant reaches ≥ 33.8 return.
  - Cost: minutes of CPU. The current teacher costs roughly 4 ms per call; I inferred that from runtimes, it isn't measured.
- **P0-2 Recalibrate the judge.**
  - `judge.user.md`: score = round(sim_score) + adjustment, with |adjustment| ≤ 3, and only for problems that cost return (overflows, missed success targets, clamped budgets). Explicitly: no penalties for low entropy, rarely used CHECKPOINT_RESET/RETRIEVE_MEMORY, the train/val gap, or whether the hypothesis held.
  - Set `config.JUDGE_ANCHOR_TOLERANCE` from 15 to 3.
  - Remove `hypothesis` from the config the judge receives (`run_factory.py:494-497`).
  - Optional: allow one-decimal scores (validator at `run_factory.py:485`, plus `schemas.py`) so "> 2.0" works at its nominal resolution.
  - Validate by re-judging the 16 existing logs (16 LLM calls). Require |judge − sim| ≤ 3 and Spearman ≥ 0.95.
- **P0-3 Fix the docs.**
  - Contract: correct the anchors and the action table, and document the error-streak effects and the 24/40 cliff.
  - `meta.user.md`: remove "~55% is good".
  - `mlp_concepts.md`: state that closed-loop return is the objective, that val_acc saturates, and that CHECKPOINT_RESET/RETRIEVE_MEMORY are rare because they are rarely optimal.
  - `ai_docs` is checksummed (`run_factory.py:148`), so edit it and then start with `--fresh`.
- **P0-4 Give the Meta Python-computed analytics.**
  - Where: a new `analytics` block in `state_space.json`. That file is already whitelisted for the Meta; pass it through `sanitize_obj` and include no experiment ids.
  - Content:
    1. Ledger rows gain sim_score, mean_return, NOOP share, and per-family return minus teacher and clairvoyant minus return.
    2. Generic per-dimension marginals (for each value or bin: count, mean and max sim_score), looping over whatever dimensions exist so no names are hardcoded.
    3. Spearman correlations of sim_score with each numeric dimension, with action shares, and with val_acc.
    4. A paired delta against the current best, with its standard error.
    5. Plateau status: the score to beat and the iterations left in the window.
  - Files: new `factory/analytics.py`; hook into `stage_update`; add sim_score in `state_space.record`.
  - Validate offline on this run: the output should surface the −6.9 weighting effect, ρ = 0.95 within weighted runs, and "student ≥ teacher in 5 of 7 families".
- **P0-5 Free up time budget.** Compute `oracle_agreement` only at steps where t % 4 == 0 (`sim_runner.py:253`). The work outside training is about 12.5 s in most runs, and the policy's own featurize plus forward pass is under 1 s. So I infer the per-step teacher calls cost about 8 s, and this change saves about 6 s. It must happen before C, F, or using a more expensive teacher at eval.
- **P0-6 Hand-written probes (no LLM).** Run `sim_runner` directly on scratch folders:
  - exp_008 plus a NOOP logit bias of 0.25, 0.5 and 1.0;
  - an act-margin threshold;
  - a 5-net ensemble that averages logits.

  Accept a lever only if the paired improvement is ≥ 2 sim and at least 2 standard errors.

**P1: expand the protocol**

- **P1-1 Cache v2 (A and B).**
  - `env.oracle_values()` returns all 6 action values; `oracle_action` becomes their argmax.
  - `dataset.py` stores Q-vectors for teacher v1 and v2 plus v2 labels; bump `CACHE_VERSION`.
  - `sim_runner` accepts `TRAIN_CONFIG["teacher"]`, detects a `q` argument on `loss_fn`, and reports val_regret.
  - Validate with probes: exp_008 trained on v2 hard labels, on soft targets (τ = 0.3 and 1), and with a regret loss.
- **P1-2 Fine-tuning hook (C).**
  - After supervised training, an optional `FINETUNE` config runs CEM or evolution strategies over a declared parameter subset.
  - Rollouts run in lockstep on at most 42 training scenarios × a population of at most 16, for at most 10 generations, counted in env steps.
  - Estimated ~2 s per generation. Featurize measured 0.06 ms per call in exp_008, and one generation needs about 27k calls.
  - Validate with a bias-only probe on exp_008: training-scenario gains must carry over to eval. Run it twice; the TURN and SUMMARY lines must be identical.
- **P1-3 Rewrite the Meta prompt.** Make sim_score the stated objective, describe the new protocol features in the domain brief without naming dimensions, and add a rule: when all marginals are within noise, make structural moves instead of more hyperparameter tuning.
- **P1-4 Narrow the waves from K=4 to K=2 (`config.PARALLEL_WIDTH`).** That gives 5 Meta decisions per plateau window and halves CPU contention (the 26 s outlier was exp_015). Validate with `tests/test_orchestrator.py`.
- **P1-5 Expand the feature catalog.** Add belief-state filters and one canonical reference featurizer (removes the 24–45 feature confounder). Raise `MAX_FEATURES` to 1024.

**P2**

- **DAgger:** one round, 72 training episodes, current teacher.
- **Auxiliary privileged targets** (family G).
- **Less eval noise, still exactly 50 TURN lines:** each turn averages 4 seeds of one family (200 scenarios in total), which should roughly halve the paired standard error to about 0.55 sim. This changes what a "turn" means, so it needs your sign-off.
- **A hidden confirmation set** (seeds 95k and up), evaluated by Python after the halt and reported only in the paper. It measures the winner's curse from reusing the same 50 eval scenarios.

## 5. Risks and mitigations

- **Determinism.**
  - All new randomness must come from generators the harness owns, and budgets must be counted (env steps, generations), never timed.
  - Keep a fixed thread count and a fixed rollout order.
  - For the fine-tuning phase, make the time guard a hard failure rather than a silent truncation like the current `SAFETY_BRAKE` (which notes the result "may not be reproducible").
  - Test by running twice and comparing the logs byte for byte, excluding the timing fields.
- **60 s budget.**
  - The worst observed run is 26.3 s under 4-way contention. Adding up: −6 (P0-5), +16 (8 CEM generations), +11 (training a 4-net ensemble) gives about 47 s.
  - Cap the declared budgets so the estimated worst case stays ≤ 45 s. Clamp anything over and write a NOTE, as the harness already does for epochs.
- **Isolation.** Analytics go only into whitelisted files, as aggregate numbers passed through `sanitize_obj`. Rollouts run inside the harness; `model.py` never receives Scenario objects or eval seeds.
- **Gaming.**
  - Fine-tuning on eval scenarios must be impossible by construction: the harness picks the training seeds.
  - If the dynamics are documented formula by formula, a Worker could hard-code a planner inside `featurize`. Document effects and costs, not equations.
  - A tightly anchored judge can become a rubber stamp. That's acceptable because sim_score is deterministic; keep the ±3 allowance for real pathologies.
  - Selection on a fixed eval set inflates the best score; P2's hidden confirmation set measures that.
  - Pruning and re-adding a dimension (feature_window at iteration 12) satisfies the "new dimension" rule without real novelty. Have the analytics flag re-adds; the rule itself stays as the spec defines it.
- **Comparability across runs.** A new teacher changes the "oracle" line and the agreement metric. sim_score's anchors (NOOP, heuristic, clairvoyant) don't change, so scores stay comparable across runs. Rebuilding the cache takes minutes at preflight, outside the 60 s limit.