<!-- Consultation 2: claude-opus-5-5, --effort xhigh, read-only repo access, 2026-09-29; 1019 s, 43 turns, $3.38. Prompt: specs/optimizer/consultation2_prompt.md -->

# v4.2 plateau: second consultation

I checked everything below against the code and the v4.2 artefacts using read-only tools. Figures marked "est." are estimates I have not measured.

**Bottom line.** The only lever that moved the score was already used in wave 0. Iteration 1 (teacher v2, soft targets) scored 80.05, and the final best is only +0.70 ± 0.89 sim above it. After that, 4 of the 10 remaining slots told the Meta nothing: three were exact duplicates and one was silently broken. The broken run then distorted the analytics the Meta reads. Getting past the plateau needs a better teacher signal that acts on every family and adds at least 0.45 return (the 2-point rule). Features, fine-tuning and single-family fixes each fall short of that on their own.

| Obs | Verdict |
|---|---|
| O1 | **Confirmed** for features and history encoders. Temperature and class weighting were never validly tested (see 1.2). |
| O2 | **Confirmed.** Three separate causes (see 1.2). |
| O3 | **Confirmed** in all 11 valid runs. Fully closing it is worth only +0.64 sim, and the gap is mostly missing information. |
| O4 | **Both.** The search is underpowered, and the available gain is small. |
| O5 | **Real but not decisive.** The judge averages −0.84 below sim_score, yet a test on raw sim_score also halts. |
| O6 | **Not caused by the GRU.** Most likely a cold start. |

## 1. Diagnosis, ranked

**1.1 The teacher signal is used up (main cause).**
- **What the teacher's values measure.** Each action's value (Q) is the return from taking a0 (and a1), then following `hidden_rollout_policy` (`env.py:305-317`, `346-376`).
  - Planning depth is 10 with discount γ = 0.95.
  - The end-of-rollout estimate `0.7·health·min(rem,10)` (`env.py:360`) ignores both the +10 bonus for reaching 24/40 and integrity.
  - So the student can learn at most one improvement step over a crude threshold rule.
- **Why the student beats both teachers.** With τ = 1 the soft targets are close to uniform (final loss 1.69 vs ln 6 = 1.79). In that regime the network effectively ranks actions by their average teacher value given what it can observe, which is the right decision when states look alike.
  - This averages away the teachers' 6–8-future sampling noise.
  - Result: 33.16 return vs 32.14 (v2) and 31.73 (v1).
- **Nothing inside imitation changes the teacher's values.**
  - The 7 distinct valid v2-soft models all fall in 32.78–33.16 return.
  - Every paired difference from the best is within 1.5 standard errors (−0.62 to −1.69 sim, SE 0.72–1.15).

**1.2 Forty percent of the slots after wave 0 were wasted, and the analytics were distorted.**

| Slots | Cause |
|---|---|
| 5 ≡ 1 | The Meta re-proposed the baseline, filling the new dimensions with their defaults (`history_encoder: none`, `finetune: false`). |
| 8 ≡ 7, 10 ≡ 9 | `class_weighting` silently does nothing with soft targets. The weighted loss is only used in the `hard` branch (`sim_runner.py:316-324`); `label_smoothing` is the same. The log still prints `class_weighting=sqrt_balanced`. |
| 11 | A Worker bug. `_safe(lambda: _ref_features(history), None)` (`exp_011/model.py:127`) calls `float(list)`, which raises TypeError and returns None. That sets 33 of 35 features to a constant 0 (`:128-129`). `featurize_all` only checks for NaN/Inf (`sim_runner.py:115-117`). |

- **Why the duplicates got through.** The uniqueness check hashes the coordinate, not the model (`state_space.py:224-228`). Every added dimension makes every old model eligible again.
- **Why exp_011 distorted the analytics.** Marginals are plain means with no validity filter (`analytics.py:89-103`), so exp_011 produced three false signals:
  - `class_weighting` none 62.2 vs sqrt_balanced 80.38;
  - τ 0.5 = 44.38;
  - `finetune` last_layer 74.27.
- **Consequences.**
  - The final wave (iterations 10 and 11) yielded no information.
  - The tau test was never actually run: exp_011 failed because of the featurize bug, not because of τ = 0.5.

**1.3 CEM fine-tuning cannot detect gains this small.**
- **On healthy base models:** 7 attempts, 1 accepted, worth +0.04 return on evaluation (archived run, 80.36 → 80.56).
- **Last-layer search finds nothing.** In all four runs on healthy models (exp_007–010) the result was "chosen=base": none of 96 candidates beat the base model on 28 scenarios. With 390 parameters, population 12, σ = 0.1 and 3 elites, the search is a random walk.
- **The apparent gains were selection noise:**
  - exp_004: +1.07 train became −0.93 on holdout.
  - exp_003: +0.27 became −0.89.
  - The aborted exp_002: +0.53 became −0.39.
  - exp_004's +1.07 is about what the best of 32 do-nothing candidates would show with a standard error of ~0.5 return on 14 scenarios.
- **The method does work when there is room.** On the broken exp_011 it gained +5.0 on holdout.
- **The holdout gate is too small to judge small gains.** 21 scenarios give a standard error of about 0.4–0.6 return, so a true gain under ~0.3 is a coin flip.
- **Cost.** Throughput is 35 µs per decision (168 µs for the GRU model). FINETUNE appeared in 7 of the 10 post-wave-0 coordinates.

**1.4 Feedback still points at the wrong gaps.**
- **The judge only marks down.** Compared with round(sim), it adjusted downward in 10 of 12 iterations and never upward.
  - It docked the best run for debug_loop "fragility" and for a rejected fine-tune, which costs no return (`judge.user.md:12-15`).
  - Yet the student beats v2 on debug_loop by +3.56.
- **The analytics highlight long_horizon_recall** (`family_minus_teacher_v2`, `analytics.py:83`), which is only 7 of 50 turns. Closing its 1.04 gap entirely is worth +0.146 return (+0.64 sim).
  - The gap is mainly information. The student's integrity estimate guesses fact density from the recall-miss rate, `min(1,3·ri)` (`exp_007/model.py:67-71`). That density only becomes visible after recalls have already failed.
- **Gaps to the clairvoyant, by family:**

| Family | Gap (return) |
|---|---|
| topic_hopping | 5.44 |
| long_horizon_recall | 4.73 |
| mixed_chaos | 4.69 |
| tool_flood | 4.45 |
| debug_loop | 4.09 |
| instruction_drift | 4.09 |
| steady_dialogue | 3.20 |

**1.5 Stopping arithmetic (O5).**
- The rule would halt even with exact scores: judge = round(sim) gives 81 vs 80, and raw sim_score gives 80.75 − 80.05 = 0.70.
- Any run must beat wave 0 by more than 2 sim, which is more than 0.45 return.

**1.6 The 50.6 s runtime (O6).**
- The same GRU design ran in 12.3 s in exp_006.
- exp_002 spent 41.8 s outside featurize (1.03 s) and training (7.75 s).
- It was the first simulator process after the 02:11 resume, and its wave partner was still in its Worker stage (`factory.log:54-58`).
- This points to a cold start (loading torch, antivirus scanning, unpickling the cache). There are no per-phase timers to prove it.

## 2. Headroom

| Policy | Return | sim |
|---|---|---|
| Heuristic | 26.20 | 50 |
| Teacher v1 / v2 | 31.73 / 32.14 | 74.5 / 76.2 |
| Best student | 33.16 | 80.75 |
| Clairvoyant | 37.52 | 100 |

The remaining 4.36-return gap has four sources:
- knowledge of the future, which is irreducible;
- the hidden state;
- teacher bias (the rollout rule, the missing success-bonus term, γ, 8 futures);
- optimisation and seed noise.

- **Missed success bonuses cost at most 0.45 return overall** (4.5% × 10), i.e. 2.0 sim. They are concentrated in debug_loop (success 0.86) and topic_hopping (0.89), exactly where the teacher's missing bonus term hurts.
- **Most hidden variables can be reconstructed.** Salience and integrity follow deterministically from the action history plus a few per-scenario parameters. The exception is recall_p and fact density.

**Estimates (est., unmeasured):**
- best policy that sees the hidden state but not the future: 35 ± 0.7 return;
- best observation-only policy: 34.0–34.8 (sim 83.5–87);
- realistic for the next run: 33.8–34.3 (sim 83.6–86), and only through better teacher values.

**Offline probes (no LLM).** All of them:
- use a new dev set of 280 scenarios (seeds 70,000+, 40 per family), never the 200 evaluation scenarios, because v2 was already chosen on those (plan §9);
- compare paired against the exp_007 recipe, averaging 3 training seeds for student probes;
- report the difference in return ± its standard error, per family.

Costs are estimates for 8 cores, assuming v2 costs about 20 ms per state.

| Id | Probe | Cost (est.) | Accept if |
|---|---|---|---|
| P-F | exp_007 recipe × 5 torch seeds | 1 min | Measure the spread. If SD ≥ 0.5 sim, most ledger differences are seed luck. |
| P-C | Relabel the 1,440 validation states with v2 using 64 futures. Measure its agreement with v2 at 8 futures, and the regret of v2@8 and of the student under the 64-future values. | 2 min | Agreement < 0.75, or student regret < v2@8 regret. Then rebuild all value sets with ≥ 32 futures. |
| P-A | Teacher ladder. **T1:** v2 with 64 futures. **T2:** T1 + γ = 1 and rollouts to the end of the episode (exact success bonus). **T3:** T2 + a rollout rule with about 10 thresholds tuned by CEM, including COMPACT when the error streak ≥ k. **T4:** one improvement step over the exp_007 student as the rollout policy (imagined futures, batched 6×16). | 5–10 min each | The exp_007 recipe retrained on that teacher gains ≥ +0.45 return at ≥ 2 SE. Judge a teacher by the student's return, never the teacher's own. |
| P-B | Privileged student with (i) family one-hot + scenario parameters, (ii) plus salience, integrity and true redundancy/staleness. Hidden states can be recovered by replaying cached episodes using `history[t+1].last_action`. | 3 min | (ii) minus the observation-only student ≥ 0.8 means information is the limit. Below 0.3, stop feature work. |
| P-D | DAgger: run exp_007 on 252 training seeds, label the visited states with v2, add them to the data, retrain. The current data comes from v1/heuristic behaviour with 20–50% random actions (`dataset.py:37-53`). | 5 min | ≥ +0.3 return at ≥ 2 SE. |
| P-E | CEM ceiling: a 6-parameter bias and a 48-parameter bias gated by features, 256 scenarios per candidate, antithetic pairs. Also run in-sample on the dev set as an upper bound. | 10 min | Dev gain ≥ +0.25 with a budget that fits in ≤ 35 s. Otherwise delete FINETUNE. |
| P-G | On the current cache: advantage regression (MSE between centred logits and centred Q), τ ∈ {0.5, 2, 4}, and a 5-net logit ensemble. | 3 min | ≥ +0.3 at ≥ 2 SE. |

T4 is the key probe. Its targets assume an observation-only follow-up policy, so they avoid a trap in T3, whose values assume follow-up actions the student cannot reproduce. If T4 works, repeat it (approximate policy iteration) until a round adds less than 0.2 return.

## 3. Plan to break the plateau

**P0: harness hygiene (no LLM, before the next run)**
1. **Validity gates in `sim_runner`.**
   - Count constant columns in the training feature matrix; above max(3, 10%) of features, raise `ModelContractError` so the Worker retries.
   - Log a NOTE whenever a knob is ignored: class_weighting or label_smoothing under soft targets, τ under hard targets.
   - Write an `effective_config` block to `sim_metrics.json`.
2. **Functional fingerprint.** Add a separate `--fingerprint` mode (~2 s, outside the 60 s limit) that hashes:
   - the effective config;
   - the feature matrices `x_tr` and `x_va`;
   - the initial parameters and the logits on 1,024 fixed rows;
   - the value of `loss_fn` on a fixed batch;
   - the effective FINETUNE.

   On a match with iteration j:
   - skip the simulation;
   - return the slot to the Meta as a validation error ("same model as iteration j; values {diff} had no effect"), at most 2 refills per wave;
   - only a distinct model closes an iteration, so the Meta-Done rules keep their meaning;
   - after the refill cap, record the slot as `duplicate_of: j`;
   - Python also derives generic `no_effect` facts for the analytics.
3. **`analytics.py`.**
   - Leave flagged runs, and runs scoring below the heuristic, out of the marginals.
   - Use medians.
   - Compute a correlation only when at least 2 values each have n ≥ 2. The current learning-rate correlation of −0.395 rests on a single run.
   - Replace "family minus teacher_v2" with "family gap to clairvoyant × turn weight".
   - Add `lever_coverage`: for each contract key, the effective values tried so far. This uses the contract's own vocabulary, not dimension names.
   - Add the seed spread from P-F.
4. **Judge.**
   - Allow one-decimal scores (validator at `run_factory.py:509`).
   - Rubric: score = sim_score, minus 1–3 only for deterministic flags (validity or clamp NOTEs, overflow rate > 2%).
   - Remove the triggers for success below 1.00, rejected fine-tunes, and families below teacher_v2.
5. **Timing.** A warm-up simulation at preflight, plus per-phase timers in the RUNTIME line.
6. **`meta.user.md`.** Add three rules:
   - The simulator is deterministic, so never propose replicates or controls.
   - A single-family fix is worth at most turn weight × gap.
   - Ignore runs flagged invalid, and re-test the lever instead.

**P1: new training signal (gated by the probes; cache v3)**
1. **Teacher menu.**
   - `env.TEACHERS` gains the variants that pass P-A, with γ, depth ("to end of episode"), number of futures and rollout policy as parameters.
   - `dataset._train_episode` stores the values for each teacher.
   - `TRAIN_CONFIG["teacher"]` accepts a name or a list (averaged values).
   - The contract quotes each teacher's closed-loop return on the dev set only.
   - Preflight cost is about 10–30 min (est.), outside the 60 s limit.
2. **Harness reference student** (`simulator/reference_policy.py`).
   - The reference featurizer plus the soft-v2 recipe, trained deterministically on 1 thread.
   - It supplies T4's rollout policy and the on-policy states for item 3.
   - It is part of the cache fingerprint, so the cache still depends only on source code and never on an LLM-written file.
3. **Data menu.** `TRAIN_CONFIG["data"]` = `behaviour` or `behaviour+onpolicy`, if P-D passes. Featurizing stays at ≤ 3 s.
4. **Targets.** A built-in `advantage` target if P-G passes. Unlike `regret`, whose gradient vanishes, it cannot collapse to all-NOOP.
5. **Ensembles.** `TRAIN_CONFIG["ensemble_seeds"]` from 1 to 5, averaging logits inside one module. Make it K = 3 by default if P-F finds SD ≥ 0.5 sim.
6. **FINETUNE.** Delete it if P-E fails. Otherwise:
   - ≤ 48 parameters;
   - antithetic evolution strategies on ≥ 120 scenarios with common random numbers;
   - a 100-scenario holdout;
   - acceptance only above 2 standard errors;
   - at most 1.0M decisions.

**How the Meta finds these levers.** The contract and domain brief describe the menus with their dev-set effect sizes, and `lever_coverage` shows which ones have never been tried. Strategy rule 2 already requires structural moves once the marginals are flat. No dimension names go into Python.

**What to expect.** The Meta will pick the strongest documented teacher in wave 0 again, so wave 0 will jump. A plausible landing is 82–84 sim (est.). Staying past the plateau rule then needs another +0.45 return from combining data, teacher mixtures, targets and ensembles; I estimate +0.3 to +0.8 (est.). If the run still halts after a large wave-0 jump, that is fine: the rule measures progress within a run, and what matters is the final score.

**P2**
- Repeat T4 in the harness across releases (policy iteration, no LLM).
- Add auxiliary heads for hidden variables if P-B's (ii) minus (i) is at least 0.5.
- Add a hidden confirmation set (280 scenarios, seeds 95,000+), scored after the halt and reported only in the paper. It is still not implemented: nothing under `factory/` references it.

## 4. Risks and mitigations

- **Determinism.**
  - Build the cache with torch on 1 thread, deterministic algorithms and a fixed seed; hash the value arrays and check that two builds match.
  - Fingerprints must hash standardised tensors in a fixed row order.
- **Time budget.**
  - The fingerprint runs outside the 60 s limit.
  - K = 3 ensembles triple training time (now 1–8 s); cap them with the existing parameter-sample budget.
  - The warm-up removes the ~40 s cold starts, and the phase timers explain any future outlier.
- **Isolation.**
  - The Meta sees only aggregates; fingerprint differences name dimensions, never code.
  - The reference student and all rollouts are owned by the harness.
- **Gaming.**
  - Duplicate refills could be used to shop for coordinates: keep the cap at 2 per wave and log each refill in `factory_report`.
  - Describe teachers qualitatively and quote dev-set numbers only. Publishing T3's thresholds would let a Worker hard-code a planner into `featurize`.
- **Overfitting the 200 evaluation scenarios.**
  - About 20 models selected on a fixed set implies roughly +0.5–1 sim of winner's curse; the confirmation set measures it.
  - All teacher and probe selection moves to the dev set.
- **Privileged-teacher trap.** T3's values can assume follow-up actions the student cannot copy. Prefer T4, and accept any new teacher only on the student's return.