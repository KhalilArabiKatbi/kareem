# Implementation Plan — Context Health MLP Cognitive Software Factory (v4)

Source spec: `specs/optimizer/master_spec.md`.
Golden rule: **the harness is deterministic; the work is creative.** Python owns every loop,
every stop decision, every model choice, and every file write. LLMs only return JSON.

## 0. Environment decisions

| Concern | Decision |
|---|---|
| Python | `uv` venv at `.venv/` (CPython 3.11), `torch` 2.x CPU, `numpy`, `jsonschema`, `pytest` |
| LLM transport | Headless Claude Code CLI: `claude -p` (OAuth session; no API key on this machine) |
| Agent isolation | Every call runs with `--tools ""` (no file/shell tools), `--setting-sources ""` (no user hooks, plugins or CLAUDE.md), `--strict-mcp-config`, `--no-session-persistence`, cwd = empty per-call sandbox dir. The agent sees **only** the prompt that Python assembled. |
| Structured output | `--json-schema <schema>`; Python re-validates with `jsonschema` plus semantic checks. |
| Model ids | `sonnet-5` → `claude-sonnet-5`, `opus-5.5` → `claude-opus-5-5`, `fable-5.1` → `claude-fable-5-1` |
| Effort | `standard` → `--effort medium`, `high` → `--effort high` |
| Project root | `D:\Personal\context-health-factory` (spec's `/Users/you/...` placeholder mapped here) |

## 1. Directory layout

```
ai_docs/                      read-only domain docs (checksummed at preflight, verified at halt)
specs/optimizer/              this plan + master spec
factory/                      deterministic harness (ORCHESTRATOR_DIR)
  run_factory.py              CLI + FactoryOrchestrator state machine
  config.py                   paths + constants (thresholds, timeouts)
  errors.py                   MicroFail / MacroFail exception hierarchy
  router.py                   ModelRouter (escalation ladder, Fable guard)
  router_audit.py             independent table-driven re-derivation of routing (audit)
  llm_client.py               ClaudeCLIClient (+ infra retry/backoff)
  agents.py                   AgentRunner: payload whitelist -> prompt -> call -> validate
  schemas.py                  JSON schemas per agent
  isolation.py                IsolationGuard (exp-id + canary scan) and sanitizers
  state_space.py              generic mutation engine; no dimension names in code
  state_store.py              atomic JSON persistence of factory_state
  done_checker.py             Meta-Done conditions (pure functions)
  code_guard.py               AST gate for Worker code before it runs
  paper.py                    deterministic research-paper assembly
  simulator/                  deterministic simulator (no LLM)
    env.py                    ContextEnv, scenario families, oracle, heuristic, NOOP
    dataset.py                cached training set + eval baselines
    sim_runner.py             subprocess entry: load model.py, train, run 50 scenarios, log
    harness.py                spawn sim_runner with 60 s hard timeout; validate outputs
  prompts/                    prompt templates (text only; Python fills placeholders)
  templates/                  seed state space (one seed dimension), paper header
  init_workspace.py           step 4 initialisation
  tests/                      unit + integration tests (mock LLM lives here only)
workspace/                    state_space.json, research_paper.md, best_model.py, ...
experiments/exp_NNN/          per-iteration artefacts
```

## 2. Deterministic state machine (`run_factory.py`)

```
preflight()                      # venv, claude CLI, ai_docs checksum, simulator cache
loop:
  reason = done_checker.check()  # Python only: max-iter, 10-iter no new dimension, 10-iter no >2.0 gain
  if reason: halt(reason); break
  run_wave()                     # K = --parallel iterations (default 4), see section 2a
halt(): final Doc call, finalize paper, verify best_model.py, write factory_report.json
```

### 2a. Parallelism: synchronous waves (v4.1)

Research summary. Options considered for speed-up:

| Option | Speed-up | Determinism | Verdict |
|---|---|---|---|
| Sequential (v4.0) | 1x (~6 min/iteration) | full | too slow: 50 iterations ~5 h |
| Asynchronous swarm (results merged in completion order, as in async successive halving) | ~K x | **broken**: state, escalation and Meta-Done depend on thread timing | rejected |
| **Synchronous waves** (batch Bayesian optimisation / population-based search with a barrier) | ~K x on the dominant LLM + simulator time | full: merges happen in iteration order at barriers | **chosen** |
| Pipelining doc(w) with meta(w+1) | ~+15% | full, but more coupling | deferred |

Wave `w` covers iterations `[n, n+K)` (truncated by `--max-iterations`):

| Phase | Mode | Work |
|---|---|---|
| A | serial | ONE Meta call: shared mutations + exactly K distinct candidate coordinates (portfolio) |
| B | K threads | Worker -> Simulator -> Judge per member, each with its own 3-attempt loop and its own `EscalationTracker` |
| C | serial, iteration order | scores, ledger, best model, poison bookkeeping, `last_judge.json` (whole previous wave) |
| D | K threads | Doc draft (Sonnet) + Fable format per scored member, all reading the same paper snapshot |
| E | serial, iteration order | paper assembly, iteration close, Micro-Done verification |

Determinism rules for the parallel design:

- Threads only compute and write inside their own `exp_NNN/`; every shared mutation happens in the
  main thread at a barrier, in iteration order.
- The router is stateless; rule-6 escalation lives in a per-iteration `EscalationTracker`.
- Rule 3 uses the most recent judge score known when the wave starts (the last iteration of the
  previous wave, in index order).
- Rule 4 counts Meta failures across the serial Meta phase exactly as before.
- A dimension added by the wave's Meta call is attributed to the wave's first iteration.
- Meta-Done is evaluated between waves over iteration indices (unchanged windows).
- The simulator's training budget is now deterministic (<= 10,000 optimizer steps and
  <= 1e11 parameter-samples) instead of wall-clock, so concurrent load cannot change a score.
- Resume: a wave interrupted mid-flight is archived as a whole and the pre-wave snapshot restored.
- The wave's single Meta decision is replicated into every member's `router_log.json`.

`run_iteration(i)` (now the per-member pipeline) is a staged pipeline with memoised stages:

| Stage | Owner | On failure |
|---|---|---|
| 1 meta | LLM (JSON mutation ops) → Python applies ops, writes `state_space.json` + `next_coordinate.json` | MetaProposalInvalid |
| 2 worker | LLM (model.py source) → Python AST gate → writes `model.py`, `config.json` (coordinate copied verbatim from Meta) | WorkerCodeError |
| 3 simulate | Pure Python subprocess, 60 s hard kill | SimulatorTimeout / SimulatorCrash (attributed to worker; worker stage invalidated, error fed back to the *same* experiment's retry) |
| 4 judge | LLM → Python schema + anchor guard (`|score − sim_score| ≤ 15`) | AgentOutputError |
| 5 update | Pure Python: ledger, best tracking, `best_model.py` | — |
| 6 doc | LLM draft (sonnet) → Fable format pass → Python assembles paper | AgentOutputError (draft); Fable failure falls back to draft |

Two retry levels:

1. **Step level** (`AgentRunner.call`): invalid JSON / schema failure / CLI crash → retry the exact
   step once with effort bumped to `high` (rule 2). Second failure raises to the iteration level.
2. **Iteration level**: `consecutive_fails < 3`. Any `MicroFail` → `consecutive_fails += 1`,
   `router.escalate(failing_agent)`, retry from the failed stage. Three failures → coordinate
   poisoned (`state["poisoned"]` gets `{iteration, coordinate, reason}`), iteration closes.

Infra errors (rate limit / overload) are retried with a fixed backoff schedule and do not count as
micro-fails. Usage-limit exhaustion halts gracefully with resumable state.

`IsolationViolation` is a `MacroFail`: it is never caught by the micro loop, the factory aborts.

## 3. Router (rules hard-coded in `router.py`)

Ladder: `L0 sonnet-5/standard → L1 sonnet-5/high → L2 opus-5.5/standard → L3 opus-5.5/high`.

1. Baseline: L0 for meta, worker, judge, doc.
2. Micro-fail: step retry → effort `high` on the same model (L0 → sonnet-5 + high).
3. Quality: most recent judge score < 60 → worker and judge start at L2 next iteration.
4. Complexity: meta failed twice in a row (`meta_invalid_streak ≥ 2`) → meta at L3.
5. Fable: only `doc` with purpose `format` may get `fable-5.1`. Any other pairing raises
   `RoutingViolationError` (checked in router *and* again in the LLM client).
6. Iteration escalation: each iteration-level failure bumps the failing agent one ladder level.

Each decision is written to `exp_NNN/router_log.json` with rule trace, context snapshot and an
audit field produced by `router_audit.expected_route()` (a separate table-driven implementation).
`verdict: ROUTER_VERIFIED` only when every decision matches.

## 4. Isolation

- Payload whitelist per agent (Python builds payloads, agents have no tools):
  - meta: `workspace/state_space.json`, `workspace/best_config.json`, `workspace/last_judge.json`
  - worker: `next_coordinate.json`, coordinate dimension descriptions, `ai_docs/*`, current-exp retry error
  - judge: current `simulation.log`, current `config.json`
  - doc: current `config.json`, `judge.json`, simulation summary, `workspace/research_paper.md`
- Workspace copies are sanitised (experiment ids and canaries stripped).
- Every experiment gets a canary `CANARY-exp_NNN-xxxxxxxx` inside `config.json` and `simulation.log`.
- Before every call, the guard scans the prompt for `exp_\d{3}` and canaries. Anything not
  belonging to the current experiment → `IsolationViolation` → Macro abort.
- Each exp stores the exact prompts sent (`prompts/`) and `isolation_audit.json`.

## 5. State space (dynamic)

- `workspace/state_space.json` starts with **one** seed dimension (from `templates/`).
- Meta returns mutation ops: `add_dimension`, `add_values`, `prune_values`, `prune_dimension`,
  plus `next_coordinate`. Python validates (names, domains, caps, duplicates, poisoned) and applies.
- Python code contains zero dimension names. The Worker translates coordinates into code.

## 6. Simulator

- Environment: mock agent sessions with hidden context-health dynamics (utilisation, redundancy,
  staleness, instruction salience, fact integrity, error streaks, topic shifts, overflow).
- 6 intervention actions: NOOP, COMPACT, PRUNE_TOOLS, REINJECT_INSTRUCTIONS, CHECKPOINT_RESET,
  RETRIEVE_MEMORY.
- 7 scenario families; exogenous events pre-generated per scenario (common random numbers).
- Oracle = rollout-policy lookahead on hidden state; used to label training data.
- Training data + NOOP/heuristic/oracle baselines are cached in `factory/.cache/` at preflight.
- `model.py` contract: `FEATURE_NAMES`, `featurize(history)`, `build_model(input_dim, n_actions)`,
  optional `TRAIN_CONFIG`, optional `loss_fn`. Documented in `ai_docs/simulator_contract.md`.
- Output: `simulation.log` with header, exactly 50 `TURN` lines, `SUMMARY` and `FAMILY` lines;
  `sim_metrics.json`. Deterministic seeds; 60 s subprocess timeout.

## 7. Meta-Done (Python only)

1. No `add_dimension` applied in the last 10 iterations.
2. Best judge score within the last 10 iterations does not beat the best before them by > 2.0.
3. `--max-iterations` reached.
Checked before every iteration. LLM outputs have no field that can stop the loop.

## 8. Verification

- `pytest factory/tests`: router rules 1–6, Fable violation, audit match, done conditions,
  isolation guard trips on foreign exp ids/canaries, state-space validation, poisoning after 3
  fails, full mock-LLM run through the orchestrator (end-to-end, deterministic).
- Simulator calibration: NOOP ≪ heuristic < oracle; runtime < 60 s.
- Parallel tests: waves are reproducible run-to-run, one member's failures and escalations stay
  inside that member, rule 3 across waves, resume of an interrupted wave.
- Live run: `python factory/run_factory.py --max-iterations 50` (default `--parallel 4`).
- `factory/verify_run.py` checks every Micro-Done bullet per experiment and Macro-Done at the end.

## 9. v4.2: plateau fixes (from the Opus 5.5 xhigh consultation)

Diagnosis: `specs/optimizer/consultation_opus55_xhigh.md`. The v4.1 run halted at 16 iterations
because (1) imitation of hard argmax labels from one capped teacher was the only training
signal, (2) the judge rubric steered the Meta toward rare-action boosting (about -7 sim), and
(3) the stop test could not resolve about 2-point differences (paired SE about 1.1 sim).

| Priority | Change | Files |
|---|---|---|
| P0-1 | Measured teacher headroom: two-step teacher (v2) +1.0 return on 50 scenarios, +0.4 on 200 | `simulator/env.py` (`TEACHERS`, `oracle_values`) |
| P0-2 | Judge = round(sim_score) +- 3; no entropy/rare-action/accuracy-gap penalties; hypothesis withheld | `prompts/judge.user.md`, `config.JUDGE_ANCHOR_TOLERANCE=3`, `run_factory.stage_judge` |
| P0-3 | Contract corrected (anchors, action costs, error-streak effects, 24/40 cliff); concepts rewritten around closed-loop return | `ai_docs/*` |
| P0-4 | Python analytics for the Meta (rows, marginals, Spearman, paired SE vs best, stopping-rule status, re-added dims) | `factory/analytics.py`, `state_space.json["analytics"]` |
| P0-5 | Teacher-agreement probe every 8th step; lockstep batched evaluation | `simulator/sim_runner.py` |
| P1-1 | Cache v2: per-action values for v1 and v2; `TRAIN_CONFIG` `teacher`/`target`/`target_temperature`; `loss_fn(logits, targets, q)`; `val_regret` | `simulator/dataset.py`, `sim_runner.py` |
| P1-2 | `FINETUNE`: deterministic CEM on return (output bias or last layer) with a 21-scenario held-out gate | `sim_runner.finetune` |
| P1-3 | Meta prompt: sim_score objective, protocol levers, noise-aware strategy rules | `prompts/meta.user.md` |
| P1-4 | Wave width 4 -> 2 (about 5 Meta decisions per 10-iteration window) | `config.PARALLEL_WIDTH` |
| P1-5 | Belief-state features, raw history windows, reference featurizer; `MAX_FEATURES` 1024 | `ai_docs/feature_catalog.md`, `sim_runner` |
| P2 (approved) | Each of the 50 TURN lines averages 4 seeds of one family (200 scenarios) | `env.eval_turns`, `sim_runner` |

Probe results on the v4.1 best model (no LLM, deterministic, paired over 50 turns):
teacher v2 + soft targets (tau 0.5) +0.85 +- 0.21 return (about +3.7 sim); teacher v2 hard labels
+0.45 +- 0.26; v1 soft -0.18; regret target collapses to all-NOOP; CEM fine-tuning overfits
without the held-out gate (the gate now rejects it when the holdout does not improve).

### 9a. Incident 2026-09-28 21:46: CLI auto-update during a run

The Claude Code auto-updater (`npm install -g @anthropic-ai/claude-code@2.1.284`) replaced
`claude.exe` with an install placeholder mid-run. Process launch raised `OSError [WinError 216]`,
which the stage wrappers classified as a harness micro-fail: every following wave was poisoned
within one second, the stagnation rule fired, and the final synthesis crashed on the same unhandled
`OSError` (a Macro-Fail). Fixes:

- `llm_client`: an `OSError` at process launch is an infrastructure error. The binary is
  re-resolved on every attempt and the fixed backoff schedule applies; after it, `InfraUnavailable`
  pauses the factory with resumable state.
- Stage wrappers map any leaking `OSError` to `InfraUnavailable` (pause), never to a micro-fail.
- Circuit breaker (`config.BREAKER_WAVES = 2`): two consecutive waves poisoned entirely by the same
  error type pause the factory instead of burning the stagnation window.
- Tests: `test_process_launch_oserror_pauses_without_poisoning`,
  `test_circuit_breaker_pauses_after_two_fully_poisoned_waves`,
  `test_llm_client_oserror_becomes_infra_unavailable`.

The affected run was archived and restarted with `--fresh`.

### 9b. Verifier false positive (2026-09-29)

The v4.2 run's Meta-Agent named a dimension `finetune`. The hard-coded-space scan flagged it because
`sim_runner.py` and `analytics.py` use the string `"finetune"` as the metrics key of the simulator's
`FINETUNE` contract hook. The name is contract vocabulary, not a dimension enumeration, so it was
added to `verify_run.CONTRACT_VOCABULARY` and the report was regenerated.

## 10. v4.3: breaking the v4.2 plateau (consultation 2 + offline probes)

Diagnosis: `specs/optimizer/consultation2_opus55_xhigh.md`. v4.2 plateaued at sim 80.75 because
the student already beat both teachers (imitation signal used up), 40% of post-wave-0 slots were
wasted (functional duplicates, a silently broken featurize), and the analytics/judge pointed at the
wrong gaps.

Offline probes (`factory/probes/`, results in `specs/optimizer/probe_results.json`; 280-scenario
dev set, seeds 70000+, disjoint from training and evaluation; 3 seeds per arm; paired):

| Probe | delta return +- SE | delta sim | Decision |
|---|---|---|---|
| seed spread (P-F) | SD 0.12 | SD 0.54 | ensembles default 3 |
| DAgger on-policy data (P-D) | +0.34 +- 0.10 | +1.5 | adopt (`data`) |
| tau 0.5 (P-G) | +0.13 +- 0.08 | +0.6 | expose |
| 5-seed ensemble (P-G) | +0.17 +- 0.13 | +0.8 | adopt (`ensemble_seeds`) |
| **DAgger + tau 0.5 + 3-ensemble** | **+0.53 +- 0.13** | **+2.3** | **passes the +0.45 gate** |
| v2 with 32 futures (P-A T1) | +0.05 | +0.2 | reject (sampling noise is not the limit) |
| plan to episode end, gamma 1 (T2) | -0.92 | -4.1 | reject |
| student-rollout improvement step (T4) | +0.24 +- 0.15 | +1.1 | reject (hurts with DAgger: -0.34) |
| advantage target, hard labels, v1 | +0.05 / -0.25 / -0.65 | | confirms soft v2 |

A first combination run showed DAgger + T4 at -2.4; that was a probe bug (label cache keyed by
episode count, so on-policy episodes reused behaviour labels), fixed by keying on episode seeds.

Changes:

| Item | Files |
|---|---|
| Cache v3: harness-owned reference student (`simulator/reference_policy.py`), 252 on-policy episodes with v1/v2 values | `simulator/dataset.py` |
| `TRAIN_CONFIG` `data` (behaviour / behaviour+onpolicy), `ensemble_seeds` (1-5, default 3, shared budget); FINETUNE removed (ignored with a NOTE) | `simulator/sim_runner.py` |
| NOTE for every ignored setting; `effective_config` in metrics; constant-feature validity gate | `sim_runner.py` |
| Functional fingerprint (`--fingerprint`, outside the 60 s budget): config, standardised features, initial parameters/logits, loss | `sim_runner.py`, `simulator/harness.run_fingerprint` |
| Phase B split: B1 worker+fingerprint (parallel), B2 duplicate check in iteration order with single-coordinate Meta refill (no mutations, <= 2 per wave), B3 simulate+judge | `run_factory.py` |
| Analytics: valid runs only, medians, supported correlations, clairvoyant gap x turn weight, lever coverage, duplicates, seed spread | `analytics.py` |
| Judge: round(sim) minus 1-3 only for deterministic flags | `prompts/judge.user.md` |
| Meta: new levers with dev-set effect sizes, determinism/no-replicate rule, family-weight rule, invalid-run rule | `prompts/meta.user.md` |
| Per-phase timers in the RUNTIME line; preflight warm-up | `sim_runner.py`, `harness.warm_up` |

### 10a. v4.3 outcome and transfer check (2026-09-29)

The v4.3 run halted on the plateau rule after 14 iterations (all Micro-Done; Macro-Done 9/9 after
the audit fix below). Best: iteration 6, judge 81, sim 81.48 on the evaluation set (v4.2: 80.75).

Unbiased check with `factory/probes/dev_eval.py` (full simulator protocol, 200 development
scenarios disjoint from training and evaluation, paired):

| Comparison | delta return +- SE |
|---|---|
| v4.3 best vs v4.2 best | +0.03 +- 0.23 (+0.14 sim) |
| v4.2 best + cache-v3 on-policy data + tau 0.5 + 3-ensemble vs v4.2 best | +0.04 +- 0.19 |
| v4.2 best + 3-ensemble only vs v4.2 best | +0.12 +- 0.17 |

The eval-set gain of v4.3 was selection noise. The lab DAgger gain (+0.34) came from states visited
by the learner itself; cache v3 uses a fixed harness reference student, and that distribution does
not carry the benefit (DAgger is policy-specific). Next lever if the loop continues: per-experiment
DAgger inside the simulator (roll out the trained model on training seeds, label visited states with
teacher v2 in a small process pool, retrain), estimated +0.3 +- 0.1 return at ~20-30 s extra per
experiment.

Audit fix: the hard-coded-space scan now derives its API exemptions from `sim_runner.CONTRACT_KEYS`
and `TRAIN_LIMITS` (AST-parsed), so new TRAIN_CONFIG keys such as `data` and `ensemble_seeds` no
longer produce false positives.

## 11. v4.4 (plateau loop, cycle 2): per-experiment DAgger

Hypothesis from 10a: DAgger is policy-specific, so the on-policy states must come from the model
being trained. Implementation:

- `TRAIN_CONFIG["data"] = "behaviour+self"`, `self_episodes` 28-126 (default 84).
- `sim_runner`: round 1 trains one seed on behaviour states; `rollout_record` runs it closed-loop
  on training-side scenarios (seeds 20000+, disjoint from training, on-policy, development and
  evaluation seeds); `label_with_teacher` sends (seed, actions) to `simulator/labeler.py`, a light
  subprocess with a process pool (environment only, no torch) that replays each episode and returns
  the chosen teacher's values for every visited state in job order; the final model/ensemble is
  retrained from its initial parameters on behaviour + self states, with standardisation recomputed.
  All budgets stay counted; labelling has a 40 s hard limit. A `DAGGER` log line reports the round.
- Measured cost on the v4.2 best recipe: 22.6 s total (labelling 11.1 s on 6 workers, 3,360 states).

Gate (full simulator protocol, `factory/probes/dev_eval.py`, paired against the v4.2 best recipe):

| Variant | dev set A (seeds 70000+) | dev set B (seeds 75000+) | pooled |
|---|---|---|---|
| self-DAgger, 1 seed | +0.21 +- 0.16 | -0.00 +- 0.17 | about +0.10 |
| self-DAgger, 3 seeds | +0.20 +- 0.22 | | |
| **self-DAgger, 3 seeds, tau 0.5** | **+0.36 +- 0.19** | **+0.24 +- 0.19** | **about +0.30 +- 0.13 (+1.3 sim)** |
| 3 seeds only | +0.12 +- 0.17 | | |

Also: Meta dimension descriptions may now be 10-800 characters (the 400 limit caused a recurring
micro-fail in three consecutive runs).

### 11a. Hot-fix during the v4.4 run (2026-09-29 11:01)

`SAFETY_BRAKE_S` (30 s) was measured from the start of the training phase, which with
`behaviour+self` also contains round-1 training, rollouts and labelling (11-16 s). Iterations 4 and 5
(126 self episodes, 3-member ensemble) tripped it. The brake now bounds each supervised fit
separately; the 60 s subprocess kill still bounds the whole simulation. Scores are unaffected
(budgets are counted, not timed); only the false failure is removed. Applied mid-run: each
simulation is a fresh subprocess, so the retries of iterations 4 and 5 already used the fix.

### 11b. v4.4 outcome (2026-09-29)

Run: 12 iterations, halted on the plateau rule (best of last 10 = 81, best before = 80); Micro-Done
12/12, Macro-Done 9/9. Best: iteration 5 (sim 81.38; self-DAgger, 126 episodes, 3 seeds, tau 0.5,
33 reference features). Unbiased check against the v4.2 best (full protocol, paired):
dev set A +0.04 +- 0.20, dev set B +0.12 +- 0.13 return (pooled about +0.08, +0.35 sim):
not significant. The validated lever (+0.30 on the v4.2 best's 35-feature belief set) did not
materialise because no Worker reproduced that feature set: every v4.4 model used the 33-feature
reference featurizer or wider windows. The loop's stop rule (a cycle must gain >= 1.0 sim
unbiased) triggered. Candidate next lever: promote the strongest measured featurizer into
`ai_docs/feature_catalog.md` (harness-owned documentation, not an experiment artefact), so that
feature quality and the training-signal levers can be combined by the search.

### 11c. Grokking probe (2026-09-29; `factory/probes/grokking.py`, results `grokking_results.json`)

Question: does closed-loop return improve long after the simulator's training budget (delayed
generalisation)? Two recipes (v4.2 best, v4.4 best) x weight decay {0, 0.01, 0.1}, AdamW, up to
100k optimizer steps (3,000-12,000 epochs) on behaviour states, checkpoints on the development set.

| Recipe | wd | best early (<= 3k steps) | best late (>= 10k) | late - early | trend |
|---|---|---|---|---|---|
| v4.2 best | 0 | 32.39 @3k | 32.38 @12k | -0.01 +- 0.19 | declines to 31.89 @100k |
| v4.2 best | 0.01 | 32.57 @500 | 32.48 @12k | -0.09 +- 0.20 | declines to 31.86 |
| v4.2 best | 0.1 | 32.49 @500 | 32.43 @12k | -0.05 +- 0.20 | declines to 32.12 |
| v4.4 best | 0 | 32.86 @1k | 31.73 @100k | -1.14 +- 0.20 | declines from 1k |
| v4.4 best | 0.01 | 32.82 @500 | 31.97 @12k | -0.85 +- 0.19 | declines |
| v4.4 best | 0.1 | 32.78 @500 | 32.16 @12k | -0.62 +- 0.19 | declines |

No grokking-like behaviour: return peaks within 500-3,000 steps and then falls; validation regret
rises and the weight norm keeps growing even at wd 0.1 (the contract maximum). The late phase is
ordinary over-fitting to near-tie teacher targets. Weight decay > 0.1 (the regime where grokking is
usually reported) was not tested because the contract clamps it at 0.1; given the monotone decline
it is not a priority. By-product: the optimum is EARLY, so checkpoint selection by validation regret
(early stopping) is a plausible cheap lever (the v4.4 recipe loses up to ~0.6 return between 500
and 3,000 steps).

## 12. v4.5 (plateau loop, cycle 3): early stopping + reference recipe

- `TRAIN_CONFIG["early_stop"]` = `none` | `val_regret`: per ensemble member, the epoch checkpoint with
  the lowest validation regret (teacher-value gap of the chosen action) is kept; recorded in the
  effective configuration (so it enters the fingerprint), `best_epochs`, and the TRAINING line.

Gate (full protocol, pooled over development sets A and B, paired against the v4.2 best recipe):

| Variant | pooled delta return |
|---|---|
| v4.2 best + early stop | +0.07 |
| v4.2 best + self-DAgger + 3 seeds + tau 0.5 | **+0.30** |
| same + early stop | +0.27 |
| v4.4 best (same levers, its own batch ~1024 / epochs / 126 episodes) | about +0.03 (-0.27 vs the line above) |
| v4.4 best + early stop | about +0.16 |

Early stopping alone fails the +0.25 gate (exposed as an option, not a default). The binding
constraint is search, not levers: the strongest measured recipe was never reached by v4.4's
Workers. It is now documented as the reference recipe in `ai_docs/feature_catalog.md` (harness
documentation carried between runs; no experiment artefacts or ids), and the Meta prompt anchors
exploitation on it until the ledger beats it.

### 12a. v4.5 outcome and loop stop (2026-09-29)

Run: 12 iterations, plateau halt (best of last 10 = 82, before = 81); Micro-Done 12/12, Macro-Done
9/9. Wave 0 reproduced the documented reference recipe bit-identically (iteration 0; its development
returns equal the probe's exactly). Unbiased development-set check against the v4.2 best:

| Model | dev A | dev B | pooled |
|---|---|---|---|
| iteration 0 (reference recipe) | +0.25 +- 0.20 | +0.44 +- 0.22 | **+0.34 +- 0.15 return (about +1.5 sim)** |
| iteration 3 (factory best by judge score; + NOOP-margin calibration) | +0.08 +- 0.22 | +0.20 +- 0.16 | +0.14 (about +0.6 sim) |

The cycle produced a real gain (>= 1.0 sim), but the factory's selection (highest judge score on
the 200 evaluation scenarios, as the spec requires for `best_model.py`) picked a calibration tweak
whose lead was evaluation noise (winner's curse about 0.9 sim). `workspace/best_model_dev_confirmed.py`
holds the development-confirmed model next to the spec-mandated `best_model.py`.

The plateau loop stops after its third cycle (the announced maximum). Recommended next step if work
continues: a hidden confirmation set (consultation 2, P2) that re-scores the top-k evaluation models
after the halt and reports the confirmed winner in the paper, which removes the winner's curse from
the reported result without changing the spec's selection rule.

## 13. Hidden confirmation set (post-halt re-scoring)

Selection on the 200 evaluation scenarios has a winner's curse: the selected model's lead partly
fits evaluation noise. The spec's rule (`best_model.py` = highest judge score) is unchanged; the
harness now additionally measures how much of the result survives on scenarios nobody selected on.

- Cache v4 adds 200 confirmation scenarios (50 turns x 4, seeds 95000+, disjoint from training,
  on-policy, DAgger, development and evaluation seeds) with NOOP / heuristic / clairvoyant baselines.
- `sim_runner --confirm` retrains a model and evaluates it on the confirmation turns, writing only
  `confirm_metrics.json` / `confirm_error.txt` (scored artefacts are never touched).
- At halt, before the final synthesis: every scored, non-duplicate iteration whose sim_score is within
  2 x the typical paired SE of the selected best (always including the selected best; at most
  `CONFIRM_TOP_K` = 6) is re-scored. Output: `workspace/confirmation.json`,
  `workspace/best_model_confirmed.py` (highest confirmation return), a deterministic table in the
  paper (section 3.2), and a `hidden_confirmation` block in the final Doc agent's factory summary.
  Candidates were first chosen as top-k by judge score; integer judge ties dropped near-equal models,
  so the rule became "statistically tied with the selected best".

Applied retroactively to v4.5 (evaluation sim -> confirmation sim): iteration 3 82.28 -> 81.16,
5 81.95 -> 80.99, 6 81.89 -> 80.40, 10 81.50 -> 81.37, 2 81.39 -> 80.91, 11 81.16 -> 81.21. All six
are tied within noise (paired deltas vs the selected best -0.18 to +0.05 return, SE 0.09-0.15); the
confirmed best is iteration 10. The v4.2 best drops from 80.75 (evaluation) to 79.36 (confirmation).

Cycle-3 gain over the v4.2 best, three independent sets (return, paired):

| Model | dev A | dev B | hidden confirmation |
|---|---|---|---|
| v4.5 iteration 0 (reference recipe) | +0.25 +- 0.20 | +0.44 +- 0.22 | +0.40 +- 0.19 |
| v4.5 iteration 3 (selected best) | +0.08 +- 0.22 | +0.20 +- 0.16 | +0.43 +- 0.17 |

Both are real improvements of about +0.25 to +0.36 return (+1.1 to +1.6 sim) and are tied with each
other; the earlier reading that iteration 3's calibration was pure noise was overstated.

## 14. Information ceiling (probe P-B, `factory/probes/privileged.py`, `privileged_results.json`)

Reference recipe (v4.5 iteration 0 features/network/training), behaviour data, teacher v2 soft
tau 0.5, 3 seeds per arm, closed-loop on the 280-scenario development set (anchors on that set:
NOOP -9.91, heuristic 25.97, clairvoyant 37.32).

| Student inputs | sim | delta return vs observation-only |
|---|---|---|
| observations only (deployable) | 79.18 | - |
| + scenario identity (family one-hot, true parameters) | 78.68 | -0.11 +- 0.14 |
| + true hidden state (salience, integrity, true redundancy/stale, warm-up, health) | **81.44** | **+0.51 +- 0.12** |
| + both | 80.20 | +0.23 +- 0.13 |

Reading: even perfect knowledge of the hidden state adds only about +2.3 sim inside this training
paradigm; knowing the scenario family adds nothing (the per-step state already carries what
matters, and the extra inputs over-fit). The remaining gap to the clairvoyant bound (about 19 sim)
is mostly the unpredictable future plus teacher quality, not missing observations. With the
production levers (self-DAgger, ensembles) already worth about +1.1-1.6 sim, the practical ceiling
of this paradigm is roughly 83 sim; the searched models sit at about 81 on the hidden confirmation
set, so the remaining headroom is about 2 sim, reachable only through better hidden-state
estimation (salience, integrity, true redundancy/staleness). Going materially beyond needs a
stronger teacher or a different training paradigm (every teacher variant probed so far was null or
negative).

## 15. Final benchmark (`factory/benchmark/run_benchmark.py`; `benchmark.md`, `benchmark_results.json`)

1,890 never-seen scenarios: 1,050 in-distribution (150 per family) plus four 210-scenario stress
sets (heavy growth, fast instruction drift, buggy tools, heavy recall). Every learned generation was
retrained with 3 seeds through the exact simulator protocol (current defaults, so older recipes
also get the 3-seed ensemble) and averaged per scenario; baselines include both teachers and the
clairvoyant bound. 95% intervals over scenarios.

In-distribution (main, sim_score): heuristic 50, teacher v1 74.0, teacher v2 76.9, v4.1 74.6,
v4.2 80.8, v4.3 80.5, v4.4 80.9, v4.5 it. 0 81.0, v4.5 it. 3 81.1, v4.5 it. 10 81.2, clairvoyant 100.
Paired against the v4.2 recipe: v4.5 models +0.05 to +0.10 +- 0.09 return (not significant).
Stress sets: v4.5 models beat the v4.2 recipe by +0.26 to +0.47 return on `recall` and +0.30 to
+0.44 on `buggy` (95% intervals about +-0.25-0.29), +0.14 to +0.24 on `drift`, and tie on `heavy`;
v4.1 collapses under stress (heavy 64.9, recall 66.0). Decision latency 0.3-0.4 ms.

**Correction.** Earlier single-seed comparisons on 200-scenario sets (sections 12a and 13) put the
cycle-3 gain at +1.1 to +1.6 sim. With three training seeds and 1,050 scenarios the in-distribution
gain of v4.5 over the v4.2 recipe is about +0.2 to +0.4 sim and not significant; the measurable
benefit of the cycle-2/3 levers (self-DAgger, temperature 0.5) is robustness under distribution
shift (about +1 to +2 sim on recall-heavy and buggy sessions). The one large, robust step is
v4.1 -> v4.2 (teacher v2 + soft targets: about +6 sim in-distribution, +10 to +14 under stress).
