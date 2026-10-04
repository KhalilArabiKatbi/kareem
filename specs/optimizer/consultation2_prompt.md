You are a senior ML research engineer. Second consultation on an automated architecture-search factory. Your first report is in specs/optimizer/consultation_opus55_xhigh.md; its P0/P1 fixes were implemented (see specs/optimizer/implementation_plan.md section 9). The new run (v4.2) improved the best sim_score from 75.94 to 80.75 on the same 200-scenario evaluation, then halted again on the plateau rule. Diagnose the NEW plateau and prescribe how to break it. Verify against the code and artefacts (read-only tools; cwd = project root). Think deeply and quantitatively.

## Key files
- Harness: factory/run_factory.py (synchronous waves, K=2), factory/done_checker.py, factory/analytics.py, factory/router.py
- Simulator: factory/simulator/env.py (TEACHERS v1/v2, oracle_values, clairvoyant_action, hidden_rollout_policy, step dynamics), factory/simulator/dataset.py (cache v2: per-state Q-vectors for v1 and v2, 252 behaviour episodes, 200 eval scenarios), factory/simulator/sim_runner.py (targets hard/soft/regret, loss_fn(logits, targets, q), deterministic budgets, CEM FINETUNE with 21-scenario holdout gate, lockstep eval 50 turns x 4 seeds)
- Prompts: factory/prompts/meta.user.md, judge.user.md, worker.user.md; contract ai_docs/*.md
- v4.2 artefacts: workspace/state_space.json (with analytics), workspace/factory_state.json, workspace/research_paper.md, experiments/exp_000..exp_011/ (model.py, config.json, simulation.log, sim_metrics.json, judge.json; meta.json in the wave leads exp_000/002/004/006/008/010)

## Non-negotiable constraints (unchanged)
1. Deterministic Python harness owns loop/stopping/routing; LLMs never decide control flow.
2. State space generated dynamically by the Meta-Agent (one seed dimension; no dimension names hardcoded in Python).
3. Meta-Done rules fixed: halt if no new dimension in 10 iterations, or judge score not improved by > 2.0 over the last 10 iterations, or max iterations.
4. Router rules fixed (sonnet baseline, micro-retry high effort, judge<60 -> opus, meta 2 fails -> opus high, fable only for doc formatting).
5. Simulator pure deterministic Python, 60 s hard timeout per experiment, simulation.log exactly 50 TURN lines (each turn may aggregate seeds, already approved).
6. Agent isolation (Meta sees only state_space incl. analytics, best_config, last_judge; Worker/Judge no past logs).
Everything else may change: simulator protocol and what it exposes, teachers, targets, fine-tuning, eval design within 50 turns, Meta/Worker/Judge prompts, analytics, wave width, contract docs.

## v4.2 run digest (halted after 12 iterations on the plateau rule: best of last 10 = 80, best before = 79)

Baselines on 200 eval scenarios (mean return): {'noop_return': -7.25, 'heuristic_return': 26.195, 'oracle_return': 31.734, 'teacher_v2_return': 32.135, 'clairvoyant_return': 37.524}
As sim_score: teacher v1 = 74.45, teacher v2 = 76.22; 1 return point = 4.41 sim points

| it | judge | sim | return | teacher | target | tau | feats | params | val_acc | val_regret | noop_share | finetune (accepted, holdout before->after) | rt s | weakest family vs teacher_v2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 77 | 77.63 | 32.455 | v1 | hard | None | 33 | 6726 | 0.5757 | 0.168 | 0.6326 | - | 9.6 | long_horizon_recall -3.13 |
| 1 | 79 | 80.05 | 33.004 | v2 | soft | 1.0 | 33 | 6726 | 0.5951 | 0.0951 | 0.6981 | - | 10.39 | long_horizon_recall -1.70 |
| 2 | 79 | 80.12 | 33.02 | v2 | soft | 1.0 | 161 | 13574 | 0.5854 | 0.0962 | 0.6799 | - | 50.6 | long_horizon_recall -1.83 |
| 3 | 78 | 79.69 | 32.922 | v2 | soft | 1.0 | 185 | 13862 | 0.5896 | 0.0913 | 0.6826 | last_layer False 32.719->31.826 | 35.78 | long_horizon_recall -1.69 |
| 4 | 78 | 79.05 | 32.778 | v2 | soft | 1.0 | 145 | 13382 | 0.5764 | 0.0985 | 0.6896 | output_bias False 33.107->32.176 | 16.61 | long_horizon_recall -2.62 |
| 5 | 79 | 80.05 | 33.004 | v2 | soft | 1.0 | 33 | 6726 | 0.5951 | 0.0951 | 0.6981 | - | 8.01 | long_horizon_recall -1.70 |
| 6 | 79 | 80.08 | 33.009 | v2 | soft | 1.0 | 163 | 13702 | 0.5875 | 0.0982 | 0.6864 | - | 12.28 | long_horizon_recall -2.22 |
| 7 | 80 | 80.75 | 33.162 | v2 | soft | 1.0 | 35 | 6854 | 0.5931 | 0.0908 | 0.7006 | last_layer False 32.764->32.764 | 11.18 | long_horizon_recall -1.04 |
| 8 | 80 | 80.75 | 33.162 | v2 | soft | 1.0 | 35 | 6854 | 0.5931 | 0.0908 | 0.7006 | last_layer False 32.764->32.764 | 12.23 | long_horizon_recall -1.04 |
| 9 | 80 | 80.01 | 32.994 | v2 | soft | 1.0 | 38 | 7046 | 0.5875 | 0.0949 | 0.7004 | last_layer False 32.924->32.924 | 12.23 | long_horizon_recall -1.73 |
| 10 | 80 | 80.01 | 32.994 | v2 | soft | 1.0 | 38 | 7046 | 0.5875 | 0.0949 | 0.7004 | last_layer False 32.924->32.924 | 15.64 | long_horizon_recall -1.73 |
| 11 | 42 | 44.38 | 22.436 | v2 | soft | 0.5 | 35 | 6854 | 0.5257 | 0.1576 | 0.8017 | last_layer True 18.514->23.524 | 11.08 | tool_flood -15.23 |

## Best iteration 7: per-family mean return
| family | model | heuristic | teacher_v1 | teacher_v2 | clairvoyant | success |
|---|---|---|---|---|---|---|
| steady_dialogue | +37.38 | +33.91 | +36.01 | +36.96 | +40.58 | 1.00 |
| tool_flood | +33.32 | +25.26 | +29.96 | +31.08 | +37.77 | 0.96 |
| debug_loop | +27.92 | +20.72 | +25.85 | +24.36 | +32.01 | 0.86 |
| long_horizon_recall | +35.96 | +33.87 | +35.53 | +37.00 | +40.69 | 1.00 |
| topic_hopping | +28.67 | +17.84 | +28.42 | +28.18 | +34.11 | 0.89 |
| instruction_drift | +34.90 | +23.82 | +33.07 | +33.71 | +38.99 | 1.00 |
| mixed_chaos | +33.38 | +26.85 | +32.69 | +32.97 | +38.07 | 0.96 |

## Final state space (active)
- `hidden_layers`: [[64, 64]] — Widths of the MLP hidden layers, input to output. [64, 64] means two fully connected hidden layers of 64 units each, followed by a linear output layer over the 6 actions.
- `teacher`: ["v1", "v2"] — Set TRAIN_CONFIG['teacher']. 'v1' uses the one-step lookahead teacher for labels/Q-values; 'v2' uses the stronger two-step lookahead teacher, which should produce better closed-loop labels at higher label-computation cos
- `target`: ["hard", "soft"] — Set TRAIN_CONFIG['target']. 'hard' trains cross-entropy on argmax teacher labels. 'soft' trains cross-entropy on softmax(Q/tau) over the teacher's per-action Q-values, giving smoother supervision near ties; use default t
- `learning_rate`: [0.0001, 0.01] — Optimizer learning rate passed to the chosen optimizer in model.py training loop.
- `optimizer`: ["adam", "adamw", "sgd", "rmsprop"] — Set the optimizer class used to train the MLP: 'adam', 'adamw' (adam with decoupled weight decay), 'sgd', or 'rmsprop'.
- `standardize`: [true, false] — If true, standardize each input feature to zero mean/unit variance using training-set statistics before feeding the MLP; if false, feed raw engineered features unnormalized.
- `epochs`: [5, 60] — Number of training epochs over the generated (observation, teacher-label) dataset, subject to the harness's hard cap of 10000 optimizer steps total.
- `finetune`: [false, "bias_only", "last_layer"] — Controls the post-training deterministic CEM search over closed-loop return (<=8 generations, <=12 population, <=28 scenarios). false skips it. 'bias_only' searches only the output bias vector. 'last_layer' searches the 
- `belief_features`: ["none", "threshold_aware"] — If 'threshold_aware', featurize() appends 2 extra features: progress_to_threshold = clamp((24-progress)/24,-1,1) (signed distance to the 24/40 success cliff), and error_streak_decay = error_streak * (0.5 if last_action i
- `class_weighting`: ["none", "balanced", "sqrt_balanced"] — Sets TRAIN_CONFIG class_weighting. none uses uniform per-sample loss weighting. balanced reweights the loss per class inversely proportional to that actions frequency among teacher labels. sqrt_balanced uses sqrt of inve
- `error_loop_features`: ["none", "extended"] — If extended, featurize appends 3 extra features via safe fallback 0.0: consecutive_error_run = min(error_streak,10)/10.0, steps_since_last_success = min(steps since last success true,20)/20.0, last_action_was_recovery = 
- `target_temperature`: [0.2, 2] — Sets TRAIN_CONFIG target_temperature tau, used only when target is soft, controlling softmax of Q over tau sharpness over teacher per action Q values. Lower tau sharpens toward argmax; higher tau smooths further. Prior r
Pruned: ['history_encoder']

## Analytics at halt
typical_paired_se_sim: 0.89
spearman: {'noop_share': 0.159, 'val_acc': 0.639, 'val_regret': -0.618, 'features': 0.025, 'params': 0.025, 'dim:learning_rate': -0.395, 'dim:epochs': 0.395}
paired_vs_best: [{'iteration': 0, 'delta_sim_vs_best': -3.12, 'se_sim': 0.92}, {'iteration': 1, 'delta_sim_vs_best': -0.7, 'se_sim': 0.89}, {'iteration': 2, 'delta_sim_vs_best': -0.62, 'se_sim': 0.94}, {'iteration': 3, 'delta_sim_vs_best': -1.05, 'se_sim': 1.03}, {'iteration': 4, 'delta_sim_vs_best': -1.69, 'se_sim': 1.15}, {'iteration': 5, 'delta_sim_vs_best': -0.7, 'se_sim': 0.89}, {'iteration': 6, 'delta_sim_vs_best': -0.67, 'se_sim': 0.88}, {'iteration': 8, 'delta_sim_vs_best': 0.0, 'se_sim': 0.0}, {'iteration': 9, 'delta_sim_vs_best': -0.74, 'se_sim': 0.72}, {'iteration': 10, 'delta_sim_vs_best': -0.74, 'se_sim': 0.72}, {'iteration': 11, 'delta_sim_vs_best': -47.34, 'se_sim': 3.48}]

## Ledger
- 0: judge 77 sim 77.63: {"epochs": 20, "hidden_layers": [64, 64], "learning_rate": 0.001, "optimizer": "adam", "standardize": true, "target": "hard", "teacher": "v1"}
- 1: judge 79 sim 80.05: {"epochs": 30, "hidden_layers": [64, 64], "learning_rate": 0.0005, "optimizer": "adamw", "standardize": true, "target": "soft", "teacher": "v2"}
- 2: judge 79 sim 80.12: {"epochs": 30, "finetune": false, "hidden_layers": [64, 64], "history_encoder": "gru", "learning_rate": 0.0005, "optimizer": "adamw", "standardize": true, "target": "soft", "teacher": "v2"}
- 3: judge 78 sim 79.69: {"epochs": 30, "finetune": "last_layer", "hidden_layers": [64, 64], "history_encoder": "gru", "learning_rate": 0.0005, "optimizer": "adamw", "standardize": true, "target": "soft", "teacher": "v2"}
- 4: judge 78 sim 79.05: {"epochs": 30, "finetune": "bias_only", "hidden_layers": [64, 64], "history_encoder": "gru", "learning_rate": 0.0005, "optimizer": "adamw", "standardize": true, "target": "soft", "teacher": "v2"}
- 5: judge 79 sim 80.05: {"epochs": 30, "finetune": false, "hidden_layers": [64, 64], "history_encoder": "none", "learning_rate": 0.0005, "optimizer": "adamw", "standardize": true, "target": "soft", "teacher": "v2"}
- 6: judge 79 sim 80.08: {"belief_features": "threshold_aware", "epochs": 30, "finetune": false, "hidden_layers": [64, 64], "history_encoder": "gru", "learning_rate": 0.0005, "optimizer": "adamw", "standardize": true, "target": "soft", "teacher": "v2"}
- 7: judge 80 sim 80.75: {"belief_features": "threshold_aware", "epochs": 30, "finetune": "last_layer", "hidden_layers": [64, 64], "history_encoder": "none", "learning_rate": 0.0005, "optimizer": "adamw", "standardize": true, "target": "soft", "teacher": "v2"}
- 8: judge 80 sim 80.75: {"belief_features": "threshold_aware", "class_weighting": "sqrt_balanced", "epochs": 30, "error_loop_features": "none", "finetune": "last_layer", "hidden_layers": [64, 64], "history_encoder": "none", "learning_rate": 0.0005, "optimizer": "adamw", "standardize": true, "target": "soft", "teacher": "v2"}
- 9: judge 80 sim 80.01: {"belief_features": "threshold_aware", "class_weighting": "none", "epochs": 30, "error_loop_features": "extended", "finetune": "last_layer", "hidden_layers": [64, 64], "history_encoder": "none", "learning_rate": 0.0005, "optimizer": "adamw", "standardize": true, "target": "soft", "teacher": "v2"}
- 10: judge 80 sim 80.01: {"belief_features": "threshold_aware", "class_weighting": "sqrt_balanced", "epochs": 30, "error_loop_features": "extended", "finetune": "last_layer", "hidden_layers": [64, 64], "learning_rate": 0.0005, "optimizer": "adamw", "standardize": true, "target": "soft", "target_temperature": 1, "teacher": "v2"}
- 11: judge 42 sim 44.38: {"belief_features": "threshold_aware", "class_weighting": "none", "epochs": 30, "error_loop_features": "none", "finetune": "last_layer", "hidden_layers": [64, 64], "learning_rate": 0.0005, "optimizer": "adamw", "standardize": true, "target": "soft", "target_temperature": 0.5, "teacher": "v2"}

## My observations (verify or refute; do not assume)
O1. The student (80.75) now beats BOTH teachers (v1 74.45, v2 76.22). Soft-target imitation of v2 is saturating: the remaining levers inside imitation (features, GRU history encoder, class weighting, temperature) moved sim by < 1 point (within ~0.9 paired SE).
O2. Duplicate slots: iterations 5 = 1, 8 = 7, 10 = 9 produced bit-identical metrics although their coordinates differ; the Worker realises some coordinate differences as no-ops (or the dimension has no effect). Waves waste half their slots.
O3. long_horizon_recall is the weakest family vs teacher_v2 in every iteration (-1.0 to -3.1 return).
O4. CEM fine-tuning was accepted once in 5 attempts (and only for a badly under-trained model); with a good base model the holdout never improved. Is the CEM too weak (budget, parameterisation, 21-scenario holdout noise), or is return-based fine-tuning of the last layer fundamentally low-yield here?
O5. The judge score is an integer within +-3 of sim; the plateau rule needs judge > best_before + 2. With best_before = 79 (iteration 1, set in wave 0), later iterations needed >= 82 while sim only reached 80.75.
O6. A GRU model took 50.6 s of the 60 s budget.

## Deliverable (Markdown, <= 2500 words)
1. Diagnosis of the v4.2 plateau: ranked causes with evidence (numbers, files, lines).
2. Headroom estimate: what is realistically reachable for an observation-only policy in this simulator, and what limits it (teacher quality, information/partial observability, optimisation, evaluation noise). Propose cheap offline diagnostic probes I can run WITHOUT LLMs to measure each limit (for example a privileged student given hidden-state features, stronger teachers such as deeper/wider two-step search or expert iteration with a learned rollout policy, DAgger with the best student, larger CEM budgets), each with a concrete acceptance threshold.
3. Plateau-breaking plan: prioritised changes (P0/P1/P2) with file-level specifics, compatible with the constraints and the 60 s budget. Include how the Meta can discover the new levers (new dimension families) and how to stop wasting slots on no-op coordinates (e.g. a deterministic duplicate-model detector in the harness).
4. Risks and mitigations (determinism, budget, isolation, gaming, over-fitting to the 200 eval scenarios).
Be specific. Advice only; do not write code files.
