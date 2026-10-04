You are a senior ML research engineer consulting on an automated architecture-search system. Diagnose why the search gets stuck and prescribe fixes. Think deeply; verify claims against the code and artefacts in this repository (read-only tools available; cwd = project root).

## System (read these files as needed)
- Spec: specs/optimizer/master_spec.md (the constraints below are NON-NEGOTIABLE)
- Plan: specs/optimizer/implementation_plan.md
- Harness: factory/run_factory.py (synchronous parallel waves of K=4), factory/done_checker.py (Meta-Done rules), factory/router.py
- Simulator: factory/simulator/env.py (environment, teacher `oracle_action` = rollout lookahead over imagined futures, `clairvoyant_action` = search over the true future used only as the 100 anchor), factory/simulator/dataset.py (cached imitation data, 252 episodes, behaviour mix, labels = teacher argmax), factory/simulator/sim_runner.py (supervised imitation training with deterministic compute caps, closed-loop eval on 50 scenarios, sim_score anchors NOOP=0, heuristic=50, clairvoyant=100)
- Agent prompts: factory/prompts/*.md (Meta sees state_space.json incl. ledger, best_config.json, last_judge.json)
- Contract given to the Worker: ai_docs/simulator_contract.md, ai_docs/mlp_concepts.md, ai_docs/feature_catalog.md
- Last run artefacts: workspace/state_space.json, workspace/factory_state.json, workspace/research_paper.md, experiments/exp_000..exp_015/ (config.json, model.py, simulation.log, judge.json, meta.json in exp_000/004/008/012)

## Non-negotiable constraints
1. Python harness stays deterministic and owns the loop, stopping, routing. LLMs never decide control flow.
2. The state space must be generated dynamically by the Meta-Agent (only one seed dimension in a template; no dimension names hardcoded in Python).
3. Meta-Done rules are fixed by the spec: halt if no new dimension added in 10 iterations, or judge score did not improve by > 2.0 over the last 10 iterations, or max iterations.
4. Router rules fixed by spec (sonnet baseline, micro-retry high effort, <60 judge -> opus, meta 2 fails -> opus high, fable only for doc formatting).
5. Simulator must remain pure deterministic Python with a 60 s timeout per experiment; simulation.log has exactly 50 turns.
6. Agents stay isolated (no past experiment logs for Worker/Judge/Meta beyond state_space/best_config/last_judge).
Everything else (simulator training protocol and what it exposes to model.py, teacher quality, Meta prompt content and the Python-computed analytics it receives, judge rubric/anchor tolerance, wave width, contract docs) MAY be changed.

## Observed run (halted after 16 iterations / 4 waves on the plateau rule: best of last 10 = 76, best before = 76)
## Run digest (parallel run, 4-wide waves)

Baselines (mean return over 50 eval scenarios): {'noop_return': -6.32, 'heuristic_return': 25.939, 'oracle_return': 32.344, 'clairvoyant_return': 37.894}
Teacher (imitation oracle) expressed as sim_score = 76.79

| it | wave | judge | sim | return | succ | ovf | val_acc | train_acc | feats | params | epochs | agree | entropy | actions N/C/P/I/X/M | weakest family (ret vs heur) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 68 | 71.38 | 31.052 | 0.96 | 0.0 | 0.509 | 0.5478 | 45 | 7494 | 40/40 | 0.468 | 0.7911 | 873/150/433/384/5/155 | long_horizon_recall +34.1 vs +36.2 (teacher +36.3) |
| 1 | 0 | 70 | 73.07 | 31.455 | 0.96 | 0.0 | 0.5257 | 0.6859 | 24 | 11846 | 60/60 | 0.5075 | 0.7207 | 1016/145/330/437/4/68 | long_horizon_recall +32.5 vs +36.2 (teacher +36.3) |
| 2 | 0 | 76 | 77.74 | 32.572 | 0.98 | 0.0 | 0.5681 | 0.637 | 45 | 2726 | 100/100 | 0.481 | 0.6253 | 1182/150/219/431/0/18 | long_horizon_recall +34.1 vs +36.2 (teacher +36.3) |
| 3 | 0 | 72 | 71.25 | 31.02 | 0.98 | 0.0 | 0.4819 | 0.4991 | 39 | 2950 | 80/80 | 0.5005 | 0.8248 | 824/155/388/404/13/216 | long_horizon_recall +33.2 vs +36.2 (teacher +36.3) |
| 4 | 1 | 75 | 77.6 | 32.539 | 0.96 | 0.0 | 0.5722 | 0.6216 | 45 | 2726 | 100/100 | 0.492 | 0.5908 | 1238/137/208/411/0/6 | long_horizon_recall +33.4 vs +36.2 (teacher +36.3) |
| 5 | 1 | 70 | 72.76 | 31.382 | 0.96 | 0.0 | 0.5299 | 0.6123 | 39 | 4566 | 90/90 | 0.5175 | 0.7414 | 969/161/323/466/6/75 | long_horizon_recall +33.8 vs +36.2 (teacher +36.3) |
| 6 | 1 | 70 | 71.3 | 31.031 | 0.98 | 0.0 | 0.4674 | 0.4892 | 45 | 7494 | 70/70 | 0.4605 | 0.7752 | 875/187/391/448/1/98 | long_horizon_recall +32.9 vs +36.2 (teacher +36.3) |
| 7 | 1 | 72 | 76.95 | 32.382 | 0.96 | 0.0 | 0.5799 | 0.6869 | 45 | 9366 | 110/110 | 0.496 | 0.6162 | 1211/131/240/400/2/16 | long_horizon_recall +32.8 vs +36.2 (teacher +36.3) |
| 8 | 2 | 76 | 79.52 | 32.997 | 0.96 | 0.0 | 0.5625 | 0.6395 | 45 | 2726 | 100/100 | 0.4895 | 0.6091 | 1230/139/207/404/2/18 | long_horizon_recall +35.2 vs +36.2 (teacher +36.3) |
| 9 | 2 | 70 | 73.17 | 31.48 | 0.98 | 0.0 | 0.5229 | 0.6166 | 39 | 4566 | 90/90 | 0.5055 | 0.7357 | 1008/128/328/433/10/93 | long_horizon_recall +33.8 vs +36.2 (teacher +36.3) |
| 10 | 2 | 72 | 75.46 | 32.027 | 0.96 | 0.0 | 0.566 | 0.6005 | 37 | 2822 | 60/60 | 0.474 | 0.5943 | 1209/134/186/465/4/2 | long_horizon_recall +32.5 vs +36.2 (teacher +36.3) |
| 11 | 2 | 70 | 68.78 | 30.43 | 0.96 | 0.0 | 0.459 | 0.5576 | 39 | 8790 | 80/80 | 0.4545 | 0.8257 | 840/159/390/358/13/240 | long_horizon_recall +34.4 vs +36.2 (teacher +36.3) |
| 12 | 3 | 74 | 77.94 | 32.619 | 0.96 | 0.02 | 0.5583 | 0.6409 | 45 | 2726 | 100/100 | 0.486 | 0.6021 | 1244/150/182/404/1/19 | long_horizon_recall +34.4 vs +36.2 (teacher +36.3) |
| 13 | 3 | 69 | 72.06 | 31.213 | 0.96 | 0.0 | 0.4667 | 0.5699 | 45 | 4854 | 90/90 | 0.497 | 0.7838 | 890/187/388/399/0/136 | long_horizon_recall +34.0 vs +36.2 (teacher +36.3) |
| 14 | 3 | 75 | 79.4 | 32.968 | 0.96 | 0.0 | 0.5694 | 0.6453 | 45 | 2726 | 100/100 | 0.486 | 0.5957 | 1244/139/192/411/1/13 | long_horizon_recall +33.6 vs +36.2 (teacher +36.3) |
| 15 | 3 | 70 | 74.6 | 31.822 | 0.96 | 0.0 | 0.5667 | 0.6514 | 45 | 7494 | 70/70 | 0.509 | 0.6767 | 1061/135/323/453/2/26 | long_horizon_recall +33.3 vs +36.2 (teacher +36.3) |

## Per-family mean return, best iteration (8) vs baselines
    TRAINING | samples=8640 | val_samples=1440 | features=45 | params=2726 | optimizer=sgd | lr=0.01 | batch=256 | epochs_run=100/100 | class_weighting=none | custom_loss=0 | train_acc=0.6395 | val_acc=0.5625 | final_loss=0.94114 | featurize_s=0.62 | train_s=3.66
    FAMILY | steady_dialogue | n=8 | return=+39.94 | heuristic=+37.22 | oracle=+39.22 | clairvoyant=+42.05 | success=1.00 | overflow_rate=0.00
    FAMILY | tool_flood | n=7 | return=+32.44 | heuristic=+25.06 | oracle=+29.54 | clairvoyant=+37.43 | success=1.00 | overflow_rate=0.00
    FAMILY | debug_loop | n=7 | return=+23.97 | heuristic=+18.59 | oracle=+25.31 | clairvoyant=+31.86 | success=0.71 | overflow_rate=0.00
    FAMILY | long_horizon_recall | n=7 | return=+35.19 | heuristic=+36.17 | oracle=+36.28 | clairvoyant=+41.71 | success=1.00 | overflow_rate=0.00
    FAMILY | topic_hopping | n=7 | return=+29.63 | heuristic=+16.41 | oracle=+27.92 | clairvoyant=+34.61 | success=1.00 | overflow_rate=0.00
    FAMILY | instruction_drift | n=7 | return=+34.25 | heuristic=+22.61 | oracle=+32.86 | clairvoyant=+38.34 | success=1.00 | overflow_rate=0.00
    FAMILY | mixed_chaos | n=7 | return=+34.56 | heuristic=+23.90 | oracle=+34.31 | clairvoyant=+38.67 | success=1.00 | overflow_rate=0.00
    SUMMARY | sim_score=79.52 | mean_return=+32.997 | noop_return=-6.320 | heuristic_return=+25.939 | oracle_return=+32.344 | clairvoyant_return=+37.894 | success_rate=0.960 | overflow_rate=0.000 | mean_health=0.831 | oracle_agreement=0.489 | action_entropy=0.609 | actions=N=1230 C=139 P=207 I=404 X=2 M=18

## Final state space: active dimensions
- `hidden_layers` (categorical, added it -1): [[64, 64], [128, 64], [32, 32], [64], [48, 48], [96, 48]] — Widths of the MLP hidden layers, input to output. [64, 64] means two fully connected hidden layers of 64 units each, followed by a linear output layer over the 
- `learning_rate` (float, added it 0): [0.0001, 0.03] — Learning rate passed to the chosen torch optimizer's lr argument in the training loop.
- `optimizer` (categorical, added it 0): ["adam", "adamw", "sgd", "rmsprop"] — torch.optim class to instantiate: adam=Adam, adamw=AdamW, sgd=SGD(momentum=0.9), rmsprop=RMSprop.
- `epochs` (integer, added it 0): [10, 120] — Number of full passes over the training set, set as epochs in the training loop.
- `class_weighting` (categorical, added it 0): ["none", "balanced", "sqrt_balanced"] — Loss weighting: none=plain CrossEntropyLoss, balanced=weight classes by 1/frequency, sqrt_balanced=weight by 1/sqrt(frequency), passed as weight= to CrossEntrop
- `weight_decay` (float, added it 4): [0, 0.01] — L2 weight decay passed to optimizer's weight_decay argument (works for adam/adamw/sgd/rmsprop in the training loop).
- `dropout` (float, added it 4): [0, 0.4] — Dropout probability applied via nn.Dropout(p) after each hidden-layer activation in the MLP; 0.0 means no dropout layers are inserted.
- `standardize` (categorical, added it 4): [true, false] — If true, compute per-feature mean/std on the training set and apply (x-mean)/std normalization to inputs at train and inference time; if false, feed raw enginee
- `loss_type` (categorical, added it 4): ["cross_entropy", "focal"] — cross_entropy = standard nn.CrossEntropyLoss (optionally class-weighted per class_weighting); focal = implement focal loss (gamma=2.0) manually in model.py, dow
- `rare_action_logit_bias` (float, added it 8): [0, 3] — Additive scalar bias applied only to the CHECKPOINT_RESET and RETRIEVE_MEMORY output logits right before argmax/softmax in the forward pass (e.g. logits[:, 4] +
- `checkpoint_reset_extra_bias` (float, added it 12): [0, 2.5] — Additional scalar bias added only to the CHECKPOINT_RESET logit (index 4) in forward(), applied after rare_action_logit_bias, e.g. logits[:,4] += checkpoint_res
- `feature_window` (integer, added it 12): [1, 10] — Number of trailing per-step observations (incl. current) folded into the feature vector via deltas/rolling means of utilization, tokens_used, error_streak, etc.

Pruned: []

## Ledger coordinates (iteration: judge score: coordinate)
- 0: 68: {"class_weighting": "balanced", "epochs": 40, "feature_window": 3, "hidden_layers": [64, 64], "learning_rate": 0.001, "optimizer": "adamw"}
- 1: 70: {"class_weighting": "sqrt_balanced", "epochs": 60, "feature_window": 1, "hidden_layers": [128, 64], "learning_rate": 0.003, "optimizer": "adam"}
- 2: 76: {"class_weighting": "none", "epochs": 100, "feature_window": 5, "hidden_layers": [32, 32], "learning_rate": 0.01, "optimizer": "sgd"}
- 3: 72: {"class_weighting": "balanced", "epochs": 80, "feature_window": 6, "hidden_layers": [64], "learning_rate": 0.0005, "optimizer": "adamw"}
- 4: 75: {"class_weighting": "none", "dropout": 0.1, "epochs": 100, "feature_window": 5, "hidden_layers": [32, 32], "learning_rate": 0.01, "loss_type": "cross_entropy", "optimizer": "sgd", "standardize": true, "weight_decay": 0.0002}
- 5: 70: {"class_weighting": "sqrt_balanced", "dropout": 0.15, "epochs": 90, "feature_window": 5, "hidden_layers": [48, 48], "learning_rate": 0.003, "loss_type": "focal", "optimizer": "adamw", "standardize": true, "weight_decay": 0.0005}
- 6: 70: {"class_weighting": "balanced", "dropout": 0.2, "epochs": 70, "feature_window": 4, "hidden_layers": [64, 64], "learning_rate": 0.002, "loss_type": "cross_entropy", "optimizer": "adam", "standardize": false, "weight_decay": 0.001}
- 7: 72: {"class_weighting": "none", "dropout": 0.05, "epochs": 110, "feature_window": 6, "hidden_layers": [96, 48], "learning_rate": 0.015, "loss_type": "focal", "optimizer": "sgd", "standardize": true, "weight_decay": 0}
- 8: 76: {"class_weighting": "none", "dropout": 0, "epochs": 100, "feature_window": 6, "hidden_layers": [32, 32], "learning_rate": 0.01, "loss_type": "cross_entropy", "optimizer": "sgd", "rare_action_logit_bias": 0.7, "standardize": true, "weight_decay": 0}
- 9: 70: {"class_weighting": "sqrt_balanced", "dropout": 0.15, "epochs": 90, "feature_window": 5, "hidden_layers": [48, 48], "learning_rate": 0.003, "loss_type": "focal", "optimizer": "adamw", "rare_action_logit_bias": 1.5, "standardize": true, "weight_decay": 0.0005}
- 10: 72: {"class_weighting": "none", "dropout": 0, "epochs": 60, "feature_window": 3, "hidden_layers": [64], "learning_rate": 0.005, "loss_type": "cross_entropy", "optimizer": "adam", "rare_action_logit_bias": 0, "standardize": false, "weight_decay": 0}
- 11: 70: {"class_weighting": "balanced", "dropout": 0.15, "epochs": 80, "feature_window": 4, "hidden_layers": [96, 48], "learning_rate": 0.004, "loss_type": "cross_entropy", "optimizer": "rmsprop", "rare_action_logit_bias": 2, "standardize": true, "weight_decay": 0.0005}
- 12: 74: {"checkpoint_reset_extra_bias": 1.5, "class_weighting": "none", "dropout": 0, "epochs": 100, "feature_window": 6, "hidden_layers": [32, 32], "learning_rate": 0.01, "loss_type": "cross_entropy", "optimizer": "sgd", "rare_action_logit_bias": 0.7, "standardize": true, "weight_decay": 0}
- 13: 69: {"checkpoint_reset_extra_bias": 1.8, "class_weighting": "balanced", "dropout": 0.1, "epochs": 90, "feature_window": 5, "hidden_layers": [48, 48], "learning_rate": 0.003, "loss_type": "focal", "optimizer": "adamw", "rare_action_logit_bias": 1, "standardize": true, "weight_decay": 0.0003}
- 14: 75: {"checkpoint_reset_extra_bias": 0, "class_weighting": "none", "dropout": 0, "epochs": 100, "feature_window": 9, "hidden_layers": [32, 32], "learning_rate": 0.01, "loss_type": "cross_entropy", "optimizer": "sgd", "rare_action_logit_bias": 0.7, "standardize": true, "weight_decay": 0}
- 15: 70: {"checkpoint_reset_extra_bias": 1, "class_weighting": "sqrt_balanced", "dropout": 0.1, "epochs": 70, "feature_window": 4, "hidden_layers": [64, 64], "learning_rate": 0.002, "loss_type": "cross_entropy", "optimizer": "adam", "rare_action_logit_bias": 1.2, "standardize": true, "weight_decay": 0.0008}

## Mutation log
- it 0: add_values hidden_layers [[128, 64], [32, 32], [64]]
- it 0: add_dimension learning_rate [0.0001, 0.03]
- it 0: add_dimension optimizer ["adam", "adamw", "sgd", "rmsprop"]
- it 0: add_dimension epochs [10, 120]
- it 0: add_dimension class_weighting ["none", "balanced", "sqrt_balanced"]
- it 0: add_dimension feature_window [1, 6]
- it 4: add_dimension weight_decay [0, 0.01]
- it 4: add_dimension dropout [0, 0.4]
- it 4: add_dimension standardize [true, false]
- it 4: add_dimension loss_type ["cross_entropy", "focal"]
- it 4: add_values hidden_layers [[48, 48], [96, 48]]
- it 8: add_dimension rare_action_logit_bias [0, 3]
- it 12: add_dimension checkpoint_reset_extra_bias [0, 2.5]
- it 12: prune_dimension feature_window "long_horizon_recall has underperformed the heuristic across feature_window values 3-6; widening the
- it 12: add_dimension feature_window [1, 10]

## My preliminary hypotheses (verify or refute each with evidence; do not assume they are right)
H1. Imitation ceiling: the best students (sim 79.5) already exceed the teacher's own closed-loop score (76.8). Supervised imitation of teacher argmax labels cannot go much further; every dimension the Meta explored (optimizer, lr, epochs, dropout, class weighting, loss type, logit biases, windows) only changes how well the student fits a capped teacher.
H2. Label noise: the teacher's argmax over near-tied action values makes labels ~45% irreducibly noisy (val_acc tops at ~0.58), so hard-label cross-entropy wastes signal; per-action values (soft targets / cost-sensitive targets) are discarded.
H3. The simulator contract fixes the training procedure (offline imitation on a fixed behaviour dataset); on-policy correction (DAgger), return-based fine-tuning, or value-based targets are not expressible, so the Meta cannot discover them however it mutates the space.
H4. Harness dynamics: with K=4 waves, the 10-iteration plateau window spans only ~2.5 Meta decision points; the judge rounds and deflates sim_score (e.g. sim 79.52 -> judge 76), hiding real gains within the 2.0 threshold; 50 eval scenarios may make differences of ~1-2 points noise-level.
H5. The Meta lacks analytic feedback (e.g. per-dimension marginal effects, per-family gaps vs teacher/clairvoyant), so it cannot target the real bottleneck (long_horizon_recall below heuristic; CHECKPOINT_RESET almost never used).

## Deliverable (Markdown, <= 2500 words)
1. Root-cause diagnosis: ranked causes of the plateau with concrete evidence (numbers/files/lines).
2. For each hypothesis H1-H5: verdict (confirmed / partly / refuted) + evidence.
3. What the state space still needs to discover: concrete new dimension FAMILIES that would plausibly move sim_score materially, and for each, what the simulator/contract must expose so a Worker can realise it (e.g. soft-target/Q-value access, DAgger rounds, return-based fine-tuning via deterministic evolution strategies within the time budget, recurrent/history encoders, teacher selection). Estimate the achievable ceiling vs the clairvoyant bound.
4. Prioritised fix list (P0/P1/P2) with file-level changes, each compatible with the non-negotiable constraints, plus how to validate each fix cheaply before a full run (e.g. a hand-written model.py probe).
5. Risks (determinism, 60 s budget, isolation, gaming the judge) and mitigations.
Be specific and quantitative. Do not write code files; this is advice only.
