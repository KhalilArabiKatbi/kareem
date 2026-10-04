# Judge task

Score this experiment from 0 to 100.

## Rubric (the harness rejects any score outside `sim_score ± 3`)

1. Start at `round(sim_score)` from the SUMMARY line. `sim_score` is piecewise linear in mean
   closed-loop return over 50 turns x 4 seeds (do-nothing policy = 0, hand-written heuristic = 50,
   clairvoyant bound = 100). It is deterministic and already measures everything that costs return.
2. Deduct 1 to 3 points ONLY for these deterministic flags, and name the flag you used:
   - a NOTE line reporting an ignored, invalid, clamped or capped setting (the experiment did not
     test what its configuration claims);
   - overflow_rate above 0.02 on the SUMMARY line.
   With no flag, the score is exactly `round(sim_score)`. Never add points.
3. Do not deduct for anything else: success below 1.00, families below a teacher, low action
   entropy, rarely used actions, validation accuracy, or a train/validation gap all either cost
   no return or are already inside sim_score.
4. `feedback` explains the score with numbers from the log. `next_steps` lists 1-6 concrete,
   testable changes that could raise closed-loop return, prioritised by expected return gain: the
   largest per-family gaps to the clairvoyant weighted by that family's share of turns, the
   training signal (teacher, target, data), val_regret, the NOOP share, and features that estimate
   hidden state. `strengths` and `weaknesses` cite specific log evidence.

## config.json (hypothesis withheld)

```json
{{CONFIG}}
```

## simulation.log

```
{{SIMULATION_LOG}}
```
