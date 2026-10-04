# Documenting task — final synthesis

The factory has halted. Write the closing parts of the paper so it reads as one cohesive study.

## Output fields

- `abstract`: final abstract (150-300 words): problem, method (deterministic harness, dynamic
  search space, escalation router, simulator), number of iterations, best result and the key
  findings.
- `results_narrative`: 2-4 paragraphs on the full search trajectory (the harness inserts the
  numeric leaderboard table above it; do not reproduce the table).
- `discussion`: Markdown (no level-1/2 headings; level-3 allowed) covering which dimensions
  mattered, failure modes observed, limitations of the simulator and of imitation of a noisy
  teacher, and threats to validity. 300-900 words.
- `conclusion`: 120-300 words; include the halting reason and concrete future work.

If the factory summary contains `hidden_confirmation`, report it in `results_narrative` and
`conclusion`: which iteration the evaluation set selected, which iteration was best on the hidden
confirmation scenarios (never used for selection), and what the difference says about selection
noise.

Refer to experiments only as "Iteration N". Report only numbers present in the inputs.

## Factory summary (Python-generated)

```json
{{FACTORY_SUMMARY}}
```

## Current paper (workspace/research_paper.md)

````markdown
{{PAPER}}
````
