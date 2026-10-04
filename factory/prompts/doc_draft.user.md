# Documenting task — Iteration {{ITERATION_LABEL}}

Document the iteration below and refresh the paper's abstract and results narrative.

## Output fields

- `iteration_section`: Markdown for the new section. It must start with the exact heading line
  `### Iteration {{ITERATION_LABEL}}: <short descriptive title>` and contain the subsections
  `**Configuration.**`, `**Hypothesis and rationale.**`, `**Results.**`, `**Assessment.**`.
  Cite concrete numbers (sim_score, judge score, mean return versus heuristic and clairvoyant
  baselines, success and overflow rates, validation accuracy, action distribution, weakest
  family). 250-600 words. Do not use level-1 or level-2 headings.
- `abstract`: the refreshed abstract for the whole paper so far (150-300 words), covering the
  method, the number of iterations completed, the best score to date and the main findings.
- `results_narrative`: 1-3 paragraphs summarising the search trajectory so far (trends, best
  configuration, what has and has not helped). The harness inserts a numeric leaderboard table
  above this narrative, so do not reproduce a table.

Refer to experiments only as "Iteration N". Never write raw experiment directory names.

## This iteration's config.json

```json
{{CONFIG}}
```

## This iteration's judge.json

```json
{{JUDGE}}
```

## This iteration's simulation summary

```
{{SIM_SUMMARY}}
```

## Current paper (workspace/research_paper.md)

````markdown
{{PAPER}}
````
