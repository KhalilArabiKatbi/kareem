# Autonomous Discovery of Context-Health Intervention Policies with a Deterministic Cognitive Software Factory

<!-- BEGIN:STATUS -->
*Status: search in progress. This document is maintained by the factory harness; sections between markers are regenerated automatically.*
<!-- END:STATUS -->

## Abstract

<!-- BEGIN:ABSTRACT -->
*Pending: the abstract is written after the first completed iteration.*
<!-- END:ABSTRACT -->

## 1. Introduction

Autonomous language-model agents that run for many steps accumulate context: dialogue, tool
output, failed retries, and material from abandoned sub-topics. Well before the hard context limit
is reached, task performance degrades as instructions lose salience, redundant and stale content
dilutes attention, and early facts are summarised away or truncated. A lightweight *intervention
policy* can monitor inexpensive telemetry and decide, at each step, whether to compact the
context, prune tool output, re-inject instructions, reset from a checkpoint, retrieve memory, or
do nothing. Because every intervention has a cost, the policy must act selectively.

This paper studies small multilayer perceptrons (MLPs) as intervention policies. The design space
(feature engineering over the telemetry history, topology, regularisation, loss shaping and
optimisation) is large and poorly understood, so it is not fixed in advance. Instead, the space is
grown and pruned during the search by a language-model Meta-Agent, while a deterministic Python
harness owns control flow, model routing, evaluation and stopping.

## 2. Methodology

### 2.1 Deterministic harness

A Python state machine executes each iteration as a fixed pipeline: Meta-Agent (search-space
mutation and coordinate selection), Worker Agent (implementation of the coordinate as PyTorch
code), deterministic simulator, Judge Agent (scoring), a pure-Python state update, and the
Documenting Agent. Language models return JSON only; they cannot read files, run tools, alter the
loop, or stop it. Each agent receives a whitelisted payload, and every prompt is scanned for
references to other experiments before it is sent. Invalid outputs are retried with escalated
effort; an iteration that fails three times has its coordinate poisoned.

To shorten wall-clock time, iterations execute in synchronous parallel *waves*: a single Meta-Agent
call proposes a portfolio of K coordinates, the K Worker-simulator-Judge pipelines and the K
documentation passes run concurrently, and all shared state is merged by Python in iteration order
at barriers. Consequently, coordinates within a wave cannot learn from each other's results, a
trade-off analogous to batch Bayesian optimisation, while the outcome remains independent of thread
timing.

### 2.2 Model and effort routing

The harness assigns models by rule: every agent starts on Sonnet 5 at standard effort; an in-place
retry after invalid output raises effort to high; a judge score below 60 moves the next
iteration's Worker and Judge to Opus 5.5; two consecutive invalid Meta proposals move the Meta-Agent
to Opus 5.5 at high effort; and Fable 5.1 is reserved for Markdown formatting of this document.
Every routing decision is logged and independently re-derived for audit.

### 2.3 Simulation environment

Each evaluation scenario is a 40-step agent session with hidden dynamics: context utilisation,
redundant and stale fractions, instruction salience, fact integrity, error loops, topic shifts and
overflow. Seven scenario families (steady dialogue, tool flooding, debugging loops, long-horizon
recall, topic hopping, instruction drift, and random mixtures) with per-scenario parameter jitter
are used. The policy observes 21 noisy telemetry fields per step and selects one of six
interventions. Rewards count successful sub-tasks minus intervention costs, recall-miss and
overflow penalties, plus a terminal bonus for task success.

### 2.4 Training and evaluation protocol

Two teachers plan on hidden state over imagined futures (never the realised one): a one-step
lookahead teacher (v1) and a two-step lookahead teacher (v2). The cache stores both teachers'
per-action values for approximately 10,000 states from 252 behaviour-policy episodes and for as
many states visited by a harness-owned reference student (on-policy, DAgger-style data). A policy
is trained on hard labels, on soft targets derived from the per-action values, or on a custom loss
that receives them, optionally on the on-policy states as well, and optionally as a seed ensemble
whose logits are averaged. All training budgets are counted in optimizer steps and
parameter-samples, so results do not depend on machine load. Before simulation, each model is
fingerprinted (effective configuration, standardised features, initialisation, loss); a model
functionally identical to an earlier one is not simulated again, and its slot returns to the
Meta-Agent. Evaluation runs the trained policy closed-loop on 50 turns of four scenarios each (200
held-out scenarios, one family per turn). The deterministic `sim_score` is piecewise linear in
mean episode return, with anchors at the do-nothing policy (0), a hand-written threshold heuristic
(50) and a clairvoyant search over the true future (100, an unreachable bound). The Judge Agent
starts from the rounded `sim_score` and may only deduct up to three points for deterministic
flags. The Meta-Agent receives Python-computed analytics (per-dimension marginals over valid runs,
rank correlations, paired standard errors, lever coverage and duplicate records) alongside the
ledger. Candidate protocol changes were selected offline on a separate 280-scenario development
set, never on the evaluation scenarios.

### 2.5 Stopping rule

The harness halts when no new search-space dimension has been added in ten iterations, when the
best judge score of the last ten iterations fails to exceed the earlier best by more than 2.0
points, or when the iteration budget is exhausted.

## 3. Results

### 3.1 Leaderboard

<!-- BEGIN:RESULTS_TABLE -->
*No completed iterations yet.*
<!-- END:RESULTS_TABLE -->

### 3.2 Hidden confirmation

<!-- BEGIN:CONFIRMATION -->
*Pending: after the search halts, the top iterations are re-scored on 200 hidden scenarios that were
never used for selection.*
<!-- END:CONFIRMATION -->

### 3.3 Analysis

<!-- BEGIN:RESULTS_NARRATIVE -->
*Pending.*
<!-- END:RESULTS_NARRATIVE -->

## 4. Iteration Log

<!-- BEGIN:ITERATIONS -->
<!-- END:ITERATIONS -->

## 5. Discussion

<!-- BEGIN:DISCUSSION -->
*Pending: written when the search halts.*
<!-- END:DISCUSSION -->

## 6. Conclusion

<!-- BEGIN:CONCLUSION -->
*Pending: written when the search halts.*
<!-- END:CONCLUSION -->

## Appendix A. Search-Space Evolution

<!-- BEGIN:APPENDIX_SPACE -->
*The seed search space contains a single dimension.*
<!-- END:APPENDIX_SPACE -->
