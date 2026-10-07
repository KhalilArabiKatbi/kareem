<p align="center">
  <img src="assets/logo.png" alt="Kareem | Context surveillance" width="560" />
</p>

**A tiny trained policy that watches an autonomous agent's context.** Every step, one forward pass (under 1 ms, CPU-only) tells the harness **when to compact, prune, or re-inject instructions**. Trained by an autonomous research factory over 100 experiments and confirmed on a hidden set never used for selection.

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests](https://github.com/KhalilArabiKatbi/kareem/actions/workflows/test.yml/badge.svg)](https://github.com/KhalilArabiKatbi/kareem/actions/workflows/test.yml)
[![Sim score](https://img.shields.io/badge/sim%20score-81.4%20of%20~83%20ceiling-brightgreen)](BENCHMARKS.md)
[![CPU only](https://img.shields.io/badge/inference-CPU%20only-lightgrey)](#install)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-hook-D97757)](#claude-code-integration-shadowrecommend-mode)

</div>

## Why Kareem

Every long-running agent hits the same wall: the context fills with stale tool output, the original instructions lose salience, error streaks compound, and eventually the model overflows or quietly degrades. Most harnesses handle this with a hand-written threshold ("compact at 90%"). Kareem replaces that guess with a learned policy.

- **One forward pass, under a millisecond.** A 3-member MLP ensemble, CPU-only. Only dependency: `torch`.
- **Six typed actions.** `NOOP`, `COMPACT`, `PRUNE_TOOLS`, `REINJECT_INSTRUCTIONS`, `CHECKPOINT_RESET`, `RETRIEVE_MEMORY`.
- **Shadow-first.** Logs every decision, so you tune the threshold on your own sessions before acting on anything.
- **Auditable.** Every experiment, judge score, and selection decision is in `experiments/`.

## Measured performance (simulator)

Scores on a scale where a clairvoyant planner that sees the future = 100. The estimated ceiling for any observation-only policy is ~83.

| Policy | Eval (200 scenarios) | Hidden confirmation (200 scenarios) |
|---|---|---|
| Do nothing | 0.0 | — |
| Hand-written threshold heuristic | 50.0 | — |
| Claude-teacher distill | — | 73.8 |
| Rule-teacher distill | — | 75.7 |
| **Kareem (3-member MLP ensemble)** | **81.5** | **81.4** |

Details: [BENCHMARKS.md](BENCHMARKS.md) and the simulator contract in [ai_docs/simulator_contract.md](ai_docs/simulator_contract.md). Every experiment, judge score, and selection decision is written to `experiments/` when you rerun the factory (not committed).

## Real-session results (shadow evaluation)

The simulator number above does not transfer on its own. We replayed 39 real Claude Code sessions (about 7.3k tool steps, 1M-token window) through Kareem and compared it with an Opus judge that saw only what existed at each step, then ran an online A/B on synthetic coding tasks. Harness and method: [`eval/`](eval/); summary table: [BENCHMARKS.md](BENCHMARKS.md#real-session-evaluation).

| Check | Result |
|---|---|
| Kareem on real sessions | `NOOP` on 99.5% of steps; only `REINJECT_INSTRUCTIONS` otherwise |
| Agreement with the Opus judge (286 labels) | 96% under 20% context use, 0% above 60% (Opus wants `PRUNE_TOOLS` or `COMPACT` there) |
| A plain utilization threshold vs Opus's "intervene" label | F1 0.78 on held-out sessions |
| Online A/B, 36 runs (cost per run) | native $1.56, compact-at-70% threshold $1.28, Kareem $1.42; every arm scored 20/20 |

Read this as: Kareem is a safe, low-impact monitor today, and a hand-written threshold at about 70% beat it on cost. Its inputs are sufficient, so the fix is retraining on real-session labels. See [BENCHMARKS.md](BENCHMARKS.md#real-session-evaluation) for limits.

## Install

```bash
pip install -e .             # only dependency: torch
pip install -e ".[claude]"   # optional: Claude Agent SDK wrapper
```

## Quickstart

```python
import kareem

agent = kareem.load_policy()                  # bundled weights
tracker = kareem.SessionTracker(capacity_tokens=200_000)

for step in agent_steps:                      # your agent loop
    obs = tracker.observe(
        tokens_used=count_tokens(context),    # real token counts; units handled internally
        tool_tokens=tool_output_tokens(context),
        last_error=step_failed,               # tool/test/command failure this step
        last_success=step_succeeded,
        progress=subtasks_done,
    )
    decision = agent.decide(tracker.history, min_confidence=0.35)
    if not decision.deferred:
        apply_intervention(decision.action)   # your harness implements the six actions
        tracker.applied(decision.action)
    else:
        tracker.applied(kareem.Action.NOOP)
```

`decision` carries `action` (enum), `confidence`, and the full `probabilities` dict. Note on thresholds: confidence is a 6-way softmax trained with soft targets, so typical max-prob is 0.3–0.4 — pick `min_confidence` from a shadow-log sweep (below), not by intuition.

## The six actions

| Action | What your harness should do |
|---|---|
| `NOOP` | nothing |
| `COMPACT` | summarise the conversation, replace with the summary; clears the error streak |
| `PRUNE_TOOLS` | drop old tool outputs, keep the most recent |
| `REINJECT_INSTRUCTIONS` | re-append the system prompt / task instructions; halves the error streak |
| `CHECKPOINT_RESET` | fresh context seeded with a checkpoint summary (rarely chosen) |
| `RETRIEVE_MEMORY` | pull facts from long-term memory / notes / RAG back in |

## Claude Code integration (shadow/recommend mode)

Works today via [hooks](https://code.claude.com/docs/en/hooks) — no modification to Claude Code. Add to `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart":       [{"hooks": [{"type": "command", "timeout": 10, "command": "python -m kareem.integrations.claude_code_hook"}]}],
    "PostToolUse":        [{"matcher": "*", "hooks": [{"type": "command", "timeout": 10, "command": "python -m kareem.integrations.claude_code_hook"}]}],
    "PostToolUseFailure": [{"matcher": "*", "hooks": [{"type": "command", "timeout": 10, "command": "python -m kareem.integrations.claude_code_hook"}]}]
  }
}
```

What it does: reads real context usage from the transcript, tracks per-session state, logs every decision to `~/.kareem/shadow/`, and when the policy is confident, injects a recommendation into Claude's context (e.g. "finish this step, then /compact"). Hooks **cannot** trigger compaction themselves — this is recommend-only by design. For real actuation, use the Agent SDK wrapper below.

Overhead: the common path is pure stdlib; torch loads only once utilization crosses `KAREEM_MIN_UTIL` (default 0.25) or an error streak forms. Config via env vars: `KAREEM_DISABLED`, `KAREEM_CAPACITY_TOKENS`, `KAREEM_MIN_CONFIDENCE`, `KAREEM_MIN_UTIL`, `KAREEM_STATE_DIR`.

### Status line (see the recommendation under the chat)

Add a Claude Code [status line](https://code.claude.com/docs/en/statusline) that shows Kareem's current read of the session:

```json
{ "statusLine": { "type": "command", "command": "python -m kareem.integrations.statusline" } }
```

```
Kareem  34% ctx  OK
Kareem  71% ctx  PRUNE_TOOLS 0.52  drop old tool output
Kareem  off
```

It reads the shadow log the hook already writes (stdlib only, no torch), so it needs the hooks above. Toggle everything (hook and status line) with `kareem-toggle on|off|status`, or `KAREEM_DISABLED=1`; off is a flag file at `~/.kareem/disabled`.

After a few sessions, tune the threshold from real data:

```bash
kareem-shadow report ~/.kareem/shadow/<session>.jsonl
```

prints the action histogram, confidence distribution, and a threshold sweep ("at min_confidence 0.35 the policy would have intervened on 12% of steps").

### Full actuation via the Agent SDK

When you own the loop, the policy can act instead of recommend. `GuardedAgent` wraps the [Claude Agent SDK](https://docs.anthropic.com/en/docs/agent-sdk/python) (`pip install claude-agent-sdk`, or `pip install -e ".[claude]"`):

```python
from claude_agent_sdk import ClaudeAgentOptions
from kareem.integrations.claude_agent import GuardedAgent

async with GuardedAgent(
    options=ClaudeAgentOptions(allowed_tools=["Read", "Bash"], cwd="."),
    instructions="<original task instructions>",   # used by REINJECT_INSTRUCTIONS
    memory=fetch_relevant_notes,                   # used by RETRIEVE_MEMORY
    min_confidence=0.35,
) as agent:
    async for msg in agent.run_turn("Fix the failing test in parser.py"):
        ...
```

Actuation per action: `REINJECT_INSTRUCTIONS` prepends the original instructions to the next prompt; `RETRIEVE_MEMORY` prepends your memory source's content; `COMPACT`/`CHECKPOINT_RESET` ask the live session for a continuation summary, then start a fresh session seeded with it (the SDK can't edit a live session's history, so compaction = checkpoint + new session); `PRUNE_TOOLS` degrades to a don't-re-read-old-outputs instruction — the one action the SDK can't truly perform.

## Caveats (read before deploying)

- **Quiet on real sessions.** On 1M-window sessions of hundreds of steps it almost never recommends anything, and its `PRUNE_TOOLS` cannot be applied through the SDK (signal only). Expect `OK` on the status line most of the time until it is retrained on real-session labels.
- **Trained on synthetic telemetry.** The simulator's dynamics (salience decay, compaction loss, overflow) are a model of real agents, not measurements. Expect a sim-to-real gap — hence shadow-first.
- **Units and scale matter.** Trained at 128k capacity, 40-step sessions. Rescale `capacity`/`horizon` to comparable units if your setup differs wildly, or retrain.
- **Confidence is diffuse by design** (soft-target distillation). Use the threshold sweep, not gut feel.
- **Retraining with real data** is the production path: record real sessions in the observation schema, build labels, rerun the factory's training protocol.

## How it was made

Kareem is the artifact; the **factory** is what produced it — an autonomous pipeline of LLM agents (Meta proposes experiments → Worker writes `model.py` → deterministic simulator trains/evaluates → Judge scores → Router escalates model effort) that ran 100 experiments over 10 iterations with a hidden confirmation set to prevent selection drift.

Documentation ladder, least to most technical:

1. [specs/optimizer/factory_explained.pdf](specs/optimizer/factory_explained.pdf) — the system as a cooking-show story
2. [specs/optimizer/factory_flows.pdf](specs/optimizer/factory_flows.pdf) — plain-language state diagrams
3. [specs/optimizer/factory_pipeline.pdf](specs/optimizer/factory_pipeline.pdf) — technical walkthrough of the state machines
4. [specs/optimizer/factory_visualization.pdf](specs/optimizer/factory_visualization.pdf) — the full atlas
5. `workspace/deploy/research_paper.md` — experiment-by-experiment account (generated by a factory run, not committed)

## Repo layout

```
kareem/               the product: runtime, weights, observation adapters, shadow tooling
  weights/            intervention_policy.pt + policy_model.py + policy_card.json
  integrations/       claude_code_hook.py (shadow/recommend) + statusline.py + claude_agent.py (actuation)
eval/                 real-session replay, Opus judge, comparison, online A/B harness (data in eval/out, gitignored)
factory/              the machine that made it (agents, router, simulator, wave executor)
specs/optimizer/      master spec, benchmark protocol, docs ladder (PDFs)
ai_docs/              agent-facing contracts (simulator contract, schemas)
experiments/          run archives (gitignored — runtime output)
workspace/            factory output: reports, deploy bundle (gitignored)
```

## License

Apache-2.0. See [LICENSE](LICENSE).
