# Master Spec: Context Health MLP Cognitive Software Factory (v4)

## The Golden Rule
**The harness is deterministic; the work is creative.**
The LLMs (Agents) will write code, explore spaces, and write papers. They *will* drift, hallucinate, and lose context. Therefore, the Python Orchestrator (`run_factory.py`) must act as an unforgiving, deterministic state machine. The LLM is never allowed to write the `if/else` logic that controls the loop, decides when to stop, or routes the models. Python does that.

## Problem
Optimizing an intervention MLP for autonomous agent context health is an unknown, multi-dimensional search problem. Hardcoding the state space limits discovery. Using a single LLM for everything wastes compute. Letting an LLM control its own execution loop guarantees drift and infinite loops.

## Solution
Build a deterministic Python-driven "Cognitive Software Factory" featuring 5 isolated AI agents, a strict Model/Effort Router, and an ironclad Python state machine.

## Variables & Directory Structure
- `PROJECT_ROOT: /Users/you/projects/context-health-factory` (this machine: `D:\Personal\context-health-factory`)
- `AI_DOCS_DIR: <PROJECT_ROOT>/ai_docs/` (Read-only. Contains initial MLP concepts, feature lists).
- `PLAN_DIR: <PROJECT_ROOT>/specs/optimizer/`
- `ORCHESTRATOR_DIR: <PROJECT_ROOT>/factory/` (Contains the deterministic Python harness).
- `WORKING_DIR: <PROJECT_ROOT>/workspace/` (Contains `state_space.json`, `research_paper.md`, `best_model.py`).
- `EXPERIMENTS_DIR: <PROJECT_ROOT>/experiments/` (Contains `exp_000/`, `exp_001/`, etc.).

---

## 1. The Model & Effort Router (The Escalation Ladder)
The Python Harness strictly controls which model is used. The LLM cannot request a model; Python assigns it based on failure history.

**Models:** `sonnet-5` (Fast/Default), `opus-5.5` (Slow/Heavy), `fable-5.1` (Ultra-fast/Last resort).
**Efforts:** `standard` (Normal), `high` (Extended thinking).

**Routing Rules (Hardcoded in Python):**
1. **Baseline:** All core agents start at `sonnet-5` + `standard`.
2. **Micro-Fail Escalation:** If an agent returns invalid JSON or crashes, retry that exact step with `sonnet-5` + `high`.
3. **Quality Escalation:** If the Judge scores < 60, the *next* iteration's Worker and Judge are upgraded to `opus-5.5` + `standard`.
4. **Complexity Escalation:** If the Meta-Agent fails to propose a valid state space mutation twice, upgrade it to `opus-5.5` + `high`.
5. **The Fable Constraint:** `fable-5.1` may ONLY be used by the Documenting Agent for strict Markdown formatting. If Python detects `fable-5.1` routed to Meta, Worker, Judge, or Simulator, it throws a `RoutingViolationError` and fails the iteration.

---

## 2. The Agents (Strictly Isolated)

### A. Meta-Agent (The Explorer)
- **Input:** Current `state_space.json`, `best_config.json`, last `judge.json`.
- **Task:** Dynamically mutate the state space. Add new feature dimensions, invent new topologies, prune dead ends. Select the `next_coordinate`.
- **Output:** Updated `state_space.json` and `next_coordinate.json`.
- **Constraint:** Cannot see past experiment logs.

### B. Worker Agent (The Builder)
- **Input:** `next_coordinate.json`, `AI_DOCS_DIR` context.
- **Task:** Write the exact PyTorch `model.py` and `config.json` for this coordinate.
- **Constraint:** Cannot see past experiment logs.

### C. Simulator (Deterministic Python)
- **Task:** Reads `model.py`. Instantiates it. Trains it on synthetic data. Runs 50 mock context scenarios. Generates `simulation.log`.
- **Constraint:** Pure Python. No LLM involved. Must timeout after 60 seconds.

### D. Judge Agent (The Evaluator)
- **Input:** `simulation.log`, `config.json`.
- **Task:** Score the run (0-100), write `feedback`, suggest `next_steps`.
- **Constraint:** Cannot see past experiment logs.

### E. Documenting Agent (The Researcher)
- **Input:** Current iteration's `config.json`, `judge.json`, `simulation.log` summary, existing `research_paper.md`.
- **Task:** Append a new "Iteration X" section to the paper. Update Abstract/Results. Tone must be formal and academic.
- **Constraint:** Runs *after* the Judge. Uses `fable-5.1` ONLY for formatting; otherwise uses `sonnet-5`.

---

## 3. The Deterministic Harness (`factory/run_factory.py`)

This is the anchor. It prevents drift. It must be implemented exactly according to these logic rules:

```python
# Pseudo-logic for the Harness
class FactoryOrchestrator:
    def run_iteration(self, i):
        consecutive_fails = 0

        while consecutive_fails < 3:
            try:
                # 1. META-AGENT
                meta_out = self.call_agent("meta", payload)
                if meta_out.get("meta_done"): return self.halt("Meta-Done signaled")

                # 2. WORKER-AGENT
                worker_out = self.call_agent("worker", payload)

                # 3. SIMULATOR (Pure Python)
                sim_log = self.run_simulator(worker_out)

                # 4. JUDGE-AGENT
                judge_out = self.call_agent("judge", payload)

                # 5. UPDATE STATE (Pure Python)
                self.update_state(judge_out["score"])

                # 6. DOC-AGENT
                self.call_agent("doc", payload)

                return # SUCCESS: Break retry loop

            except (JSONDecodeError, SimulatorTimeout, RoutingViolation) as e:
                consecutive_fails += 1
                self.router.escalate() # Bump model/effort for next try

        # Failed 3 times. Poison the coordinate.
        self.state["poisoned"].append(i)
```

**Meta-Done Conditions (Checked by Python, NOT the LLM):**
1. `state_space` has not mutated (added new dimensions) in 10 iterations.
2. Judge score has not improved by > 2.0 points over the last 10 iterations.
3. Hard limit of `--max-iterations` reached.

---

## 4. Definition of Done

### Micro-Done (A single iteration is successful)
- `exp_XXX/config.json` maps to a coordinate generated by the Meta-Agent.
- `exp_XXX/model.py` compiles and runs in the Simulator without Python errors.
- `exp_XXX/simulation.log` contains exactly 50 simulated turns.
- `exp_XXX/judge.json` contains valid JSON with `score` (int 0-100).
- `research_paper.md` is updated with a new section detailing Iteration XXX.
- `exp_XXX/router_log.json` proves the Model Router correctly selected the models.

### Macro-Done (The Factory run is complete)
- The Python Harness hits one of the Meta-Done conditions and exits gracefully.
- `workspace/best_model.py` contains the highest-scoring architecture discovered.
- `workspace/research_paper.md` is a complete, cohesive academic paper detailing the entire search journey.

---

## 5. Definition of Fail

### Micro-Fail (Iteration failed, but Factory survives)
- Worker code throws a Python error.
- Judge returns invalid JSON.
- Simulator times out (>60s).
- Router attempts to use `fable-5.1` for a core agent.
*Recovery:* Python logs error, escalates model/effort, retries up to 3 times. If 3 fails, coordinate is poisoned.

### Macro-Fail (Catastrophic Factory Abort)
- **Instant Failure:** The Python Harness crashes due to an unhandled exception.
- **Instant Failure:** Agent isolation is broken (e.g., Worker prompt accidentally contains data from `exp_002` while running `exp_005`).
- **Instant Failure:** The state space is hardcoded in the Python script. It MUST be dynamically generated by the Meta-Agent.
- **Instant Failure:** The LLM is allowed to write the `if/else` logic for the loop or the Meta-Done checks.
- **Instant Failure:** You write project deliverables outside the designated directories.

---

## Workflow for the AI (You)

1. **Plan.** Create the implementation plan for the Orchestrator, Router, and Agent wrappers. Save in `PLAN_DIR/`.
2. **Build the Harness.** Implement the strict Python state machine in `ORCHESTRATOR_DIR`. This is the most critical step. Do not compromise on the deterministic logic.
3. **Build the Agents.** Implement the prompt templates and API wrappers for Meta, Worker, Judge, and Doc agents.
4. **Initialize.** Create an empty `state_space.json` (with just a seed dimension) and the header for `research_paper.md` in `WORKING_DIR`.
5. **Execute.** Run `python factory/run_factory.py --max-iterations 50`.

## How You're Graded
You will be graded continuously on every bullet in the Definitions of Done and Fail.
If you let the LLM control the loop, you fail.
If you hardcode the state space, you fail.
If you don't build the escalation router, you fail.
Build the ironclad Python harness first. Then build the agents.
