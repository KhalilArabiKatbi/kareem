"""Paths and constants for the deterministic harness.

Everything that controls the loop lives here as plain constants. No LLM output can change them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_DOCS_DIR = PROJECT_ROOT / "ai_docs"
PLAN_DIR = PROJECT_ROOT / "specs" / "optimizer"
ORCHESTRATOR_DIR = PROJECT_ROOT / "factory"
WORKING_DIR = PROJECT_ROOT / "workspace"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

PROMPTS_DIR = ORCHESTRATOR_DIR / "prompts"
TEMPLATES_DIR = ORCHESTRATOR_DIR / "templates"
CACHE_DIR = ORCHESTRATOR_DIR / ".cache"
SANDBOX_DIR = ORCHESTRATOR_DIR / ".sandbox"

# Workspace files (WORKING_DIR)
STATE_SPACE_PATH = WORKING_DIR / "state_space.json"
FACTORY_STATE_PATH = WORKING_DIR / "factory_state.json"
RESEARCH_PAPER_PATH = WORKING_DIR / "research_paper.md"
BEST_MODEL_PATH = WORKING_DIR / "best_model.py"
BEST_CONFIG_PATH = WORKING_DIR / "best_config.json"
LAST_JUDGE_PATH = WORKING_DIR / "last_judge.json"
FACTORY_LOG_PATH = WORKING_DIR / "factory.log"
FACTORY_REPORT_PATH = WORKING_DIR / "factory_report.json"

# Python interpreter used for the simulator subprocess (the same venv that runs the harness).
PYTHON_EXE = sys.executable

# ---- Loop control -------------------------------------------------------------------------
MAX_ITERATION_ATTEMPTS = 3          # consecutive_fails < 3, then poison the coordinate
BREAKER_WAVES = 2                   # consecutive fully-poisoned waves (same error type) that pause the factory
STEP_RETRIES = 1                    # one in-place retry per agent step (rule 2)
STAGNATION_WINDOW = 10              # Meta-Done conditions 1 and 2
MIN_IMPROVEMENT = 2.0               # Meta-Done condition 2 ("improved by > 2.0 points")
QUALITY_THRESHOLD = 60              # rule 3
META_FAIL_ESCALATION = 2            # rule 4 ("fails ... twice")

# ---- Simulator ----------------------------------------------------------------------------
SIM_TIMEOUT_S = 60                  # hard kill
SIM_EXPECTED_TURNS = 50
JUDGE_ANCHOR_TOLERANCE = 3          # |judge.score - sim_score| must stay within this (judge = sim_score +- 3)

# ---- Parallelism ----------------------------------------------------------------------------
# Synchronous waves: Meta proposes PARALLEL_WIDTH coordinates, the Worker -> Simulator -> Judge
# pipelines and the Doc drafts run concurrently, and Python merges results in iteration order at a
# barrier. The width is a harness constant (or CLI flag), never an LLM decision.
PARALLEL_WIDTH = 2                  # 2 keeps ~5 Meta decisions inside each 10-iteration window
MAX_PARALLEL_WIDTH = 8
MAX_REFILLS_PER_WAVE = 2            # functional-duplicate slots handed back to the Meta per wave
CONFIRM_TOP_K = 6                   # cap on iterations re-scored on the hidden confirmation set: every iteration
                                    # whose sim_score is within 2 x the typical paired SE of the selected best

# ---- Measured noise (probe P-F, specs/optimizer/probe_results.json) --------------------------
SEED_SPREAD_SIM = 0.54              # sim-score SD of one recipe retrained with different torch seeds

# ---- LLM transport -------------------------------------------------------------------------
CLAUDE_BIN = os.environ.get("FACTORY_CLAUDE_BIN", "claude")
LLM_CALL_TIMEOUT_S = 900
INFRA_BACKOFF_S = (30, 60, 120, 240, 480)   # fixed schedule for overload / rate-limit errors

MODEL_IDS = {
    "sonnet-5": "claude-sonnet-5",
    "opus-5.5": "claude-opus-5-5",
    "fable-5.1": "claude-fable-5-1",
}
EFFORT_FLAGS = {
    "standard": "medium",
    "high": "high",
}

CORE_AGENTS = ("meta", "worker", "judge", "doc")
ALL_ROUTABLE = ("meta", "worker", "judge", "doc", "simulator")


def exp_name(iteration: int) -> str:
    return f"exp_{iteration:03d}"


def exp_dir(iteration: int) -> Path:
    return EXPERIMENTS_DIR / exp_name(iteration)
