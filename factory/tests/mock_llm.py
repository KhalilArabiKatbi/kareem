"""Scripted stand-in for the Claude CLI, used only by tests.

Behaviour per call is taken from a queue keyed by "<agent>/<purpose>"; "ok" when the queue is empty.
Dimension names created here exist only in tests (they prove the harness itself hardcodes none).
"""
from __future__ import annotations

import json
import re
import threading

from factory.errors import AgentCallError, InfraUnavailable
from factory.llm_client import LLMResponse
from factory.router import enforce_fable_constraint

GOOD_MODEL = '''
import torch

WIDTH = 32
import torch.nn as nn

FEATURE_NAMES = ["util", "tool_frac", "since_instr", "stale", "red", "recall_miss", "since_compact", "err_streak"]


def featurize(history):
    o = history[-1]
    return [o["utilization"], o["tool_fraction"], o["steps_since_instruction"] / 40.0, o["stale_est"],
            o["redundancy_est"], float(o["recall_miss"]), o["steps_since_compaction"] / 40.0,
            min(o["error_streak"], 6) / 6.0]


class Net(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, WIDTH), nn.ReLU(), nn.Linear(WIDTH, n))

    def forward(self, x):
        return self.body(x)


def build_model(input_dim, n_actions):
    return Net(input_dim, n_actions)


TRAIN_CONFIG = {"lr": 3e-3, "epochs": 8, "batch_size": 512, "optimizer": "adam", "ensemble_seeds": 1}
'''

WEAK_MODEL = GOOD_MODEL.replace(
    "        return self.body(x)",
    "        return self.body(x) * 0.0 + torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])",  # always NOOP
)

CRASH_MODEL = GOOD_MODEL.replace("    o = history[-1]\n", "    o = history[-1]\n    o = o['missing_key']\n", 1)

BANNED_MODEL = "import os\n" + GOOD_MODEL


class MockLLM:
    def __init__(self, plan: dict[str, list[str]] | None = None):
        self.plan = {k: list(v) for k, v in (plan or {}).items()}
        self.calls: list[dict] = []
        self.meta_count = 0
        self._lock = threading.Lock()

    def _next(self, key: str) -> str:
        q = self.plan.get(key)
        return q.pop(0) if q else "ok"

    def call(self, decision, system_prompt, user_prompt, json_schema):
        enforce_fable_constraint(decision.agent, decision.purpose, decision.model)
        key = f"{decision.agent}/{decision.purpose}"
        m = re.search(r'"experiment": "(exp_\d{3})"', user_prompt)
        with self._lock:
            # A per-experiment queue ("worker/primary@exp_001") wins, so parallel tests stay deterministic.
            exp_key = f"{key}@{m.group(1)}" if m else None
            behaviour = self._next(exp_key) if exp_key and self.plan.get(exp_key) else self._next(key)
            self.calls.append({"key": key, "model": decision.model, "effort": decision.effort,
                               "rules": list(decision.rules), "behaviour": behaviour,
                               "experiment": m.group(1) if m else None})
        if behaviour == "crash":
            raise AgentCallError(f"{decision.agent} process crashed (mock)", agent=decision.agent)
        if behaviour == "infra":
            raise InfraUnavailable("mock quota exhausted")
        if behaviour == "oserror":
            raise OSError(216, "This version of %1 is not compatible (mock binary swap)")
        if behaviour == "bad_json":
            return LLMResponse("this is {not json", None, 0.0, 0.01, [decision.model_id])
        out = getattr(self, f"_{decision.agent}_{decision.purpose}")(user_prompt, behaviour)
        return LLMResponse(json.dumps(out), None, 0.0, 0.01, [decision.model_id])

    # ---- agents ----
    @staticmethod
    def _json_block_after(prompt: str, marker: str):
        seg = prompt[prompt.index(marker):]
        m = re.search(r"```json\n(.*?)\n```", seg, re.S)
        return json.loads(m.group(1))

    def _meta_primary(self, prompt, behaviour):
        space = self._json_block_after(prompt, "## Current state space")
        width = int(re.search(r"choose exactly (\d+) coordinate", prompt).group(1))
        dims = space["dimensions"]
        with self._lock:
            self.meta_count += 1
            k = self.meta_count
        name = f"mockdim_{k:02d}"
        values = [k * 100 + j for j in range(max(2, width))]
        mutations = [{"op": "add_dimension", "name": name,
                      "spec": {"type": "categorical", "values": values,
                               "description": f"Test-only dimension number {k}; see exp_999 for nothing."}}]
        base = {d: (s["values"][0] if s["type"] == "categorical" else s["min"]) for d, s in dims.items()}
        cands = []
        refill = "Refill request (harness)" in prompt
        if refill:
            behaviour = "no_mutation"
            with self._lock:
                self.refills = getattr(self, "refills", 0) + 1
        if behaviour == "no_mutation":
            mutations = []
            used = {json.dumps(e["coordinate"], sort_keys=True) for e in space.get("ledger", [])}
            unused = []
            for dname, spec in dims.items():
                if spec["type"] != "categorical":
                    continue
                for v in spec["values"]:
                    trial = dict(base, **{dname: v})
                    key = json.dumps(trial, sort_keys=True)
                    if key not in used:
                        used.add(key)
                        unused.append(trial)
            # a refill must avoid coordinates already tried in this wave: rotate from the end of the list
            cands = [unused[-self.refills % len(unused)]] if refill else unused[:width]
        else:
            cands = [dict(base, **{name: values[j]}) for j in range(width)]
        if behaviour == "invalid_proposal":
            cands[0]["not_a_dimension"] = 1
        return {"analysis": "mock analysis", "mutations": mutations,
                "candidates": [{"next_coordinate": c, "hypothesis": f"mock hypothesis {k}.{j}"}
                               for j, c in enumerate(cands)]}

    def _worker_primary(self, prompt, behaviour):
        code = {"ok": GOOD_MODEL, "same": GOOD_MODEL, "weak": WEAK_MODEL, "crash_code": CRASH_MODEL,
                "banned": BANNED_MODEL}[behaviour]
        if behaviour != "same":  # a distinct (but equally valid) model per coordinate, like a real Worker
            h = re.search(r'"coordinate_hash": "([0-9a-f]+)"', prompt).group(1)
            code = code.replace("WIDTH = 32", f"WIDTH = {16 + int(h[:4], 16) % 32}")
        return {"model_py": code, "design_notes": "mock design notes; mentions exp_123 which must be scrubbed"}

    def _judge_primary(self, prompt, behaviour):
        sim = float(re.search(r"SUMMARY \| sim_score=([\d.]+)", prompt).group(1))
        score = int(round(sim))
        if behaviour == "far_score":
            score = 100 if sim < 50 else 0
        return {"score": score, "feedback": "mock feedback grounded in the SUMMARY line.",
                "next_steps": ["try something"], "strengths": ["s"], "weaknesses": ["w"]}

    def _doc_draft(self, prompt, behaviour):
        label = re.search(r"# Documenting task — Iteration (\d{3})", prompt).group(1)
        head = f"### Iteration {label}: Mock study" if behaviour != "bad_heading" else "## Wrong heading"
        body = ("**Configuration.** Mock configuration 12.5.\n\n**Hypothesis and rationale.** Mock.\n\n"
                "**Results.** sim_score 42.0 recorded.\n\n**Assessment.** Adequate. " + "Filler text. " * 20)
        return {"iteration_section": f"{head}\n\n{body}", "abstract": "Mock abstract. " * 20,
                "results_narrative": "Mock narrative. " * 10}

    def _doc_format(self, prompt, behaviour):
        section = re.search(r"````markdown\n(.*?)\n````", prompt, re.S).group(1)
        if behaviour == "lossy":
            section = re.sub(r"\d+\.\d+", "X", section)
        return {"formatted_markdown": section.strip() + "\n"}

    def _doc_final(self, prompt, behaviour):
        return {"abstract": "Final mock abstract. " * 15, "results_narrative": "Final narrative. " * 10,
                "discussion": "Final discussion. " * 30, "conclusion": "Final conclusion. " * 12}
