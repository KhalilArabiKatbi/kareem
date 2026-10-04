"""Agent isolation: payload whitelists, sanitisers, and the prompt guard.

Agents have no tools, so the only data they see is the prompt Python assembles. The guard scans
every prompt right before it is sent. A reference to any experiment other than the current one
(by id or by canary) is an IsolationViolation, which is a Macro-Fail: the factory aborts.
"""
from __future__ import annotations

import copy
import re
import secrets

from .errors import IsolationViolation

EXP_RE = re.compile(r"exp_(\d{3})")
CANARY_RE = re.compile(r"CANARY-exp_\d{3}-[0-9a-f]{8}")

# Which workspace/experiment sources each agent may receive. Enforced by AgentRunner.
PAYLOAD_WHITELIST = {
    "meta": {"workspace/state_space.json", "workspace/best_config.json", "workspace/last_judge.json"},
    "worker": {"exp/next_coordinate.json", "exp/coordinate_dimensions", "ai_docs/*", "exp/retry_error"},
    "judge": {"exp/simulation.log", "exp/config.json"},
    "doc": {"exp/config.json", "exp/judge.json", "exp/simulation_summary", "workspace/research_paper.md",
            "exp/doc_draft"},
    "doc_final": {"workspace/research_paper.md", "workspace/factory_summary"},
}


def new_canary(experiment: str) -> str:
    return f"CANARY-{experiment}-{secrets.token_hex(4)}"


class IsolationGuard:
    def __init__(self, canaries: dict[str, str]):
        self._canaries = canaries  # live reference to the persisted registry

    def check(self, agent: str, current_exp: str | None, sources: list[str], text: str) -> dict:
        allowed = PAYLOAD_WHITELIST.get(agent)
        if allowed is None:
            raise IsolationViolation(f"no payload whitelist for agent {agent!r}")
        for src in sources:
            if src not in allowed and not any(a.endswith("*") and src.startswith(a[:-1]) for a in allowed):
                raise IsolationViolation(f"{agent} payload source {src!r} is not whitelisted")

        foreign_ids = sorted({f"exp_{m}" for m in EXP_RE.findall(text)} - {current_exp})
        foreign_canaries = sorted(
            c for c in set(CANARY_RE.findall(text)) if not (current_exp and c.startswith(f"CANARY-{current_exp}-"))
        )
        for exp, canary in self._canaries.items():
            if exp != current_exp and canary in text and canary not in foreign_canaries:
                foreign_canaries.append(canary)
        if foreign_ids or foreign_canaries:
            raise IsolationViolation(
                f"isolation broken: {agent} prompt for {current_exp} references "
                f"{foreign_ids + foreign_canaries}"
            )
        return {
            "agent": agent,
            "current_experiment": current_exp,
            "sources": sources,
            "foreign_experiment_ids": [],
            "foreign_canaries": [],
            "prompt_chars": len(text),
            "verdict": "ISOLATED",
        }


def sanitize_text(text: str, replacement: str = "a prior experiment") -> str:
    text = CANARY_RE.sub("[canary-removed]", text)
    return EXP_RE.sub(replacement, text)


def sanitize_obj(obj, replacement: str = "a prior experiment"):
    """Deep-copy a JSON-like object, dropping canary/experiment keys and scrubbing ids in strings."""
    if isinstance(obj, dict):
        return {
            k: sanitize_obj(v, replacement)
            for k, v in obj.items()
            if k not in {"canary", "experiment", "exp_dir"}
        }
    if isinstance(obj, list):
        return [sanitize_obj(v, replacement) for v in obj]
    if isinstance(obj, str):
        return sanitize_text(obj, replacement)
    return copy.deepcopy(obj)


def scrub_foreign(obj, current_exp: str | None):
    """Remove references to experiments other than `current_exp` from agent output at ingestion.

    Agents never see foreign ids, but an LLM can still hallucinate one; scrubbing here keeps such a
    string from reaching a later prompt (where the guard would have to abort the factory).
    """
    if isinstance(obj, dict):
        return {k: scrub_foreign(v, current_exp) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_foreign(v, current_exp) for v in obj]
    if isinstance(obj, str):
        obj = CANARY_RE.sub(lambda m: m.group(0) if current_exp and m.group(0).startswith(f"CANARY-{current_exp}-")
                            else "[canary-removed]", obj)
        return EXP_RE.sub(lambda m: m.group(0) if m.group(0) == current_exp else "another experiment", obj)
    return obj


def paperize(text: str) -> str:
    """Rewrite experiment ids as iteration labels so the paper never carries raw exp ids."""
    text = CANARY_RE.sub("", text)
    return EXP_RE.sub(lambda m: f"Iteration {m.group(1)}", text)
