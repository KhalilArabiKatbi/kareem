"""Atomic JSON persistence for Python-owned factory state."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def new_factory_state(ai_docs_checksum: str) -> dict:
    return {
        "schema_version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "next_iteration": 0,
        "iterations": [],
        "scores": {},
        "sim_scores": {},
        "best": None,
        "last_judge_score": None,
        "meta_invalid_streak": 0,
        "last_dimension_added_iteration": -1,
        "poisoned": [],
        "canaries": {},
        "halted": None,
        "ai_docs_checksum": ai_docs_checksum,
        "llm_usage": {"calls": 0, "cost_usd": 0.0},
    }
