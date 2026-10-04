"""Generic state-space mutation engine.

This module knows the *grammar* of a state space (dimension types, mutation ops, domain checks).
It knows no dimension names. Every dimension beyond the seed comes from the Meta-Agent.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

from .errors import MetaProposalInvalid

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
MAX_ACTIVE_DIMS = 32
MAX_CATEGORICAL_VALUES = 16
MAX_MUTATIONS_PER_ITERATION = 6
MAX_LIST_LEN = 8
MAX_STR_LEN = 80
DIM_TYPES = ("categorical", "integer", "float")
OPS = ("add_dimension", "add_values", "prune_values", "prune_dimension")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def coordinate_hash(coord: dict) -> str:
    return hashlib.sha256(canonical(coord).encode()).hexdigest()[:12]


def _check_value(v, where: str) -> None:
    if isinstance(v, bool) or v is None:
        return
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            raise MetaProposalInvalid(f"{where}: non-finite number")
        return
    if isinstance(v, str):
        if not 0 < len(v) <= MAX_STR_LEN:
            raise MetaProposalInvalid(f"{where}: string values must be 1..{MAX_STR_LEN} chars")
        return
    if isinstance(v, list):
        if not 0 < len(v) <= MAX_LIST_LEN or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in v
        ):
            raise MetaProposalInvalid(f"{where}: list values must hold 1..{MAX_LIST_LEN} numbers")
        return
    raise MetaProposalInvalid(f"{where}: unsupported value type {type(v).__name__}")


def _validate_spec(name: str, spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise MetaProposalInvalid(f"dimension {name}: spec must be an object")
    kind = spec.get("type")
    if kind not in DIM_TYPES:
        raise MetaProposalInvalid(f"dimension {name}: type must be one of {DIM_TYPES}")
    desc = spec.get("description", "")
    if not isinstance(desc, str) or not 10 <= len(desc) <= 800:
        raise MetaProposalInvalid(f"dimension {name}: description must be 10..800 chars")
    out = {"type": kind, "description": desc}
    if kind == "categorical":
        values = spec.get("values")
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_CATEGORICAL_VALUES:
            raise MetaProposalInvalid(f"dimension {name}: categorical needs 1..{MAX_CATEGORICAL_VALUES} values")
        seen = set()
        for i, v in enumerate(values):
            _check_value(v, f"dimension {name} value[{i}]")
            c = canonical(v)
            if c in seen:
                raise MetaProposalInvalid(f"dimension {name}: duplicate value {c}")
            seen.add(c)
        out["values"] = list(values)
    else:
        lo, hi = spec.get("min"), spec.get("max")
        num = int if kind == "integer" else (int, float)
        if not isinstance(lo, num) or not isinstance(hi, num) or isinstance(lo, bool) or isinstance(hi, bool):
            raise MetaProposalInvalid(f"dimension {name}: {kind} needs numeric min and max")
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi or (kind == "float" and lo == hi):
            raise MetaProposalInvalid(f"dimension {name}: invalid range [{lo}, {hi}]")
        out["min"], out["max"] = lo, hi
    return out


def in_domain(dim: dict, value) -> bool:
    kind = dim["type"]
    if kind == "categorical":
        c = canonical(value)
        return any(canonical(v) == c for v in dim["values"])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if kind == "integer" and not (isinstance(value, int) or float(value).is_integer()):
        return False
    return dim["min"] <= value <= dim["max"]


class StateSpace:
    def __init__(self, data: dict):
        self.data = data

    @property
    def dimensions(self) -> dict:
        return self.data["dimensions"]

    def visited_hashes(self) -> set[str]:
        return {e["coordinate_hash"] for e in self.data.get("ledger", [])}

    def poisoned_hashes(self) -> set[str]:
        return {e["coordinate_hash"] for e in self.data.get("ledger", []) if e.get("status") == "poisoned"}

    def apply_wave_proposal(self, proposal: dict, iterations: list[int],
                            reserved: set[str] | None = None) -> tuple["StateSpace", list[dict], dict]:
        """Validate a wave proposal: shared mutations + exactly len(iterations) distinct coordinates.

        Mutations are attributed to the wave's first iteration. Returns (new_space, coords, report).
        """
        cands = proposal.get("candidates")
        if not isinstance(cands, list) or len(cands) != len(iterations):
            raise MetaProposalInvalid(
                f"candidates must hold exactly {len(iterations)} coordinates (one per parallel iteration), "
                f"got {len(cands) if isinstance(cands, list) else type(cands).__name__}"
            )
        first = {"mutations": proposal.get("mutations", []), "next_coordinate": cands[0].get("next_coordinate")}
        new_space, c0, report = self.apply_proposal(first, iterations[0])
        if reserved and report["coordinate_hash"] in reserved:
            raise MetaProposalInvalid("candidates[0] repeats a coordinate already used in this wave")
        coords, hashes = [c0], [report["coordinate_hash"]]
        for k, cand in enumerate(cands[1:], start=1):
            try:
                _, ck, rk = new_space.apply_proposal({"mutations": [], "next_coordinate": cand.get("next_coordinate")},
                                                     iterations[k])
            except MetaProposalInvalid as e:
                raise MetaProposalInvalid(f"candidates[{k}]: {e}") from e
            if reserved and rk["coordinate_hash"] in reserved:
                raise MetaProposalInvalid(f"candidates[{k}] repeats a coordinate already used in this wave")
            if rk["coordinate_hash"] in hashes:
                raise MetaProposalInvalid(f"candidates[{k}] duplicates candidates[{hashes.index(rk['coordinate_hash'])}]")
            coords.append(ck)
            hashes.append(rk["coordinate_hash"])
        report = dict(report, coordinate_hashes=hashes)
        return new_space, coords, report

    def apply_proposal(self, proposal: dict, iteration: int) -> tuple["StateSpace", dict, dict]:
        """Validate and apply a Meta proposal on a copy. Returns (new_space, coordinate, report).

        Raises MetaProposalInvalid on the first problem; the original space is never modified.
        """
        data = copy.deepcopy(self.data)
        dims = data["dimensions"]
        pruned = data.setdefault("pruned_dimensions", {})
        mutations = proposal.get("mutations", [])
        if not isinstance(mutations, list) or len(mutations) > MAX_MUTATIONS_PER_ITERATION:
            raise MetaProposalInvalid(f"mutations must be a list of at most {MAX_MUTATIONS_PER_ITERATION} ops")

        applied, added = [], []
        for k, m in enumerate(mutations):
            op, name = m.get("op"), m.get("name")
            where = f"mutation[{k}] {op} {name}"
            if op not in OPS:
                raise MetaProposalInvalid(f"{where}: unknown op; allowed {OPS}")
            if not isinstance(name, str) or not NAME_RE.match(name):
                raise MetaProposalInvalid(f"{where}: name must match {NAME_RE.pattern}")
            if op == "add_dimension":
                if name in dims:
                    raise MetaProposalInvalid(f"{where}: dimension already active")
                if len(dims) >= MAX_ACTIVE_DIMS:
                    raise MetaProposalInvalid(f"{where}: at most {MAX_ACTIVE_DIMS} active dimensions")
                spec = _validate_spec(name, m.get("spec"))
                spec.update({"added_iteration": iteration, "origin": "meta"})
                dims[name] = spec
                pruned.pop(name, None)
                added.append(name)
                detail = spec.get("values", [spec.get("min"), spec.get("max")])
            elif op == "add_values":
                dim = dims.get(name)
                if dim is None or dim["type"] != "categorical":
                    raise MetaProposalInvalid(f"{where}: needs an active categorical dimension")
                values = m.get("values")
                if not isinstance(values, list) or not values:
                    raise MetaProposalInvalid(f"{where}: values must be a non-empty list")
                existing = {canonical(v) for v in dim["values"]}
                for i, v in enumerate(values):
                    _check_value(v, f"{where} value[{i}]")
                    if canonical(v) in existing:
                        raise MetaProposalInvalid(f"{where}: value {canonical(v)} already present")
                    existing.add(canonical(v))
                if len(existing) > MAX_CATEGORICAL_VALUES:
                    raise MetaProposalInvalid(f"{where}: more than {MAX_CATEGORICAL_VALUES} values")
                dim["values"].extend(values)
                detail = values
            elif op == "prune_values":
                dim = dims.get(name)
                if dim is None or dim["type"] != "categorical":
                    raise MetaProposalInvalid(f"{where}: needs an active categorical dimension")
                drop = {canonical(v) for v in (m.get("values") or [])}
                keep = [v for v in dim["values"] if canonical(v) not in drop]
                if not drop or len(keep) == len(dim["values"]):
                    raise MetaProposalInvalid(f"{where}: none of the values exist")
                if not keep:
                    raise MetaProposalInvalid(f"{where}: would leave the dimension empty (use prune_dimension)")
                dim["values"] = keep
                detail = m.get("values")
            else:  # prune_dimension
                if name not in dims:
                    raise MetaProposalInvalid(f"{where}: dimension not active")
                if len(dims) <= 1:
                    raise MetaProposalInvalid(f"{where}: cannot prune the last active dimension")
                spec = dims.pop(name)
                spec.update({"pruned_iteration": iteration, "reason": str(m.get("reason", ""))[:300]})
                pruned[name] = spec
                detail = m.get("reason", "")
            applied.append({"iteration": iteration, "op": op, "dimension": name, "detail": detail})

        coord = proposal.get("next_coordinate")
        if not isinstance(coord, dict):
            raise MetaProposalInvalid("next_coordinate must be an object")
        missing = sorted(set(dims) - set(coord))
        extra = sorted(set(coord) - set(dims))
        if missing or extra:
            raise MetaProposalInvalid(
                f"next_coordinate must assign exactly the active dimensions; missing={missing} extra={extra}"
            )
        for name, value in coord.items():
            if not in_domain(dims[name], value):
                raise MetaProposalInvalid(f"next_coordinate[{name}]={canonical(value)} is outside its domain")
        coord = {k: coord[k] for k in sorted(coord)}
        h = coordinate_hash(coord)
        if h in self.poisoned_hashes():
            raise MetaProposalInvalid(f"next_coordinate {h} is poisoned")
        if h in self.visited_hashes():
            raise MetaProposalInvalid(f"next_coordinate {h} was already evaluated; choose an unexplored point")

        data.setdefault("mutation_log", []).extend(applied)
        data["revision"] = data.get("revision", 0) + (1 if applied else 0)
        if added:
            data["last_dimension_added_iteration"] = iteration
        report = {"applied": applied, "dimensions_added": added, "coordinate_hash": h}
        return StateSpace(data), coord, report

    def record(self, iteration: int, coord: dict, status: str, score: int | None,
               sim_score: float | None = None) -> None:
        ledger = self.data.setdefault("ledger", [])
        h = coordinate_hash(coord)
        ledger[:] = [e for e in ledger if e["iteration"] != iteration]
        ledger.append({"iteration": iteration, "coordinate_hash": h, "coordinate": coord, "status": status,
                       "score": score, "sim_score": sim_score})
        ledger.sort(key=lambda e: e["iteration"])
