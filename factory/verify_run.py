"""Independent verification of the Definitions of Done and Fail.

    python -m factory.verify_run            # prints the JSON report

Each check re-reads artefacts from disk; nothing is trusted from the orchestrator's memory.
"""
from __future__ import annotations

import json
import re
import sys

from . import config
from .isolation import CANARY_RE, EXP_RE
from .router_audit import expected_route
from .state_store import read_json

TURN_RE = re.compile(r"(?m)^TURN \d{3}/050 \| ")
ALLOWED_ROOT = {"ai_docs", "specs", "factory", "workspace", "experiments", ".venv",
                ".claude", ".git"}  # tool metadata (Claude Code session locks, VCS), never deliverables
HALT_PREFIXES = ("max-iterations reached", "state space stagnant", "score plateau")


def _router_ok(log: dict) -> tuple[bool, str]:
    if log.get("verdict") != "ROUTER_VERIFIED":
        return False, f"verdict {log.get('verdict')}"
    for e in log.get("decisions", []):
        if e["outcome"] == "routing_violation":
            continue
        ctx, dec = e["context"], e["decision"]
        exp_model, exp_effort = expected_route(ctx)
        if (dec["model"], dec["effort"]) != (exp_model, exp_effort):
            return False, f"decision {e['seq']} {dec['model']}/{dec['effort']} != {exp_model}/{exp_effort}"
        if dec["model"] == "fable-5.1" and (ctx["agent"], ctx["purpose"]) != ("doc", "format"):
            return False, f"decision {e['seq']} routed fable-5.1 to {ctx['agent']}"
    return True, f"{len(log.get('decisions', []))} decisions re-derived"


def _isolation_ok(exp: str, d) -> tuple[bool, str]:
    n = 0
    for p in sorted((d / "prompts").glob("*.md")):
        text = p.read_text(encoding="utf-8")
        foreign = {f"exp_{m}" for m in EXP_RE.findall(text)} - {exp}
        foreign |= {c for c in CANARY_RE.findall(text) if not c.startswith(f"CANARY-{exp}-")}
        if foreign:
            return False, f"{p.name} references {sorted(foreign)}"
        n += 1
    audit = read_json(d / "isolation_audit.json", [])
    if any(a.get("verdict") != "ISOLATED" for a in audit):
        return False, "isolation_audit has a non-ISOLATED entry"
    return True, f"{n} prompts rescanned"


def check_experiment(i: int, state: dict | None = None, space: dict | None = None) -> dict:
    state = state or read_json(config.FACTORY_STATE_PATH)
    space = space or read_json(config.STATE_SPACE_PATH)
    exp, d = config.exp_name(i), config.exp_dir(i)
    row = next((r for r in state["iterations"] if r["iteration"] == i), None)
    status = row["status"] if row else "open"
    checks: dict[str, dict] = {}

    def put(name, ok, detail=""):
        checks[name] = {"ok": bool(ok), "detail": detail}

    paper_text = config.RESEARCH_PAPER_PATH.read_text(encoding="utf-8") if config.RESEARCH_PAPER_PATH.exists() else ""
    put("paper_section", re.search(rf"(?m)^### Iteration {i:03d}\b", paper_text), f"### Iteration {i:03d}")
    rl = read_json(d / "router_log.json", {})
    ok, why = _router_ok(rl) if rl else (False, "router_log.json missing")
    put("router_log_verified", ok, why)
    ok, why = _isolation_ok(exp, d)
    put("isolation", ok, why)

    if status == "poisoned":
        attempts = read_json(d / "attempts.json", [])
        put("poison_recorded", any(p["iteration"] == i for p in state.get("poisoned", [])), f"{len(attempts)} failures")
        put("three_attempts", len(attempts) == config.MAX_ITERATION_ATTEMPTS, f"{len(attempts)} attempts")
        return {"experiment": exp, "status": status, "micro_done": False, "micro_fail_recovered":
                all(c["ok"] for c in checks.values()), "checks": checks}

    nc = read_json(d / "next_coordinate.json", {})
    cfg = read_json(d / "config.json", {})
    ledger = {e["iteration"]: e for e in space.get("ledger", [])}
    put("config_maps_to_meta_coordinate",
        cfg.get("coordinate") == nc.get("coordinate") and cfg.get("coordinate_hash") == nc.get("coordinate_hash")
        and ledger.get(i, {}).get("coordinate_hash") == nc.get("coordinate_hash"),
        str(nc.get("coordinate_hash")))
    try:
        compile((d / "model.py").read_text(encoding="utf-8"), "model.py", "exec")
        compiled = True
    except (OSError, SyntaxError):
        compiled = False
    ran = (d / "sim_metrics.json").exists() and not (d / "sim_error.txt").exists()
    put("model_compiles_and_runs", compiled and ran, "compile + simulator exit 0")
    log_text = (d / "simulation.log").read_text(encoding="utf-8") if (d / "simulation.log").exists() else ""
    n_turns = len(TURN_RE.findall(log_text))
    put("simulation_log_50_turns", n_turns == 50, f"{n_turns} TURN lines")
    judge = read_json(d / "judge.json", {})
    score = judge.get("score")
    put("judge_json_valid", isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 100, f"score={score}")
    return {"experiment": exp, "status": status, "micro_done": all(c["ok"] for c in checks.values()), "checks": checks}


# String literals that are part of the simulator's documented model.py API (TRAIN_CONFIG keys, read
# by name) or of its metrics schema. A Meta dimension may legitimately share such a name; the audit
# looks for dimension enumerations, not API keys. The TRAIN_CONFIG keys come straight from the
# simulator (CONTRACT_KEYS), so the audit cannot drift from the API.
METRIC_AND_LEGACY_KEYS = {"features", "params", "finetune", "method", "generations", "population", "scenarios",
                          "sigma", "elite_frac", "enabled"}


def contract_vocabulary() -> set[str]:
    import ast

    tree = ast.parse((config.ORCHESTRATOR_DIR / "simulator" / "sim_runner.py").read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("TRAIN_LIMITS", "CONTRACT_KEYS")
                                                for t in node.targets):
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    keys.add(sub.value)
    return keys | METRIC_AND_LEGACY_KEYS


def _hardcoded_dimension_scan(space: dict) -> tuple[bool, str]:
    names = [n for n, s in {**space.get("dimensions", {}), **space.get("pruned_dimensions", {})}.items()
             if s.get("origin") == "meta" and n not in contract_vocabulary()]
    import ast

    wanted = set(names)
    hits = []
    for p in config.ORCHESTRATOR_DIR.rglob("*.py"):
        if "tests" in p.parts or ".cache" in p.parts:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        docstrings = {
            id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                if node.value in wanted:
                    hits.append(f"{p.name}:{node.value}")
    return (not hits), (f"{len(names)} meta-created dimension names; none is a string literal in harness code"
                        if not hits else f"found {hits[:10]}")


def build_report() -> dict:
    from .init_workspace import ai_docs_checksum

    state = read_json(config.FACTORY_STATE_PATH)
    space = read_json(config.STATE_SPACE_PATH)
    per_exp = [check_experiment(r["iteration"], state, space) for r in state["iterations"]]
    scored = [r for r in state["iterations"] if r["status"] == "scored"]

    macro: dict[str, dict] = {}

    def put(name, ok, detail=""):
        macro[name] = {"ok": bool(ok), "detail": detail}

    halted = state.get("halted") or {}
    put("halted_by_meta_done_condition", any(str(halted.get("reason", "")).startswith(p) for p in HALT_PREFIXES),
        halted.get("reason", "not halted"))
    if scored:
        top = max(scored, key=lambda r: (r["score"], r["sim_score"]))
        best = state.get("best") or {}
        exp_src = (config.exp_dir(top["iteration"]) / "model.py").read_text(encoding="utf-8")
        best_src = config.BEST_MODEL_PATH.read_text(encoding="utf-8") if config.BEST_MODEL_PATH.exists() else ""
        same = best_src.split("\n", 1)[1] == exp_src if "\n" in best_src else False
        put("best_model_is_highest_scoring", best.get("iteration") == top["iteration"] and same,
            f"iteration {top['iteration']} score {top['score']}")
    else:
        put("best_model_is_highest_scoring", False, "no scored iterations")
    text = config.RESEARCH_PAPER_PATH.read_text(encoding="utf-8")
    missing = [r["iteration"] for r in state["iterations"] if not re.search(rf"(?m)^### Iteration {r['iteration']:03d}\b", text)]
    put("paper_complete", "*Pending" not in text and not missing and "*Status: complete." in text,
        f"missing sections: {missing}" if missing else "all sections present")
    put("no_raw_experiment_ids_in_paper", not EXP_RE.search(text), "paper refers to iterations only")
    ok, why = _hardcoded_dimension_scan(space)
    put("state_space_not_hardcoded", ok, why)
    put("seed_was_single_dimension",
        len(read_json(config.TEMPLATES_DIR / "state_space_seed.json")["dimensions"]) == 1, "templates/state_space_seed.json")
    extra = sorted(p.name for p in config.PROJECT_ROOT.iterdir() if p.name not in ALLOWED_ROOT)
    put("deliverables_in_designated_dirs", not extra, f"unexpected top-level entries: {extra}" if extra else "ok")
    put("ai_docs_read_only", ai_docs_checksum() == state["ai_docs_checksum"], "checksum unchanged")
    put("no_macro_fail_recorded", not (config.WORKING_DIR / "macro_fail.json").exists(), "")

    return {
        "generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "halted": halted,
        "best": state.get("best"),
        "iterations": len(state["iterations"]),
        "micro_done_summary": {
            "scored": sum(1 for e in per_exp if e["status"] == "scored"),
            "scored_all_checks_passed": sum(1 for e in per_exp if e["micro_done"]),
            "poisoned": sum(1 for e in per_exp if e["status"] == "poisoned"),
            "poisoned_recovered_cleanly": sum(1 for e in per_exp if e.get("micro_fail_recovered")),
        },
        "macro_done": {"all_passed": all(c["ok"] for c in macro.values()), "checks": macro},
        "experiments": per_exp,
        "llm_usage": state.get("llm_usage"),
    }


if __name__ == "__main__":
    rep = build_report()
    json.dump(rep, sys.stdout, indent=2)
    print()
    sys.exit(0 if rep["macro_done"]["all_passed"] else 1)
