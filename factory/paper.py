"""Deterministic assembly of workspace/research_paper.md.

LLMs write prose for bounded blocks; Python decides where it goes, builds every table, and makes
sure no raw experiment id ever lands in the paper (it is fed back to later agents).
"""
from __future__ import annotations

import json
import re

from .isolation import paperize

BLOCKS = ("STATUS", "ABSTRACT", "RESULTS_TABLE", "CONFIRMATION", "RESULTS_NARRATIVE", "ITERATIONS", "DISCUSSION",
          "CONCLUSION", "APPENDIX_SPACE")
NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")


def _block_re(name: str) -> re.Pattern:
    return re.compile(rf"(<!-- BEGIN:{name} -->\n)(.*?)(<!-- END:{name} -->)", re.S)


def get_block(text: str, name: str) -> str:
    m = _block_re(name).search(text)
    if not m:
        raise ValueError(f"paper block {name} missing")
    return m.group(2)


def replace_block(text: str, name: str, content: str) -> str:
    rx = _block_re(name)
    if not rx.search(text):
        raise ValueError(f"paper block {name} missing")
    body = content.strip("\n") + "\n" if content.strip() else ""
    return rx.sub(lambda m: m.group(1) + body + m.group(3), text, count=1)


def iteration_heading(iteration: int) -> str:
    return f"### Iteration {iteration:03d}"


def upsert_iteration(text: str, iteration: int, section: str) -> str:
    """Insert (or replace, on retry) the section for `iteration`, keeping iterations ordered."""
    body = get_block(text, "ITERATIONS")
    parts = re.split(r"(?m)^(?=### Iteration \d{3})", body)
    sections = {}
    for p in parts:
        m = re.match(r"### Iteration (\d{3})", p)
        if m:
            sections[int(m.group(1))] = p.strip("\n")
    sections[iteration] = paperize(section).strip("\n")
    new_body = "\n\n".join(sections[k] for k in sorted(sections))
    return replace_block(text, "ITERATIONS", new_body)


def has_iteration(text: str, iteration: int) -> bool:
    return re.search(rf"(?m)^{re.escape(iteration_heading(iteration))}\b", text) is not None


def leaderboard(state: dict, ledger: list[dict]) -> str:
    rows = []
    for it in state.get("iterations", []):
        i = it["iteration"]
        coord = next((e["coordinate"] for e in ledger if e["iteration"] == i), None)
        cstr = ", ".join(f"{k}={json.dumps(v)}" for k, v in (coord or {}).items())
        if len(cstr) > 160:
            cstr = cstr[:157] + "..."
        judge = "—" if it.get("score") is None else str(it["score"])
        sim = "—" if it.get("sim_score") is None else f"{it['sim_score']:.2f}"
        ret = "—" if it.get("mean_return") is None else f"{it['mean_return']:+.2f}"
        mark = " **(best)**" if state.get("best") and state["best"]["iteration"] == i else ""
        rows.append(f"| {i:03d} | {it['status']}{mark} | {judge} | {sim} | {ret} | {it.get('dims_added', 0)} | `{cstr}` |")
    if not rows:
        return "*No completed iterations yet.*"
    head = [
        "| Iter. | Status | Judge score | sim_score | Mean return | Dims added | Coordinate |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    base = state.get("baselines")
    foot = []
    if base:
        foot = [
            "",
            f"Reference returns on the 200 evaluation scenarios (50 turns x 4 seeds): do-nothing {base['noop_return']:+.2f}, "
            f"heuristic {base['heuristic_return']:+.2f}, teacher v1 {base['oracle_return']:+.2f}, "
            f"teacher v2 {base.get('teacher_v2_return', float('nan')):+.2f}, "
            f"clairvoyant bound {base['clairvoyant_return']:+.2f}.",
        ]
    return "\n".join(head + rows + foot)


def space_evolution(space: dict) -> str:
    lines = ["| Iter. | Operation | Dimension | Detail |", "|---:|---|---|---|"]
    for m in space.get("mutation_log", []):
        detail = json.dumps(m.get("detail"))
        if len(detail) > 120:
            detail = detail[:117] + "..."
        lines.append(f"| {m['iteration']:03d} | {m['op']} | `{m['dimension']}` | `{detail}` |")
    active = ", ".join(f"`{k}`" for k in space.get("dimensions", {}))
    pruned = ", ".join(f"`{k}`" for k in space.get("pruned_dimensions", {})) or "none"
    summary = [
        "",
        f"Active dimensions ({len(space.get('dimensions', {}))}): {active}.",
        "",
        f"Pruned dimensions: {pruned}.",
    ]
    if len(lines) == 2:
        return "*No mutations applied yet; the space holds only the seed dimension.*\n" + "\n".join(summary)
    return "\n".join(lines + summary)


def confirmation_table(conf: dict) -> str:
    rows = conf.get("rows", [])
    if not rows:
        return "*No confirmation re-score was run.*"
    lines = [
        "| Iteration | Judge score | Evaluation sim_score | Confirmation sim_score | Confirmation return | "
        "Paired delta vs selected best (return) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['iteration']:03d} | {r['judge']} | {r['eval_sim']:.2f} | failed | failed | failed |")
            continue
        mark = " **(confirmed best)**" if r["iteration"] == conf.get("confirmed_best") else ""
        lines.append(
            f"| {r['iteration']:03d}{mark} | {r['judge']} | {r['eval_sim']:.2f} | {r['confirm_sim']:.2f} | "
            f"{r['confirm_return']:+.3f} | {r['delta_vs_selected']:+.3f} +- {r['se']:.3f} |")
    lines += ["", conf.get("note", "")]
    return "\n".join(lines)


def numbers(text: str) -> set[str]:
    return set(NUM_RE.findall(text))


def format_preserves(draft: str, formatted: str) -> tuple[bool, str]:
    """Check that the Fable formatting pass changed formatting only."""
    first = draft.strip().splitlines()[0].strip()
    if not formatted.strip().startswith(first):
        return False, "first heading line changed"
    missing = numbers(draft) - numbers(formatted)
    if missing:
        return False, f"numbers lost: {sorted(missing)[:10]}"
    ratio = len(formatted) / max(1, len(draft))
    if not 0.6 <= ratio <= 1.6:
        return False, f"length ratio {ratio:.2f} outside [0.6, 1.6]"
    if re.search(r"(?m)^#{1,2} ", formatted):
        return False, "level-1/2 heading introduced"
    return True, "ok"
