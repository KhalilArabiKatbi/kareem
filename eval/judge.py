"""Opus judge: label sampled checkpoints of real sessions with one of Kareem's six actions.

Deterministic harness: fixed prompt, fixed checkpoint sampling, structured output, results
cached by content hash. CAUSAL: at each checkpoint the judge sees only what exists at that
step (first user prompt + a recent window + the same telemetry the MLP gets), never the
future, exactly like the policy in real use.

  python -m eval.judge --dry-run                 # chunk/token estimate, no calls
  python -m eval.judge --limit-chunks 4          # pilot
  python -m eval.judge --budget 25               # full run, stops at the cap (USD)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.transcript import parse, render, steps_with_estimates  # noqa: E402

OUT = ROOT / "eval" / "out"
MODEL = "claude-opus-5-5"
ACTIONS = ["NOOP", "COMPACT", "PRUNE_TOOLS", "REINJECT_INSTRUCTIONS", "CHECKPOINT_RESET", "RETRIEVE_MEMORY"]
CHUNK_CHARS = 400_000
EVERY = 8
MAX_CP = 30
PROMPT_VERSION = "v2-causal"
WINDOW_CHARS = 120_000   # ~30k tokens of recent history
PER_SESSION = 12

SYSTEM = """You are an expert reviewer of long-running AI coding-agent sessions (Claude Code). You label checkpoints with the single context-management action that would most help the rest of the session. You see ONLY the session up to the checkpoint (task prompt, a recent window, live telemetry); you cannot see the future, exactly like the monitor you are being compared to. Decide what a monitor should do right now.

Actions:
- NOOP: no intervention helps; context is healthy or any action would cost more than it saves.
- COMPACT: summarise the conversation and replace it with the summary. Right when context is large/noisy and the work is at a natural boundary, with little detail that must stay verbatim.
- PRUNE_TOOLS: drop old tool outputs, keep recent ones. Right when bulky, already-consumed tool results dominate the context but the dialogue and decisions are still valuable.
- REINJECT_INSTRUCTIONS: re-append the original task instructions / constraints. Right when the agent is drifting from the user's goal or constraints, or forgot a stated rule.
- CHECKPOINT_RESET: fresh context seeded with a checkpoint summary. Right when context is badly degraded: loops, contradictions, repeated failures, confusion that compaction cannot fix.
- RETRIEVE_MEMORY: pull earlier facts/notes back in. Right when the agent lost or contradicts a fact established earlier (re-reads, re-asks, wrong guesses about earlier decisions).

Rules: Prefer NOOP unless an action clearly pays off. Context window is 1,000,000 tokens; the markers give real context_tokens. Output only the structured result."""

SCHEMA = {"type": "object", "required": ["action", "confidence", "reason"], "properties": {
    "action": {"type": "string", "enum": ACTIONS}, "confidence": {"type": "number"},
    "reason": {"type": "string"}}}


def checkpoints_for(path: Path) -> list[dict]:
    from eval.replay import MIN_UTIL  # noqa: F401
    events = parse(path)
    steps = steps_with_estimates(events)
    by_step = {s["step"]: s for s in steps}
    err_run, cps = 0, []
    for s in steps:
        err_run = err_run + 1 if s["error"] else 0
        if s["step"] > 0 and (s["step"] % EVERY == 0 or (err_run >= 3 and err_run % 2 == 1)):
            cps.append((s["step"], err_run))
    if len(cps) > PER_SESSION:
        idx = sorted({round(j * (len(cps) - 1) / (PER_SESSION - 1)) for j in range(PER_SESSION)})
        cps = [cps[j] for j in idx]
    first = next((e["text"] for e in events if e["kind"] == "prompt"), "")
    pos = {e["step"]: i for i, e in enumerate(events) if e["kind"] == "tool_result"}
    out = []
    for step, err in cps:
        upto = events[: pos[step] + 1]
        used, tail = 0, []
        for ev in reversed(upto):
            r = render([ev])
            if used + len(r) > WINDOW_CHARS:
                break
            tail.append(r)
            used += len(r)
        omitted = len(upto) - len(tail)
        since_compact = step - max((e["step"] for e in upto if e["kind"] == "tool_result"
                                    and by_step[e["step"]]["compact_before"]), default=0)
        st = by_step[step]
        header = (f"TELEMETRY at step {step}: context_tokens={st['tokens']} utilization={st['tokens'] / 1e6:.3f} "
                  f"error_streak={err} steps_since_compaction={since_compact} "
                  f"redundancy_est={st['redundancy_est']} stale_est={st['stale_est']}")
        first_r = render([{"kind": "prompt", "text": first}])[:3000]
        body = f"{first_r}\n[... {omitted} earlier events omitted ...]\n" + "\n".join(reversed(tail))
        out.append({"session": path.stem, "project": path.parent.name, "step": step, "header": header, "body": body})
    return out


def prompt_for(c: dict) -> str:
    return (f"{c['header']}\n\n=== SESSION SO FAR ===\n{c['body']}\n"
            f"=== NOW (step {c['step']}) ===\nWhich single action should the monitor take at this step?")


def call_opus(user_prompt: str) -> dict:
    claude = shutil.which("claude") or "claude"
    p = Path(claude)
    if p.suffix.lower() in (".cmd", ".bat", ".ps1", ""):
        native = p.parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if native.exists():
            claude = str(native)
    cmd = [claude, "-p", "--model", MODEL, "--effort", "high", "--tools", "", "--setting-sources", "",
           "--strict-mcp-config", "--no-session-persistence", "--system-prompt", SYSTEM,
           "--output-format", "json", "--json-schema", json.dumps(SCHEMA, separators=(",", ":"))]
    with tempfile.TemporaryDirectory() as d:
        proc = subprocess.run(cmd, input=user_prompt, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=d, timeout=1200)
    env = json.loads(proc.stdout)
    if env.get("is_error") or env.get("subtype") != "success":
        raise RuntimeError(str(env.get("result") or proc.stderr)[-300:])
    return {"label": env.get("structured_output") or {}, "cost": float(env.get("total_cost_usd") or 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-size", type=int, default=500_000)
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from eval.replay import find_sessions
    chunks = [c for p in find_sessions(a.min_size) for c in checkpoints_for(p)]
    if a.limit_chunks:
        step = max(1, len(chunks) // a.limit_chunks)
        chunks = chunks[::step][: a.limit_chunks]
    toks = sum(len(prompt_for(c)) // 4 for c in chunks)
    print(f"{len(chunks)} chunks, ~{toks:,} input tokens, (1 checkpoint per call)")
    if a.dry_run:
        return
    jd = OUT / "judge"
    jd.mkdir(parents=True, exist_ok=True)
    spent, lock, stop = [0.0], threading.Lock(), threading.Event()

    def work(c):
        key = hashlib.sha256((PROMPT_VERSION + MODEL + prompt_for(c)).encode()).hexdigest()[:16]
        f = jd / f"{c['session']}_{c['step']}_{key}.json"
        if f.exists():
            return "cached"
        if stop.is_set():
            return "skipped"
        try:
            r = call_opus(prompt_for(c))
        except Exception as e:  # noqa: BLE001
            print("FAIL", c["session"][:8], c["step"], str(e)[:150])
            return "fail"
        with lock:
            spent[0] += r["cost"]
            if spent[0] >= a.budget:
                stop.set()
        f.write_text(json.dumps({"session": c["session"], "project": c["project"], "step": c["step"],
                                 "cost": r["cost"], "label": r["label"]}),
                     encoding="utf-8")
        print(f"ok {c['session'][:8]} step {c['step']} {r['label'].get('action')} cost=${r['cost']:.2f} total=${spent[0]:.2f}")
        return "ok"

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(work, chunks))
    print({k: res.count(k) for k in set(res)}, f"spent=${spent[0]:.2f}")


if __name__ == "__main__":
    main()
