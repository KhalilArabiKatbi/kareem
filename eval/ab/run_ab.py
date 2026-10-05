"""Online A/B: native auto-compact vs hand threshold vs Kareem, same synthetic tasks, real actuation.

All three arms run through GuardedAgent with the same actuator; only the decision function differs.
Every arm keeps Claude Code's own auto-compact as a backstop, with the effective window shrunk to
CAPACITY tokens so context pressure appears within ~8 turns (cheap) and native compaction really fires.

  python -m eval.ab.run_ab --tasks 2 --repeats 1 --budget 15            # pilot
  python -m eval.ab.run_ab --tasks 8 --repeats 2 --budget 60 --workers 3

Outputs (gitignored): eval/out/ab/results.jsonl  (one row per run; resumable)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.ab.tasks import make_task, score  # noqa: E402
from kareem.actions import Action  # noqa: E402
from kareem.runtime import Decision  # noqa: E402

OUT = ROOT / "eval" / "out" / "ab"
CAPACITY = 100_000
MODEL = "claude-sonnet-5-5"
PER_RUN_BUDGET = 5.0
ARMS = ["native", "threshold", "kareem"]
NATIVE_CLAUDE = str(Path.home() / "AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe")


@dataclass
class FixedPolicy:
    """Decision function stub with the same `decide` interface as InterventionPolicy."""
    kind: str
    thr: float = 0.6

    def decide(self, history, min_confidence=0.0) -> Decision:
        util = history[-1]["utilization"] if history else 0.0
        act = Action.COMPACT if (self.kind == "threshold" and util >= self.thr) else Action.NOOP
        return Decision(action=act, confidence=1.0, probabilities={act.name: 1.0}, deferred=False)


def make_policy(arm: str):
    if arm == "kareem":
        from kareem import load_policy
        return load_policy()
    return FixedPolicy(arm)


async def run_one(arm: str, seed: int, workdir: Path) -> dict:
    from claude_agent_sdk import ClaudeAgentOptions
    from kareem.integrations.claude_agent import GuardedAgent

    task = make_task(seed, workdir / "repo")
    opts = ClaudeAgentOptions(
        model=MODEL, cwd=str(workdir / "repo"), permission_mode="acceptEdits", cli_path=NATIVE_CLAUDE,
        # Least privilege: file tools inside the temp repo, Bash only for the two commands the task needs.
        allowed_tools=["Read", "Edit", "Write", "Grep", "Glob", "Bash(python tools/check.py)", "Bash(python -m pytest:*)"],
        disallowed_tools=["WebFetch", "WebSearch", "Task"], setting_sources=[],
        max_budget_usd=PER_RUN_BUDGET, effort="low"
        )
    log, peak = [], 0
    results_by_session: dict[str, list[float]] = {}
    compactions = 0
    result_errors: list[str] = []
    t0 = time.time()

    def tap(msg):
        nonlocal compactions
        c = getattr(msg, "total_cost_usd", None)
        if c is not None and getattr(msg, "duration_ms", None) is not None:
            results_by_session.setdefault(getattr(msg, "session_id", "?"), []).append(float(c))
        if getattr(msg, "duration_ms", None) is not None and (getattr(msg, "is_error", False) or getattr(msg, "subtype", "success") != "success"):
            result_errors.append(f"{getattr(msg, 'subtype', '?')}:{str(getattr(msg, 'result', ''))[:80]}")
        if type(msg).__name__ == "SystemMessage" and "compact" in str(getattr(msg, "subtype", "")):
            compactions += 1

    from claude_agent_sdk import ClaudeSDKClient

    class TapClient(ClaudeSDKClient):  # sees EVERY message, including the summary calls GuardedAgent makes
        async def receive_response(self):
            async for m in super().receive_response():
                tap(m)
                yield m

    async with GuardedAgent(options=opts, policy=make_policy(arm), instructions=task["instructions"],
                            min_confidence=0.35, capacity_tokens=CAPACITY, client_factory=TapClient) as agent:
        orig = agent._absorb

        def absorb(msg):  # per-call usage from AssistantMessage = real context size
            nonlocal peak
            u = getattr(msg, "usage", None)
            if isinstance(u, dict) and hasattr(msg, "message_id"):
                agent._prev_tokens = int(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                                         + u.get("cache_creation_input_tokens", 0))
                peak = max(peak, agent._prev_tokens)
                return
            keep = agent._prev_tokens
            orig(msg)
            if getattr(msg, "duration_ms", None) is not None:  # ResultMessage usage is a per-turn SUM, not context size
                agent._prev_tokens = keep

        agent._absorb = absorb
        for turn in task["turns"]:
            async for _ in agent.run_turn(turn):
                pass
            rec = agent.turns[-1]
            log.append({"util": round(rec.obs["utilization"], 3), "action": rec.decision.action_name,
                        "deferred": rec.decision.deferred, "actuated": rec.actuated})
    # total_cost_usd may be cumulative per session or per query: cumulative lists are non-decreasing
    cost = sum(v[-1] if v == sorted(v) else sum(v) for v in results_by_session.values())
    sc = score(workdir / "repo", task["hidden"])
    return {"arm": arm, "seed": seed, "cost": round(cost, 3), "peak_tokens": peak, "resets": agent.session_resets, "native_compactions": compactions,
            "cost_lists": results_by_session, "result_errors": result_errors,
            "turn_log": log, "score": sc, "secs": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["arm"], r["seed"], r.get("rep", 0)))
    jobs = [(arm, 100 + s, rep) for s in range(a.tasks) for rep in range(a.repeats) for arm in a.arms.split(",")
            if (arm, 100 + s, rep) not in done]
    print(f"{len(jobs)} runs, budget ${a.budget}")
    spent, lock, stop = [0.0], threading.Lock(), threading.Event()

    def work(job):
        arm, seed, rep = job
        if stop.is_set():
            return "skipped"
        d = Path(tempfile.mkdtemp(prefix=f"ab_{arm}_{seed}_"))
        try:
            r = asyncio.run(run_one(arm, seed, d))
        except Exception as e:  # noqa: BLE001
            print("FAIL", job, repr(e)[:200])
            return "fail"
        finally:
            shutil.rmtree(d, ignore_errors=True)
        if r["cost"] <= 0 or r["peak_tokens"] == 0 or r["result_errors"]:  # dead/errored run: never record as a result
            print("INVALID", job, r["cost"], r["peak_tokens"], r["result_errors"][:2])
            return "invalid"
        r["rep"] = rep
        with lock:
            spent[0] += r["cost"]
            with res_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            if spent[0] >= a.budget:
                stop.set()
        print(f"ok {arm:9} seed={seed} rep={rep} score={r['score']['total']}/{r['score']['max']} cost=${r['cost']:.2f} "
              f"peak={r['peak_tokens']} resets={r['resets']} total=${spent[0]:.2f}")
        return "ok"

    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(work, jobs))
    print({k: out.count(k) for k in set(out)}, f"spent=${spent[0]:.2f}")


if __name__ == "__main__":
    main()
