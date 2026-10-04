#!/usr/bin/env python
"""Context Health MLP Cognitive Software Factory: the deterministic harness.

    python factory/run_factory.py --max-iterations 50 [--parallel 4]

Python owns the loop, every stop decision, every model choice and every file write. Agents are
pure functions from a whitelisted prompt to JSON. See specs/optimizer/implementation_plan.md.

Execution model: synchronous waves.
    A  (serial)    Meta-Agent proposes shared mutations + K coordinates   (one call, retried/escalated)
    B  (parallel)  K x [Worker -> Simulator -> Judge], each with its own 3-attempt retry loop
    C  (serial)    state update in iteration order (scores, ledger, best, poison)
    D  (parallel)  K x Documenting Agent (Sonnet draft + Fable format)
    E  (serial)    paper assembly + iteration close, in iteration order
Threads only compute; every shared-state mutation happens in the main thread at a barrier, in
iteration order, so the outcome never depends on which thread finishes first.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    # Launched as a script: re-exec inside the project venv (torch lives there), then import the package.
    _venv_py = _ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if _venv_py.exists() and os.environ.get("FACTORY_REEXEC") != "1" and Path(sys.executable).resolve() != _venv_py.resolve():
        _env = dict(os.environ, FACTORY_REEXEC="1")
        sys.exit(subprocess.call([str(_venv_py), str(Path(__file__).resolve()), *sys.argv[1:]], env=_env))
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

from factory import analytics, config, done_checker, paper, verify_run  # noqa: E402
from factory.agents import AgentCall, AgentRunner  # noqa: E402
from factory.code_guard import check_model_source  # noqa: E402
from factory.errors import (  # noqa: E402
    AgentCallError, AgentOutputError, InfraUnavailable, JudgeAnchorError, MacroFail, MicroFail,
    ReadOnlyViolation, SimulatorError, WorkerCodeError,
)
from factory.init_workspace import ai_docs_checksum, initialize, is_initialized  # noqa: E402
from factory.isolation import IsolationGuard, new_canary, paperize, sanitize_obj  # noqa: E402
from factory.llm_client import ClaudeCLIClient  # noqa: E402
from factory.router import EscalationTracker, ModelRouter, RouteContext, RouterLog  # noqa: E402
from factory.simulator import harness as sim_harness  # noqa: E402
from factory.simulator.dataset import load_cache  # noqa: E402
from factory.state_space import StateSpace  # noqa: E402
from factory.state_store import read_json, write_json, write_text  # noqa: E402

BUILD_STAGES = ("worker", "fingerprint")
CORE_STAGES = ("worker", "fingerprint", "simulate", "judge")
STAGE_AGENT = {"meta": "meta", "worker": "worker", "fingerprint": "worker", "simulate": "worker", "judge": "judge",
               "doc": "doc"}


class HarnessStageError(MicroFail):
    """An unexpected exception inside a stage, converted so the factory survives."""


def _dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


@dataclass
class Member:
    """One iteration inside a wave. Owned by exactly one thread during the parallel phases."""

    i: int
    exp: str
    d: Path
    canary: str
    rlog: RouterLog
    esc: EscalationTracker = field(default_factory=EscalationTracker)
    memo: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    retry_error: str | None = None
    consecutive_fails: int = 0
    status: str = "open"            # open -> ready | poisoned
    section: str | None = None      # paper section produced in phase D (or by Python)
    draft: dict | None = None
    duplicate_of: int | None = None  # functional duplicate of an earlier iteration (after the refill cap)
    refills: list = field(default_factory=list)


@dataclass(frozen=True)
class WaveContext:
    wave: int
    last_judge_score: int | None
    meta_invalid_streak: int


class FactoryOrchestrator:
    def __init__(self, max_iterations: int, llm=None, router: ModelRouter | None = None,
                 sim_timeout: int = config.SIM_TIMEOUT_S, echo: bool = True, parallel: int = config.PARALLEL_WIDTH):
        if not 1 <= parallel <= config.MAX_PARALLEL_WIDTH:
            raise ValueError(f"parallel width must be 1..{config.MAX_PARALLEL_WIDTH}")
        self.max_iterations = max_iterations
        self.sim_timeout = sim_timeout
        self.echo = echo
        self.parallel = parallel
        self._log_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self.state = read_json(config.FACTORY_STATE_PATH)
        self.space = StateSpace(read_json(config.STATE_SPACE_PATH))
        self.router = router or ModelRouter()
        self.guard = IsolationGuard(self.state["canaries"])
        self.llm = llm or ClaudeCLIClient(log=self.log)
        self.runner = AgentRunner(self.llm, self.router, self.guard, usage_cb=self._usage, log=self.log)

    # ---- plumbing -----------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with self._log_lock:
            if self.echo:
                print(line, flush=True)
            with open(config.FACTORY_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _usage(self, cost: float) -> None:
        with self._usage_lock:
            u = self.state["llm_usage"]
            u["calls"] += 1
            u["cost_usd"] = round(u["cost_usd"] + float(cost or 0.0), 6)

    def save_state(self) -> None:
        with self._usage_lock:
            write_json(config.FACTORY_STATE_PATH, self.state)

    def save_space(self) -> None:
        write_json(config.STATE_SPACE_PATH, self.space.data)

    def _snap_dir(self) -> Path:
        return config.WORKING_DIR / ".snapshots"

    # ---- preflight / resume ---------------------------------------------------------------
    def preflight(self) -> None:
        for p in (config.AI_DOCS_DIR, config.PROMPTS_DIR, config.TEMPLATES_DIR):
            if not p.is_dir():
                raise MacroFail(f"required directory missing: {p}")
        if ai_docs_checksum() != self.state["ai_docs_checksum"]:
            raise ReadOnlyViolation("ai_docs changed since initialisation (read-only directory)")
        cache_path = sim_harness.ensure_cache()
        warm = sim_harness.warm_up()  # pay the torch/cache cold start here, not inside a 60 s simulation
        base = load_cache(config.CACHE_DIR)["baselines"]
        n = len(base)
        self.state["baselines"] = {
            f"{k}_return": round(sum(b[k] for b in base) / n, 3)
            for k in ("noop", "heuristic", "oracle", "teacher_v2", "clairvoyant")
        }
        self.log(f"[preflight] simulator cache {cache_path.name}; baselines {self.state['baselines']}; "
                 f"parallel width {self.parallel}; warm-up {warm:.1f}s")

        start = self.state["next_iteration"]
        stale = sorted(d for d in config.EXPERIMENTS_DIR.glob("exp_*")
                       if d.is_dir() and re.fullmatch(r"exp_\d{3}", d.name) and int(d.name[4:]) >= start)
        if stale:  # a wave was interrupted mid-flight: archive its experiments and restore the snapshot
            stamp = time.strftime("%Y%m%d-%H%M%S")
            for d in stale:
                dest = config.EXPERIMENTS_DIR / "_aborted" / f"{d.name}_{stamp}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(dest))
            snap_space = self._snap_dir() / f"state_space.before_{start:03d}.json"
            snap_state = self._snap_dir() / f"factory_state.before_{start:03d}.json"
            if snap_space.exists() and snap_state.exists():
                usage = self.state["llm_usage"]
                baselines = self.state.get("baselines")
                self.space = StateSpace(read_json(snap_space))
                self.state = read_json(snap_state)
                self.state["llm_usage"] = usage
                self.state["baselines"] = baselines
                self.guard = IsolationGuard(self.state["canaries"])
                self.runner.guard = self.guard
                self._sync_best_files()
                self.save_space()
            self.log(f"[preflight] interrupted wave archived: {[d.name for d in stale]}")
        self.save_state()

    # ---- main loop ------------------------------------------------------------------------
    def run(self) -> int:
        if self.state.get("halted"):
            self.log(f"factory already halted ({self.state['halted']['reason']}); use --fresh for a new run")
            return 0
        self.preflight()
        self.log(f"factory start: next_iteration={self.state['next_iteration']} max_iterations={self.max_iterations}")
        while True:
            # Meta-Done is decided here, by Python, before any agent runs. No LLM field can stop the loop.
            reason = done_checker.check(self.state, self.max_iterations)
            if reason:
                self.halt(reason)
                return 0
            self.run_wave()
            self._circuit_breaker()

    def _circuit_breaker(self) -> None:
        """Pause (resumable) when two consecutive waves were poisoned entirely by the same error type.

        That pattern is a systematic fault (environment or harness), not a bad coordinate; poisoning
        further iterations would only burn the stagnation window.
        """
        its = self.state["iterations"]
        waves = sorted({r.get("wave") for r in its if r.get("wave") is not None})[-config.BREAKER_WAVES:]
        if len(waves) < config.BREAKER_WAVES:
            return
        rows = [r for r in its if r.get("wave") in waves]
        if not rows or any(r["status"] != "poisoned" for r in rows):
            return
        kinds = {f.get("error_type") for r in rows for f in r.get("failures", [])}
        if len(kinds) == 1:
            raise InfraUnavailable(
                f"circuit breaker: waves {waves} were entirely poisoned by {kinds.pop()}; pausing so the cause can "
                f"be fixed (resume with the same command)")

    def _new_member(self, i: int) -> Member:
        exp, d = config.exp_name(i), config.exp_dir(i)
        (d / "prompts").mkdir(parents=True, exist_ok=True)
        canary = new_canary(exp)
        self.state["canaries"][exp] = canary
        return Member(i=i, exp=exp, d=d, canary=canary, rlog=RouterLog(d / "router_log.json", exp))

    def run_wave(self) -> None:
        start = self.state["next_iteration"]
        width = min(self.parallel, self.max_iterations - start)
        indices = list(range(start, start + width))
        wave_no = self.state.get("waves", 0)
        snap = self._snap_dir()
        snap.mkdir(parents=True, exist_ok=True)
        write_json(snap / f"state_space.before_{start:03d}.json", self.space.data)
        write_json(snap / f"factory_state.before_{start:03d}.json", self.state)
        wctx = WaveContext(wave=wave_no, last_judge_score=self.state["last_judge_score"],
                           meta_invalid_streak=self.state["meta_invalid_streak"])
        self.log(f"=== wave {wave_no}: iterations {indices} (last judge score {wctx.last_judge_score}) ===")

        lead = self._new_member(start)
        self.save_state()

        # A. Meta-Agent (serial). One call proposes the whole wave.
        proposal = self.phase_meta(lead, indices, wctx)
        if proposal is None:
            self._poison_bookkeeping(lead)
            lead.section = self._poison_section(lead)
            self._finish_wave([lead], wave_no)
            return
        members = [lead] + [self._new_member(i) for i in indices[1:]]
        self.save_state()
        self._fan_out_meta(lead, members[1:], proposal)

        # B1. Worker + functional fingerprint, one thread per iteration.
        self._parallel(lambda m: self.member_core(m, BUILD_STAGES), members)
        # B2. Duplicate check, serial, in iteration order: a model identical to an earlier one gets its
        #     slot back through a single-coordinate Meta refill (capped per wave).
        self._dedupe(members)
        # B3. Simulator -> Judge, one thread per iteration.
        self._parallel(lambda m: self.member_core(m, CORE_STAGES), [m for m in members if m.status != "poisoned"])

        # C. State update, serial, in iteration order.
        for m in members:
            if m.status == "ready":
                self.stage_update(m)
            else:
                self._poison_bookkeeping(m)
        self._write_last_judge(members)

        # D. Documenting Agent, one thread per scored iteration (all see the same paper snapshot).
        paper_snapshot = config.RESEARCH_PAPER_PATH.read_text(encoding="utf-8")
        scored = [m for m in members if m.status == "ready"]
        self._parallel(lambda m: self.member_doc(m, paper_snapshot), scored)
        for m in members:
            if m.status == "poisoned":
                m.section = self._poison_section(m)

        # E. Paper assembly + close, serial, in iteration order.
        self._finish_wave(members, wave_no)

    def _parallel(self, fn, members: list[Member]) -> None:
        if not members:
            return
        if len(members) == 1:
            fn(members[0])
            return
        with ThreadPoolExecutor(max_workers=len(members), thread_name_prefix="factory") as ex:
            futures = [ex.submit(fn, m) for m in members]
            errors = []
            for f in futures:  # join in iteration order; re-raise the first macro/infra error after all finish
                try:
                    f.result()
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)
        if errors:
            raise errors[0]

    # ---- A: Meta-Agent --------------------------------------------------------------------
    def phase_meta(self, lead: Member, indices: list[int], wctx: WaveContext) -> dict | None:
        result: dict = {}
        esc = EscalationTracker()

        def validator(out: dict) -> None:
            new_space, coords, report = self.space.apply_wave_proposal(out, indices)
            result.update(space=new_space, coords=coords, report=report)

        def on_failure(_e: Exception) -> None:
            self.state["meta_invalid_streak"] += 1
            self.save_state()

        payload = {
            "ITERATION": str(indices[0]),
            "WAVE_SIZE": str(len(indices)),
            "STATE_SPACE": _dumps(sanitize_obj(self.space.data)),
            "BEST_CONFIG": _dumps(sanitize_obj(read_json(config.BEST_CONFIG_PATH, {}))),
            "LAST_JUDGE": _dumps(sanitize_obj(read_json(config.LAST_JUDGE_PATH, {}))),
            "REFILL_NOTE": "",
        }
        for attempt in range(config.MAX_ITERATION_ATTEMPTS):
            def ctx(step_retry: int, _attempt=attempt) -> RouteContext:
                return RouteContext(iteration=indices[0], agent="meta", attempt=_attempt, step_retry=step_retry,
                                    escalation=esc.escalation("meta"), last_judge_score=wctx.last_judge_score,
                                    meta_invalid_streak=self.state["meta_invalid_streak"])
            try:
                try:
                    out = self.runner.run(AgentCall(
                        agent="meta", purpose="primary", iteration=indices[0], experiment=lead.exp,
                        prompt_dir=lead.d / "prompts", attempt=attempt, payload=payload,
                        sources=["workspace/state_space.json", "workspace/best_config.json",
                                 "workspace/last_judge.json"],
                        ctx_factory=ctx, validator=validator, on_failure=on_failure,
                        audit_path=lead.d / "isolation_audit.json",
                    ), lead.rlog)
                except (MicroFail, MacroFail, InfraUnavailable):
                    raise
                except OSError as e:  # operating-system / process-launch failure: pause, never poison
                    raise InfraUnavailable(f"{type(e).__name__} in stage meta: {e}") from e
                except Exception as e:
                    (lead.d / f"stage_error_meta_a{attempt}.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    raise HarnessStageError(f"{type(e).__name__}: {e}", agent="meta") from e
            except MicroFail as e:
                lead.consecutive_fails += 1
                lead.failures.append({"attempt": attempt, "stage": "meta", "agent": "meta",
                                      "error_type": type(e).__name__, "error": str(e)[:4000]})
                write_json(lead.d / "attempts.json", lead.failures)
                lvl = esc.escalate("meta")
                self.log(f"[{lead.exp}] micro-fail {lead.consecutive_fails}/{config.MAX_ITERATION_ATTEMPTS} at "
                         f"stage meta ({type(e).__name__}); escalate meta -> +{lvl}: {str(e)[:300]}")
                continue
            self.state["meta_invalid_streak"] = 0
            self.space = result["space"]
            self.save_space()
            report = result["report"]
            if report["dimensions_added"]:
                self.state["last_dimension_added_iteration"] = indices[0]
            self.save_state()
            self.log(f"[wave {wctx.wave}] meta: {len(report['applied'])} mutation(s), added={report['dimensions_added']}, "
                     f"coordinates {report['coordinate_hashes']}")
            return {"out": out, "coords": result["coords"], "report": report, "indices": indices}
        lead.status = "poisoned"
        return None

    def _fan_out_meta(self, lead: Member, others: list[Member], proposal: dict) -> None:
        out, coords, report, indices = proposal["out"], proposal["coords"], proposal["report"], proposal["indices"]
        members = [lead] + others
        for k, m in enumerate(members):
            nc = {
                "iteration": m.i,
                "experiment": m.exp,
                "coordinate": coords[k],
                "coordinate_hash": report["coordinate_hashes"][k],
                "hypothesis": out["candidates"][k]["hypothesis"],
                "wave_iterations": indices,
                "wave_position": k,
                "mutations_applied": report["applied"],
                "dimensions_added": report["dimensions_added"] if k == 0 else [],
            }
            write_json(m.d / "next_coordinate.json", nc)
            m.memo["meta"] = nc
            if m is not lead:
                # The wave's single Meta call is part of every member's routing history.
                for e in lead.rlog.entries:
                    m.rlog.entries.append(copy.deepcopy(dict(e, shared_from_wave_lead=True)))
                m.rlog.flush()
        write_json(lead.d / "meta.json", out)

    # ---- B: Worker -> Simulator -> Judge (thread per member) ------------------------------
    def member_core(self, m: Member, stages: tuple = CORE_STAGES) -> None:
        wctx_score = self.state["last_judge_score"]  # read-only during phase B
        while m.consecutive_fails < config.MAX_ITERATION_ATTEMPTS:
            attempt = m.consecutive_fails
            stage = stages[0]
            try:
                for stage in stages:
                    if stage in m.memo:
                        continue
                    try:
                        if stage == "worker":
                            m.memo[stage] = self.stage_worker(m, attempt, wctx_score)
                        elif stage == "fingerprint":
                            m.memo[stage] = self.stage_fingerprint(m)
                        elif stage == "simulate":
                            m.memo[stage] = self.stage_simulate(m)
                        else:
                            m.memo[stage] = self.stage_judge(m, attempt, wctx_score)
                    except (MicroFail, MacroFail, InfraUnavailable):
                        raise
                    except OSError as e:  # operating-system / disk / process-launch failure: pause, never poison
                        raise InfraUnavailable(f"{type(e).__name__} in stage {stage}: {e}") from e
                    except Exception as e:  # harness-side surprise inside a stage: log it, keep the factory alive
                        (m.d / f"stage_error_{stage}_a{attempt}.txt").write_text(traceback.format_exc(), encoding="utf-8")
                        raise HarnessStageError(f"{type(e).__name__}: {e}", agent=STAGE_AGENT[stage]) from e
                if "judge" in stages:
                    m.status = "ready"
                return
            except MicroFail as e:
                self._record_failure(m, stage, attempt, e)
                if isinstance(e, (SimulatorError, WorkerCodeError)):
                    m.memo.pop("worker", None)
                    m.memo.pop("fingerprint", None)
                    m.retry_error = str(e)[-3500:]
        m.status = "poisoned"

    def stage_fingerprint(self, m: Member) -> dict:
        fp = sim_harness.run_fingerprint(m.d, m.canary)
        self.log(f"[{m.exp}] fingerprint {fp['fingerprint'][:12]} ({fp['seconds']:.1f}s)")
        return fp

    # ---- B2: functional duplicates ---------------------------------------------------------
    def _dedupe(self, members: list[Member]) -> None:
        refills = 0
        while True:
            seen = dict(self.state.get("fingerprints", {}))
            redo = []
            for m in members:
                fp = (m.memo.get("fingerprint") or {}).get("fingerprint")
                if m.status == "poisoned" or fp is None:
                    continue
                if fp in seen:
                    dup = int(seen[fp])
                    if refills < config.MAX_REFILLS_PER_WAVE and self._refill(m, dup, members):
                        refills += 1
                        redo.append(m)
                        continue
                    m.duplicate_of = dup
                    self.log(f"[{m.exp}] functional duplicate of iteration {dup} (refill cap reached or refill failed)")
                seen.setdefault(fp, m.i)
            if not redo:
                return
            self._parallel(lambda mm: self.member_core(mm, BUILD_STAGES), redo)

    def _refill(self, m: Member, dup: int, members: list[Member]) -> bool:
        """Ask the Meta for ONE replacement coordinate for member m (no mutations)."""
        old = m.memo["meta"]
        dup_coord = next((e["coordinate"] for e in self.space.data.get("ledger", []) if e["iteration"] == dup), None)
        if dup_coord is None:
            dup_coord = next((x.memo["meta"]["coordinate"] for x in members if x.i == dup), {})
        diff = {k: old["coordinate"].get(k) for k in old["coordinate"] if old["coordinate"].get(k) != dup_coord.get(k)}
        reserved = {x.memo["meta"]["coordinate_hash"] for x in members if "meta" in x.memo}
        reserved |= {r["coordinate_hash"] for r in m.refills}
        note = (
            "\n## Refill request (harness)\n\n"
            f"Your coordinate for iteration {m.i} produced a model that is functionally identical to iteration {dup} "
            f"(same features, architecture, initialisation, training configuration and loss). These coordinate values "
            f"had no effect: {json.dumps(diff)}. Propose exactly ONE replacement coordinate that changes the trained "
            f"model. The mutations list must be empty. Do not reuse any coordinate in the ledger or in this wave.\n"
        )
        result: dict = {}

        def validator(out: dict) -> None:
            if out.get("mutations"):
                raise AgentOutputError("refill proposals must not contain mutations")
            _, coords, report = self.space.apply_wave_proposal(out, [m.i], reserved=reserved)
            result.update(coord=coords[0], report=report, hypothesis=out["candidates"][0]["hypothesis"])

        def on_failure(_e: Exception) -> None:
            self.state["meta_invalid_streak"] += 1
            self.save_state()

        def ctx(step_retry: int) -> RouteContext:
            return RouteContext(iteration=m.i, agent="meta", attempt=len(m.refills), step_retry=step_retry,
                                escalation=0, last_judge_score=self.state["last_judge_score"],
                                meta_invalid_streak=self.state["meta_invalid_streak"])

        payload = {
            "ITERATION": str(m.i),
            "WAVE_SIZE": "1",
            "STATE_SPACE": _dumps(sanitize_obj(self.space.data)),
            "BEST_CONFIG": _dumps(sanitize_obj(read_json(config.BEST_CONFIG_PATH, {}))),
            "LAST_JUDGE": _dumps(sanitize_obj(read_json(config.LAST_JUDGE_PATH, {}))),
            "REFILL_NOTE": note,
        }
        try:
            self.runner.run(AgentCall(
                agent="meta", purpose="primary", iteration=m.i, experiment=m.exp, prompt_dir=m.d / "prompts",
                attempt=10 + len(m.refills), payload=payload,
                sources=["workspace/state_space.json", "workspace/best_config.json", "workspace/last_judge.json"],
                ctx_factory=ctx, validator=validator, on_failure=on_failure, audit_path=m.d / "isolation_audit.json",
            ), m.rlog)
        except MicroFail as e:
            self.log(f"[{m.exp}] refill failed ({type(e).__name__}); keeping the duplicate")
            return False
        self.state["meta_invalid_streak"] = 0
        k = len(m.refills) + 1
        write_json(m.d / f"next_coordinate.rejected_{k}.json", dict(old, duplicate_of=dup))
        m.refills.append({"coordinate_hash": old["coordinate_hash"], "duplicate_of": dup, "no_effect": diff})
        nc = dict(old, coordinate=result["coord"], coordinate_hash=result["report"]["coordinate_hashes"][0],
                  hypothesis=result["hypothesis"], refilled_after_duplicate_of=dup)
        write_json(m.d / "next_coordinate.json", nc)
        m.memo = {"meta": nc}
        m.retry_error = None
        self.save_state()
        self.log(f"[{m.exp}] duplicate of iteration {dup}; refilled with coordinate {nc['coordinate_hash']}")
        return True

    def _record_failure(self, m: Member, stage: str, attempt: int, e: Exception) -> None:
        m.consecutive_fails += 1
        agent = getattr(e, "agent", None) or STAGE_AGENT[stage]
        if agent not in config.CORE_AGENTS:
            agent = STAGE_AGENT[stage]
        m.failures.append({"attempt": attempt, "stage": stage, "agent": agent,
                           "error_type": type(e).__name__, "error": str(e)[:4000]})
        write_json(m.d / "attempts.json", m.failures)
        lvl = m.esc.escalate(agent)
        self.log(f"[{m.exp}] micro-fail {m.consecutive_fails}/{config.MAX_ITERATION_ATTEMPTS} at stage {stage} "
                 f"({type(e).__name__}); escalate {agent} -> +{lvl}: {str(e)[:300]}")

    def _ai_docs(self) -> tuple[str, list[str]]:
        parts, sources = [], []
        for p in sorted(config.AI_DOCS_DIR.glob("*.md")):
            parts.append(f"### ai_docs/{p.name}\n\n{p.read_text(encoding='utf-8').strip()}\n")
            sources.append(f"ai_docs/{p.name}")
        return "\n".join(parts), sources

    def _ctx(self, m: Member, agent: str, attempt: int, last_score, purpose: str = "primary"):
        def ctx(step_retry: int) -> RouteContext:
            return RouteContext(iteration=m.i, agent=agent, purpose=purpose, attempt=attempt, step_retry=step_retry,
                                escalation=m.esc.escalation(agent) if purpose != "format" else 0,
                                last_judge_score=last_score,
                                meta_invalid_streak=self.state["meta_invalid_streak"])
        return ctx

    def stage_worker(self, m: Member, attempt: int, last_score) -> dict:
        nc = m.memo["meta"]
        coord = nc["coordinate"]
        dims = {
            k: {kk: vv for kk, vv in self.space.dimensions[k].items() if kk in ("type", "values", "min", "max", "description")}
            for k in coord
        }
        gate: dict = {}

        def validator(out: dict) -> None:
            gate.clear()
            gate.update(check_model_source(out["model_py"]))

        docs, doc_sources = self._ai_docs()
        retry_block = ""
        sources = ["exp/next_coordinate.json", "exp/coordinate_dimensions", *doc_sources]
        if m.retry_error:
            retry_block = (
                "\n## Previous attempt in this experiment failed\n\n"
                f"```\n{m.retry_error}\n```\n\nFix the cause; keep honouring every coordinate value.\n"
            )
            sources.append("exp/retry_error")
        payload = {
            "COORDINATE": _dumps({"experiment": m.exp, "coordinate": coord, "coordinate_hash": nc["coordinate_hash"]}),
            "DIMENSIONS": _dumps(dims),
            "AI_DOCS": docs,
            "RETRY_ERROR": retry_block,
        }
        out = self.runner.run(AgentCall(
            agent="worker", purpose="primary", iteration=m.i, experiment=m.exp, prompt_dir=m.d / "prompts",
            attempt=attempt, payload=payload, sources=sources, ctx_factory=self._ctx(m, "worker", attempt, last_score),
            validator=validator, audit_path=m.d / "isolation_audit.json",
        ), m.rlog)

        src = out["model_py"].rstrip() + "\n"
        header = f"# {m.exp} | coordinate {nc['coordinate_hash']} | generated by the Worker Agent; executed by the simulator\n"
        write_text(m.d / "model.py", header + src)
        cfg = {
            "experiment": m.exp,
            "iteration": m.i,
            "canary": m.canary,
            "coordinate": coord,
            "coordinate_hash": nc["coordinate_hash"],
            "dimension_definitions": dims,
            "hypothesis": nc["hypothesis"],
            "worker": {
                "design_notes": out["design_notes"],
                "model_py_sha256": hashlib.sha256(src.encode()).hexdigest(),
                "static_gate": dict(gate),
            },
        }
        write_json(m.d / "config.json", cfg)
        write_json(m.d / "worker.json", {"design_notes": out["design_notes"], "attempt": attempt})
        self.log(f"[{m.exp}] worker: model.py {gate.get('lines')} lines passed the static gate")
        return {"design_notes": out["design_notes"], "gate": dict(gate)}

    def stage_simulate(self, m: Member) -> dict:
        res = sim_harness.run_simulator(m.d, m.canary, timeout_s=self.sim_timeout)
        s = res.metrics
        self.log(f"[{m.exp}] simulator: sim_score={s['sim_score']} return={s['mean_return']} "
                 f"success={s['success_rate']} val_acc={s['training']['val_acc']} ({res.seconds:.1f}s)")
        return s

    def stage_judge(self, m: Member, attempt: int, last_score) -> dict:
        sim_score = float(m.memo["simulate"]["sim_score"])
        tol = config.JUDGE_ANCHOR_TOLERANCE

        def validator(out: dict) -> None:
            if isinstance(out.get("score"), bool) or not isinstance(out.get("score"), int):
                raise AgentOutputError("score must be an integer")
            delta = abs(out["score"] - sim_score)
            if delta > tol:
                raise JudgeAnchorError(
                    f"score {out['score']} is {delta:.1f} points from the deterministic sim_score {sim_score}; "
                    f"the rubric allows at most ±{tol}"
                )

        judge_view = {k: v for k, v in read_json(m.d / "config.json").items() if k != "hypothesis"}
        payload = {
            "CONFIG": _dumps(judge_view),  # the hypothesis is withheld so it cannot bias the score
            "SIMULATION_LOG": (m.d / "simulation.log").read_text(encoding="utf-8"),
        }
        out = self.runner.run(AgentCall(
            agent="judge", purpose="primary", iteration=m.i, experiment=m.exp, prompt_dir=m.d / "prompts",
            attempt=attempt, payload=payload, sources=["exp/simulation.log", "exp/config.json"],
            ctx_factory=self._ctx(m, "judge", attempt, last_score), validator=validator,
            audit_path=m.d / "isolation_audit.json",
        ), m.rlog)
        judge = dict(out)
        judge["_harness"] = {"experiment": m.exp, "iteration": m.i, "sim_score": sim_score,
                             "anchor_delta": round(out["score"] - sim_score, 2), "anchor_tolerance": tol}
        write_json(m.d / "judge.json", judge)
        self.log(f"[{m.exp}] judge: score={out['score']} (sim_score {sim_score})")
        return judge

    # ---- C: state update (main thread, iteration order) ------------------------------------
    def stage_update(self, m: Member) -> None:
        judge, sim, nc = m.memo["judge"], m.memo["simulate"], m.memo["meta"]
        score, sim_score = int(judge["score"]), float(sim["sim_score"])
        self.state["scores"][str(m.i)] = score
        self.state["sim_scores"][str(m.i)] = sim_score
        self.state["last_judge_score"] = score
        self.space.record(m.i, nc["coordinate"], "scored", score, sim_score)
        self.save_space()
        best = self.state.get("best")
        improved = best is None or score > best["score"] or (score == best["score"] and sim_score > best["sim_score"])
        if improved:
            self.state["best"] = {"iteration": m.i, "experiment": m.exp, "score": score, "sim_score": sim_score,
                                  "coordinate_hash": nc["coordinate_hash"]}
            self._sync_best_files()
            self.log(f"[{m.exp}] new best: judge {score}, sim_score {sim_score}")
        m.memo["update"] = {"score": score, "sim_score": sim_score, "new_best": improved}
        self.save_state()

    def _write_last_judge(self, members: list[Member]) -> None:
        verdicts = []
        for m in members:
            if m.status != "ready":
                continue
            judge, sim = m.memo["judge"], m.memo["simulate"]
            verdicts.append({
                "iteration": m.i,
                "coordinate": m.memo["meta"]["coordinate"],
                "score": judge["score"],
                "sim_score": sim["sim_score"],
                "feedback": judge["feedback"],
                "next_steps": judge["next_steps"],
                "strengths": judge["strengths"],
                "weaknesses": judge["weaknesses"],
                "metrics": self._metric_digest(sim),
            })
        if verdicts:
            write_json(config.LAST_JUDGE_PATH, sanitize_obj({"previous_wave": verdicts}))

    def _poison_bookkeeping(self, m: Member) -> None:
        m.status = "poisoned"
        nc = m.memo.get("meta") or {}
        coord = nc.get("coordinate")
        if coord is not None:
            self.space.record(m.i, coord, "poisoned", None)
            self.save_space()
        self.state["poisoned"].append({
            "iteration": m.i,
            "coordinate": coord,
            "coordinate_hash": nc.get("coordinate_hash"),
            "reasons": [f"{f['stage']}/{f['error_type']}: {f['error'][:300]}" for f in m.failures],
        })
        self.save_state()
        self.log(f"[{m.exp}] POISONED after {len(m.failures)} failures")

    @staticmethod
    def _metric_digest(sim: dict) -> dict:
        t = sim["training"]
        return {
            "mean_return": sim["mean_return"],
            "success_rate": sim["success_rate"],
            "overflow_rate": sim["overflow_rate"],
            "oracle_agreement": sim["oracle_agreement"],
            "action_entropy": sim["action_entropy"],
            "action_distribution": sim["action_distribution"],
            "train_acc": t["train_acc"],
            "val_acc": t["val_acc"],
            "features": t["features"],
            "params": t["params"],
        }

    def _sync_best_files(self) -> None:
        best = self.state.get("best")
        if not best:
            config.BEST_MODEL_PATH.unlink(missing_ok=True)
            write_json(config.BEST_CONFIG_PATH, {"status": "no experiment has been scored yet"})
            return
        d = config.exp_dir(best["iteration"])
        src = (d / "model.py").read_text(encoding="utf-8")
        header = (f"# BEST MODEL | iteration {best['iteration']:03d} | judge score {best['score']} | "
                  f"sim_score {best['sim_score']} | coordinate {best['coordinate_hash']}\n")
        write_text(config.BEST_MODEL_PATH, header + src)
        cfg = read_json(d / "config.json")
        sim = read_json(d / "sim_metrics.json")
        judge = read_json(d / "judge.json")
        write_json(config.BEST_CONFIG_PATH, sanitize_obj({
            "iteration": best["iteration"],
            "score": best["score"],
            "sim_score": best["sim_score"],
            "coordinate": cfg["coordinate"],
            "design_notes": cfg["worker"]["design_notes"],
            "metrics": self._metric_digest(sim),
            "judge_feedback": judge["feedback"],
        }))

    # ---- D: Documenting Agent (thread per scored member) ------------------------------------
    def member_doc(self, m: Member, paper_snapshot: str) -> None:
        while m.consecutive_fails < config.MAX_ITERATION_ATTEMPTS:
            attempt = m.consecutive_fails
            try:
                try:
                    m.section, m.draft, fmt_status = self.stage_doc(m, attempt, paper_snapshot)
                except (MicroFail, MacroFail, InfraUnavailable):
                    raise
                except OSError as e:
                    raise InfraUnavailable(f"{type(e).__name__} in stage doc: {e}") from e
                except Exception as e:
                    (m.d / f"stage_error_doc_a{attempt}.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    raise HarnessStageError(f"{type(e).__name__}: {e}", agent="doc") from e
                write_json(m.d / "doc.json", {"draft": m.draft, "format_status": fmt_status})
                return
            except MicroFail as e:
                self._record_failure(m, "doc", attempt, e)
        # The experiment is scored; only documentation failed. Python writes the section.
        m.section = self._fallback_section(m)
        m.draft = None
        write_json(m.d / "doc.json", {"format_status": "harness fallback"})

    def stage_doc(self, m: Member, attempt: int, paper_snapshot: str) -> tuple[str, dict, str]:
        label = f"{m.i:03d}"
        last_score = self.state["last_judge_score"]

        def v_draft(out: dict) -> None:
            sec = out["iteration_section"].strip()
            if not sec.startswith(f"### Iteration {label}:"):
                raise AgentOutputError(f"iteration_section must start with '### Iteration {label}: <title>'")
            if re.search(r"(?m)^#{1,2} ", sec):
                raise AgentOutputError("iteration_section must not contain level-1 or level-2 headings")

        sim_log = (m.d / "simulation.log").read_text(encoding="utf-8")
        payload = {
            "ITERATION_LABEL": label,
            "CONFIG": (m.d / "config.json").read_text(encoding="utf-8"),
            "JUDGE": (m.d / "judge.json").read_text(encoding="utf-8"),
            "SIM_SUMMARY": sim_harness.summary_lines(sim_log),
            "PAPER": paper_snapshot,
        }
        draft = self.runner.run(AgentCall(
            agent="doc", purpose="draft", iteration=m.i, experiment=m.exp, prompt_dir=m.d / "prompts", attempt=attempt,
            payload=payload,
            sources=["exp/config.json", "exp/judge.json", "exp/simulation_summary", "workspace/research_paper.md"],
            ctx_factory=self._ctx(m, "doc", attempt, last_score, "draft"), validator=v_draft,
            audit_path=m.d / "isolation_audit.json",
        ), m.rlog)
        section = paperize(draft["iteration_section"].strip())

        # Fable 5.1: Markdown formatting only. Any content change is rejected and the draft is kept.
        def v_fmt(out: dict) -> None:
            ok, why = paper.format_preserves(section, out["formatted_markdown"])
            if not ok:
                raise AgentOutputError(f"formatting pass changed content: {why}")

        fmt_status = "formatted"
        try:
            fmt = self.runner.run(AgentCall(
                agent="doc", purpose="format", iteration=m.i, experiment=m.exp, prompt_dir=m.d / "prompts",
                attempt=attempt, payload={"SECTION": section}, sources=["exp/doc_draft"],
                ctx_factory=self._ctx(m, "doc", attempt, last_score, "format"), validator=v_fmt,
                audit_path=m.d / "isolation_audit.json",
            ), m.rlog)
            final_section = paperize(fmt["formatted_markdown"].strip())
        except (AgentOutputError, AgentCallError) as e:
            final_section = section
            fmt_status = f"draft kept (formatting rejected: {str(e)[:200]})"
            self.log(f"[{m.exp}] doc: {fmt_status}")
        self.log(f"[{m.exp}] doc: section ready ({fmt_status})")
        return final_section, draft, fmt_status

    def _fallback_section(self, m: Member) -> str:
        sim, judge, nc = m.memo["simulate"], m.memo["judge"], m.memo["meta"]
        coord = ", ".join(f"`{k}={json.dumps(v)}`" for k, v in nc["coordinate"].items())
        return (
            f"### Iteration {m.i:03d}: Harness-generated record\n\n"
            f"**Configuration.** Coordinate {coord}.\n\n"
            f"**Hypothesis and rationale.** {nc['hypothesis']}\n\n"
            f"**Results.** Judge score {judge['score']}; sim_score {sim['sim_score']}; mean return "
            f"{sim['mean_return']:+.2f}; success rate {sim['success_rate']:.3f}; overflow rate "
            f"{sim['overflow_rate']:.3f}; validation accuracy {sim['training']['val_acc']:.4f}.\n\n"
            f"**Assessment.** {judge['feedback']}\n\n"
            f"*The Documenting Agent failed on this iteration; this section was generated deterministically by "
            f"the harness.*"
        )

    def _poison_section(self, m: Member) -> str:
        nc = m.memo.get("meta") or {}
        coord = nc.get("coordinate")
        lines = [f"### Iteration {m.i:03d}: Poisoned coordinate", ""]
        if coord is not None:
            lines.append("**Configuration.** " + ", ".join(f"`{k}={json.dumps(v)}`" for k, v in coord.items()) + ".")
        else:
            lines.append("**Configuration.** No valid coordinate was proposed.")
        lines += ["", f"**Outcome.** The harness rejected this iteration after {len(m.failures)} consecutive "
                  "micro-failures and poisoned the coordinate so it is never proposed again:", ""]
        for f in m.failures:
            msg = re.sub(r"\s+", " ", f["error"])[:220]
            lines.append(f"- Attempt {f['attempt'] + 1}, stage `{f['stage']}` ({f['error_type']}): {msg}")
        lines += ["", "*This section was generated deterministically by the harness.*"]
        return "\n".join(lines)

    # ---- E: paper assembly + close (main thread, iteration order) ---------------------------
    def _row(self, m: Member) -> dict:
        sim = m.memo.get("simulate") or {}
        judge = m.memo.get("judge") or {}
        nc = m.memo.get("meta") or {}
        status = "scored" if m.status == "ready" else "poisoned"
        return {
            "iteration": m.i,
            "experiment": m.exp,
            "status": status,
            "score": judge.get("score") if status == "scored" else None,
            "sim_score": sim.get("sim_score") if status == "scored" else None,
            "mean_return": sim.get("mean_return") if status == "scored" else None,
            "coordinate_hash": nc.get("coordinate_hash"),
            "dims_added": len(nc.get("dimensions_added", [])),
            "attempts": len(m.failures) + (1 if status == "scored" else 0),
            "duplicate_of": m.duplicate_of,
            "refills": m.refills,
            "failures": [{k: f[k] for k in ("attempt", "stage", "agent", "error_type")} for f in m.failures],
        }

    def _leaderboard(self, extra: list[dict] | None = None) -> str:
        view = dict(self.state)
        view["iterations"] = list(self.state["iterations"]) + list(extra or [])
        return paper.leaderboard(view, self.space.data.get("ledger", []))

    def _status_line(self, n_closed: int) -> str:
        best = self.state.get("best")
        b = f"best judge score {best['score']} (Iteration {best['iteration']:03d})" if best else "no scored iteration yet"
        return (f"*Status: search in progress: {n_closed} iteration(s) closed; {b}. Sections between markers are "
                f"regenerated by the factory harness.*")

    def _finish_wave(self, members: list[Member], wave_no: int) -> None:
        rows = [self._row(m) for m in members]
        text = config.RESEARCH_PAPER_PATH.read_text(encoding="utf-8")
        for m in members:
            text = paper.upsert_iteration(text, m.i, m.section or self._poison_section(m))
        drafted = [m for m in members if m.draft]
        if drafted:  # the latest iteration's draft carries the freshest abstract and narrative
            latest = drafted[-1].draft
            text = paper.replace_block(text, "ABSTRACT", paperize(latest["abstract"]))
            text = paper.replace_block(text, "RESULTS_NARRATIVE", paperize(latest["results_narrative"]))
        text = paper.replace_block(text, "RESULTS_TABLE", self._leaderboard(rows))
        text = paper.replace_block(text, "APPENDIX_SPACE", paper.space_evolution(self.space.data))
        text = paper.replace_block(text, "STATUS", self._status_line(len(self.state["iterations"]) + len(rows)))
        write_text(config.RESEARCH_PAPER_PATH, text)

        fps = self.state.setdefault("fingerprints", {})
        for m in members:
            fp = (m.memo.get("fingerprint") or {}).get("fingerprint")
            if m.status == "ready" and fp and fp not in fps:
                fps[fp] = m.i
        for m, row in zip(members, rows):
            row["wave"] = wave_no
            row["router_verdict"] = m.rlog.verdict()
            row["closed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.state["iterations"].append(row)
            self.state["next_iteration"] = m.i + 1
        self.state["waves"] = wave_no + 1
        self.save_state()
        # Analytics for the next Meta call include every iteration closed so far (this wave too).
        self.space.data["analytics"] = analytics.compute(self.state, self.space.data, self.parallel)
        self.save_space()
        for m, row in zip(members, rows):
            checks = verify_run.check_experiment(m.i, self.state, self.space.data)
            write_json(m.d / "iteration_summary.json", {**row, "micro_done": checks})
            self.log(f"[{m.exp}] closed: status={row['status']} score={row['score']} "
                     f"micro_done={checks.get('micro_done')} router={row['router_verdict']}")
        first = members[0].i
        for p in self._snap_dir().glob(f"*before_{first:03d}.json"):
            p.unlink(missing_ok=True)

    # ---- halt -----------------------------------------------------------------------------
    def _factory_summary(self, reason: str) -> dict:
        its = self.state["iterations"]
        scored = [r for r in its if r["status"] == "scored"]
        best = self.state.get("best")
        return {
            "halting_reason": reason,
            "iterations_closed": len(its),
            "waves": self.state.get("waves", 0),
            "parallel_width": self.parallel,
            "scored": len(scored),
            "poisoned": len(its) - len(scored),
            "scores_by_iteration": {f"Iteration {r['iteration']:03d}": r["score"] for r in scored},
            "sim_scores_by_iteration": {f"Iteration {r['iteration']:03d}": r["sim_score"] for r in scored},
            "best": None if not best else {
                "iteration": f"Iteration {best['iteration']:03d}", "judge_score": best["score"],
                "sim_score": best["sim_score"],
                "coordinate": next((e["coordinate"] for e in self.space.data.get("ledger", [])
                                    if e["iteration"] == best["iteration"]), None),
            },
            "baselines": self.state.get("baselines"),
            "active_dimensions": list(self.space.dimensions),
            "pruned_dimensions": list(self.space.data.get("pruned_dimensions", {})),
            "mutations_applied": len(self.space.data.get("mutation_log", [])),
            "llm_calls": self.state["llm_usage"]["calls"],
        }

    def _confirm_top(self) -> dict | None:
        """Re-score the top iterations (by judge score) on the hidden confirmation set.

        The spec's selection rule is untouched (best_model.py = highest judge score). This only measures
        how much of the selected model's lead survives on scenarios nobody selected on, and names the
        confirmed winner, so the reported result is free of the evaluation set's winner's curse.
        """
        k = config.CONFIRM_TOP_K
        scored = [r for r in self.state["iterations"] if r["status"] == "scored" and r.get("duplicate_of") is None]
        if k <= 0 or not scored:
            return None
        # Candidates: every iteration statistically tied with the selected best on the evaluation set
        # (integer judge scores alone would drop near-ties), always including the selected best.
        se = (self.space.data.get("analytics") or {}).get("typical_paired_se_sim") or 0.6
        best_it = (self.state.get("best") or {}).get("iteration")
        best_sim = max(r["sim_score"] for r in scored)
        tied = [r for r in scored if r["sim_score"] >= best_sim - 2 * se or r["iteration"] == best_it]
        top = sorted(tied, key=lambda r: (r["iteration"] != best_it, -r["sim_score"], r["iteration"]))[:k]
        self.log(f"[confirm] re-scoring iterations {[r['iteration'] for r in top]} on the hidden confirmation set")

        def one(r):
            try:
                return r, sim_harness.run_confirmation(config.exp_dir(r["iteration"])), None
            except (SimulatorError, OSError) as e:
                return r, None, str(e)[:300]

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="confirm") as ex:
            results = list(ex.map(one, top))
        selected = (self.state.get("best") or {}).get("iteration")
        ref = next((m for r, m, _e in results if r["iteration"] == selected and m), None)
        rows = []
        for r, m, err in results:
            row = {"iteration": r["iteration"], "judge": r["score"], "eval_sim": r["sim_score"]}
            if m is None:
                row["error"] = err
            else:
                d, se = (0.0, 0.0)
                if ref is not None:
                    diffs = [a - b for a, b in zip(m["turn_returns"], ref["turn_returns"])]
                    d = sum(diffs) / len(diffs)
                    se = (sum((x - d) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs)) ** 0.5 if any(diffs) else 0.0
                row.update(confirm_sim=m["sim_score"], confirm_return=m["mean_return"],
                           delta_vs_selected=round(d, 3), se=round(se, 3))
            rows.append(row)
        ok = [r for r in rows if "error" not in r]
        confirmed = max(ok, key=lambda r: (r["confirm_return"], -r["iteration"]))["iteration"] if ok else None
        note = (f"The selected best (highest judge score) is Iteration {selected:03d}; the confirmed best on the hidden "
                f"set is Iteration {confirmed:03d}." if confirmed is not None and selected is not None else "")
        conf = {"set": "200 hidden scenarios (seeds 95000+), never used for selection", "rows": rows,
                "selected_best": selected, "confirmed_best": confirmed, "note": note}
        write_json(config.WORKING_DIR / "confirmation.json", conf)
        if confirmed is not None:
            src = (config.exp_dir(confirmed) / "model.py").read_text(encoding="utf-8")
            r = next(x for x in rows if x["iteration"] == confirmed)
            header = (f"# CONFIRMED MODEL | iteration {confirmed:03d} | confirmation sim_score {r['confirm_sim']} | "
                      f"evaluation sim_score {r['eval_sim']} | selected best (spec) is iteration {selected:03d}\n")
            write_text(config.WORKING_DIR / "best_model_confirmed.py", header + src)
        self.log(f"[confirm] selected best {selected}, confirmed best {confirmed}: "
                 + "; ".join(f"{r['iteration']}: eval {r['eval_sim']} -> confirm {r.get('confirm_sim', 'failed')}" for r in rows))
        return conf

    def halt(self, reason: str) -> None:
        self.log(f"HALT (Meta-Done, decided by Python): {reason}")
        n = self.state["next_iteration"]
        conf = self._confirm_top()
        summary = self._factory_summary(reason)
        if conf:
            summary["hidden_confirmation"] = {
                "note": conf["note"],
                "rows": [{("Iteration" if k == "iteration" else k): (f"{v:03d}" if k == "iteration" else v)
                          for k, v in r.items()} for r in conf["rows"]],
            }
        final_dir = config.WORKING_DIR / "final_synthesis"
        final_dir.mkdir(parents=True, exist_ok=True)
        rlog = RouterLog(final_dir / "router_log.json", "final_synthesis")
        text = config.RESEARCH_PAPER_PATH.read_text(encoding="utf-8")
        try:
            out = self.runner.run(AgentCall(
                agent="doc", purpose="final", iteration=n, experiment=None, prompt_dir=final_dir, attempt=0,
                payload={"FACTORY_SUMMARY": _dumps(summary), "PAPER": text},
                sources=["workspace/research_paper.md", "workspace/factory_summary"],
                ctx_factory=lambda r: RouteContext(iteration=n, agent="doc", purpose="final", step_retry=r),
                audit_path=final_dir / "isolation_audit.json",
            ), rlog)
        except (AgentCallError, AgentOutputError) as e:
            self.log(f"final synthesis failed ({e}); writing deterministic closing sections")
            best = summary["best"]
            btxt = f"The best configuration ({best['iteration']}) reached a judge score of {best['judge_score']}." if best else "No iteration was scored."
            out = {
                "abstract": get_block_safe(text, "ABSTRACT"),
                "results_narrative": get_block_safe(text, "RESULTS_NARRATIVE"),
                "discussion": f"The search closed {summary['iterations_closed']} iterations ({summary['scored']} scored, "
                              f"{summary['poisoned']} poisoned). {btxt}",
                "conclusion": f"The harness halted: {reason}. {btxt}",
            }
        text = paper.replace_block(text, "ABSTRACT", paperize(out["abstract"]))
        text = paper.replace_block(text, "RESULTS_NARRATIVE", paperize(out["results_narrative"]))
        text = paper.replace_block(text, "DISCUSSION", paperize(out["discussion"]))
        text = paper.replace_block(text, "CONCLUSION", paperize(out["conclusion"]))
        text = paper.replace_block(text, "RESULTS_TABLE", self._leaderboard())
        conf_text = (paper.confirmation_table(conf) if conf else
                     "*No hidden confirmation was run (disabled, or no scored iteration).*")
        try:
            text = paper.replace_block(text, "CONFIRMATION", conf_text)
        except ValueError:  # a paper started from an older template has no confirmation block
            pass
        text = paper.replace_block(text, "APPENDIX_SPACE", paper.space_evolution(self.space.data))
        best = self.state.get("best")
        btxt = f"best judge score {best['score']} (Iteration {best['iteration']:03d})" if best else "no scored iteration"
        text = paper.replace_block(text, "STATUS", (
            f"*Status: complete. The search halted after {n} iteration(s) in {self.state.get('waves', 0)} parallel "
            f"wave(s); halting rule: {paperize(reason)}. Final result: {btxt}. Generated by the Context Health "
            f"Factory harness.*"))
        write_text(config.RESEARCH_PAPER_PATH, text)

        if ai_docs_checksum() != self.state["ai_docs_checksum"]:
            raise ReadOnlyViolation("ai_docs changed during the run (read-only directory)")
        self.state["halted"] = {"reason": reason, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "iterations": n}
        self.save_state()
        report = verify_run.build_report()
        write_json(config.FACTORY_REPORT_PATH, report)
        self.log(f"factory report: macro_done={report['macro_done']['all_passed']} "
                 f"micro_done={report['micro_done_summary']}")


def get_block_safe(text: str, name: str) -> str:
    try:
        return paper.get_block(text, name).strip() or "*Not available.*"
    except ValueError:
        return "*Not available.*"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Context Health MLP Cognitive Software Factory (deterministic harness)")
    ap.add_argument("--max-iterations", type=int, default=50)
    ap.add_argument("--parallel", type=int, default=config.PARALLEL_WIDTH,
                    help=f"iterations per synchronous wave (1..{config.MAX_PARALLEL_WIDTH})")
    ap.add_argument("--fresh", action="store_true", help="archive any previous run and start from the seed")
    ap.add_argument("--sim-timeout", type=int, default=config.SIM_TIMEOUT_S)
    args = ap.parse_args(argv)
    if args.max_iterations < 1:
        ap.error("--max-iterations must be >= 1")
    if not 1 <= args.parallel <= config.MAX_PARALLEL_WIDTH:
        ap.error(f"--parallel must be 1..{config.MAX_PARALLEL_WIDTH}")

    if args.fresh or not is_initialized():
        initialize(fresh=args.fresh)
    orch = FactoryOrchestrator(max_iterations=args.max_iterations, sim_timeout=args.sim_timeout,
                               parallel=args.parallel)
    try:
        return orch.run()
    except MacroFail as e:
        orch.log(f"MACRO-FAIL ({type(e).__name__}): {e}")
        write_json(config.WORKING_DIR / "macro_fail.json", {"type": type(e).__name__, "error": str(e),
                                                              "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return 2
    except InfraUnavailable as e:
        orch.save_state()
        orch.log(f"PAUSED: LLM infrastructure unavailable ({e}). State is resumable; rerun the same command.")
        return 3
    except KeyboardInterrupt:
        orch.save_state()
        orch.log("PAUSED by operator (KeyboardInterrupt). State is resumable; rerun the same command.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
