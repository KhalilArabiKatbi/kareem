"""Agent wrappers: whitelisted payload -> rendered prompt -> isolation check -> routed call ->
JSON extraction -> schema + semantic validation, with one in-place retry (rule 2).

The runner never loops beyond STEP_RETRIES and never decides anything about the iteration; it
raises to the orchestrator, which owns the retry/poison state machine.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import jsonschema

from . import config, schemas
from .errors import AgentCallError, AgentOutputError, RoutingViolationError, WorkerCodeError
from .isolation import IsolationGuard, scrub_foreign
from .router import ModelRouter, RouteContext, RouterLog
from .router_audit import audit
from .state_store import write_json

PROMPT_FILES = {
    ("meta", "primary"): ("meta.system.md", "meta.user.md"),
    ("worker", "primary"): ("worker.system.md", "worker.user.md"),
    ("judge", "primary"): ("judge.system.md", "judge.user.md"),
    ("doc", "draft"): ("doc.system.md", "doc_draft.user.md"),
    ("doc", "format"): ("doc_format.system.md", "doc_format.user.md"),
    ("doc", "final"): ("doc.system.md", "doc_final.user.md"),
}
GUARD_KEY = {("doc", "final"): "doc_final"}
PLACEHOLDER_RE = re.compile(r"\{\{([A-Z_]+)\}\}")


@dataclass
class AgentCall:
    agent: str
    purpose: str
    iteration: int
    experiment: str | None           # current experiment id, None for the final synthesis
    prompt_dir: Path                 # where the exact prompts are archived
    attempt: int
    payload: dict[str, str]
    sources: list[str]
    ctx_factory: Callable[[int], RouteContext]
    validator: Callable[[dict], None] | None = None
    on_failure: Callable[[Exception], None] | None = None
    audit_path: Path | None = None
    meta: dict = field(default_factory=dict)


def render(template: str, payload: dict[str, str]) -> str:
    needed = set(PLACEHOLDER_RE.findall(template))
    missing = needed - set(payload)
    if missing:
        raise KeyError(f"prompt placeholders without payload: {sorted(missing)}")
    return PLACEHOLDER_RE.sub(lambda m: payload[m.group(1)], template)


def extract_json(structured, text: str) -> dict:
    if isinstance(structured, dict):
        return structured
    t = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if fence:
        t = fence.group(1)
    elif not t.startswith("{"):
        a, b = t.find("{"), t.rfind("}")
        if a == -1 or b <= a:
            raise AgentOutputError("invalid JSON: no object found in the response")
        t = t[a: b + 1]
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        raise AgentOutputError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise AgentOutputError("invalid JSON: top-level value is not an object")
    return obj


class AgentRunner:
    def __init__(self, llm, router: ModelRouter, guard: IsolationGuard, usage_cb=None, log=print):
        self.llm = llm
        self.router = router
        self.guard = guard
        self.usage_cb = usage_cb or (lambda cost: None)
        self.log = log

    def run(self, call: AgentCall, router_log: RouterLog) -> dict:
        system_file, user_file = PROMPT_FILES[(call.agent, call.purpose)]
        system = (config.PROMPTS_DIR / system_file).read_text(encoding="utf-8").strip()
        base_prompt = render((config.PROMPTS_DIR / user_file).read_text(encoding="utf-8"), call.payload)
        schema = schemas.BY_NAME[(call.agent, call.purpose)]
        guard_key = GUARD_KEY.get((call.agent, call.purpose), call.agent)
        call.prompt_dir.mkdir(parents=True, exist_ok=True)

        last_err: Exception | None = None
        for step_retry in range(config.STEP_RETRIES + 1):
            ctx = call.ctx_factory(step_retry)
            try:
                decision = self.router.route(ctx)
            except RoutingViolationError as e:
                router_log.record(ctx, None, {"match": False, "blocked": True}, "routing_violation", str(e))
                raise
            prompt = base_prompt
            if step_retry and last_err is not None:
                prompt += (
                    "\n\n## Harness rejection of your previous answer\n\n"
                    f"```\n{str(last_err)[:3000]}\n```\n\n"
                    "Return a corrected JSON object that satisfies every rule above.\n"
                )
            iso = self.guard.check(guard_key, call.experiment, call.sources, system + "\n" + prompt)
            tag = f"{call.agent}_{call.purpose}_a{call.attempt}_r{step_retry}"
            (call.prompt_dir / f"{tag}.md").write_text(
                f"<!-- route: {decision.model}/{decision.effort} rules={list(decision.rules)} -->\n"
                f"<!-- system -->\n{system}\n<!-- user -->\n{prompt}",
                encoding="utf-8",
            )
            if call.audit_path is not None:
                entries = json.loads(call.audit_path.read_text(encoding="utf-8")) if call.audit_path.exists() else []
                entries.append({"call": tag, **iso})
                write_json(call.audit_path, entries)

            t0 = time.time()
            cost = None
            try:
                self.log(f"[{call.experiment or 'final'}] {call.agent}/{call.purpose} attempt={call.attempt} "
                         f"retry={step_retry} -> {decision.model}/{decision.effort} {list(decision.rules)}")
                resp = self.llm.call(decision, system, prompt, schema)
                cost = resp.cost_usd
                self.usage_cb(resp.cost_usd or 0.0)
                (call.prompt_dir / f"{tag}.response.json").write_text(
                    json.dumps({"structured": resp.structured, "text": resp.text, "cost_usd": resp.cost_usd,
                                "duration_s": resp.duration_s, "models": resp.model_reported}, indent=2),
                    encoding="utf-8",
                )
                out = scrub_foreign(extract_json(resp.structured, resp.text), call.experiment)
                try:
                    jsonschema.validate(out, schema)
                except jsonschema.ValidationError as e:
                    raise AgentOutputError(f"schema violation at {list(e.absolute_path)}: {e.message}") from e
                if call.validator is not None:
                    call.validator(out)
            except (AgentCallError, AgentOutputError, WorkerCodeError) as e:
                e.agent = call.agent
                router_log.record(ctx, decision, audit(ctx, decision), "failed", str(e)[:2000],
                                  round(time.time() - t0, 2), cost)
                self.log(f"[{call.experiment or 'final'}] {call.agent}/{call.purpose} rejected: {str(e)[:300]}")
                if call.on_failure is not None:
                    call.on_failure(e)
                last_err = e
                continue
            router_log.record(ctx, decision, audit(ctx, decision), "ok", None, round(time.time() - t0, 2), cost)
            return out
        assert last_err is not None
        raise last_err
