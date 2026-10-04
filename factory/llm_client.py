"""LLM transport: headless Claude Code CLI with every tool, hook, plugin and setting disabled.

The client is a dumb pipe. It receives a RouteDecision (already chosen by the router), re-checks
the Fable constraint, runs one call, and returns the parsed envelope. It never retries on its own
except for infrastructure errors (overload / rate limit), on a fixed backoff schedule.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import config
from .errors import AgentCallError, InfraUnavailable
from .router import RouteDecision, enforce_fable_constraint

INFRA_MARKERS = ("overloaded", "rate limit", "rate_limit", "timeout while connecting", "econnreset",
                 "etimedout", "fetch failed", "socket hang up")
INFRA_STATUSES = {408, 429, 500, 502, 503, 504, 529}
QUOTA_MARKERS = ("usage limit", "credit balance is too low", "out of credits")


@dataclass
class LLMResponse:
    text: str
    structured: dict | None
    cost_usd: float
    duration_s: float
    model_reported: list[str]


def resolve_claude_bin() -> str:
    """Prefer the native claude.exe behind the npm .cmd shim so args never pass through cmd.exe."""
    found = shutil.which(config.CLAUDE_BIN) or config.CLAUDE_BIN
    p = Path(found)
    if p.suffix.lower() in (".cmd", ".bat", ".ps1", ""):
        native = p.parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if native.exists():
            return str(native)
    return found


class ClaudeCLIClient:
    def __init__(self, log=print):
        self.bin = resolve_claude_bin()
        self.log = log

    def call(self, decision: RouteDecision, system_prompt: str, user_prompt: str, json_schema: dict | None) -> LLMResponse:
        enforce_fable_constraint(decision.agent, decision.purpose, decision.model)  # second guard
        sandbox = config.SANDBOX_DIR / f"{decision.agent}-{uuid.uuid4().hex[:10]}"
        sandbox.mkdir(parents=True, exist_ok=True)
        cmd = [
            "<bin>", "-p",
            "--model", decision.model_id,
            "--effort", decision.cli_effort,
            "--tools", "",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--system-prompt", system_prompt,
            "--output-format", "json",
        ]
        if json_schema is not None:
            cmd += ["--json-schema", json.dumps(json_schema, separators=(",", ":"))]
        try:
            for wait in (*config.INFRA_BACKOFF_S, None):
                t0 = time.time()
                # Re-resolve every attempt: the CLI can be replaced on disk (e.g. by an auto-update).
                self.bin = resolve_claude_bin()
                cmd[0] = self.bin
                try:
                    proc = subprocess.run(
                        cmd, input=user_prompt, capture_output=True, text=True, encoding="utf-8",
                        errors="replace", cwd=sandbox, timeout=config.LLM_CALL_TIMEOUT_S,
                    )
                except subprocess.TimeoutExpired as e:
                    raise AgentCallError(f"{decision.agent} call timed out after {config.LLM_CALL_TIMEOUT_S}s",
                                         agent=decision.agent) from e
                except OSError as e:
                    # The CLI binary could not be launched at all: an infrastructure problem, never an
                    # agent failure. Back off on the fixed schedule, then pause the factory.
                    if wait is None:
                        raise InfraUnavailable(f"claude CLI cannot be launched ({self.bin}): {e}") from e
                    self.log(f"[llm] cannot launch claude CLI for {decision.agent} ({e}); backing off {wait}s")
                    time.sleep(wait)
                    continue
                dur = time.time() - t0
                out = (proc.stdout or "").strip()
                err = (proc.stderr or "").strip()
                envelope = None
                try:
                    envelope = json.loads(out) if out else None
                except json.JSONDecodeError:
                    envelope = None

                if envelope is not None and not envelope.get("is_error") and envelope.get("subtype") == "success":
                    return LLMResponse(
                        text=envelope.get("result") or "",
                        structured=envelope.get("structured_output"),
                        cost_usd=float(envelope.get("total_cost_usd") or 0.0),
                        duration_s=round(dur, 2),
                        model_reported=sorted((envelope.get("modelUsage") or {}).keys()),
                    )

                if envelope is not None:
                    blob = f"{envelope.get('result') or ''}\n{err}".lower()
                    infra = envelope.get("api_error_status") in INFRA_STATUSES or any(m in blob for m in INFRA_MARKERS)
                else:
                    blob = f"{out}\n{err}".lower()
                    infra = any(m in blob for m in INFRA_MARKERS)
                if any(m in blob for m in QUOTA_MARKERS):
                    raise InfraUnavailable(f"LLM quota exhausted: {(out or err)[-400:]}")
                if infra and wait is not None:
                    self.log(f"[llm] infra error for {decision.agent}; backing off {wait}s: {(out or err)[-200:]}")
                    time.sleep(wait)
                    continue
                if infra:
                    raise InfraUnavailable(f"LLM unavailable after backoff: {(out or err)[-400:]}")
                detail = (envelope or {}).get("subtype") or (out or err)[-600:]
                raise AgentCallError(f"{decision.agent} call failed (exit {proc.returncode}): {detail}",
                                     agent=decision.agent)
            raise InfraUnavailable("LLM unavailable after backoff")  # unreachable guard
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)
