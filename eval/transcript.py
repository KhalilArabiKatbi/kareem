"""Parse a Claude Code .jsonl transcript into steps (one per tool_result) plus a condensed
text rendering for an LLM judge. Also estimates redundancy/stale shares the hook cannot see."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

STALE_AFTER = 25  # steps before a never-referenced result counts as stale

_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_\-]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9\-]{10,}|"
    r"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}|"
    r"(?i:(?:api[_-]?key|secret|token|password|passwd)\s*[=:]\s*)[^\s\"',}]{6,})")


def redact(s: str) -> str:
    return _SECRET.sub("[REDACTED]", s)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else
                         (json.dumps(b.get("content", "")) if isinstance(b, dict) and b.get("type") == "tool_result" else "")
                         for b in content)
    return ""


def _key(name: str, inp: dict) -> str:
    for k in ("file_path", "path", "notebook_path"):
        if inp.get(k):
            return f"{name}:{inp[k]}"
    sig = inp.get("command") or inp.get("pattern") or inp.get("url") or inp.get("query") or json.dumps(inp, sort_keys=True)[:120]
    return f"{name}:{sig}"


def parse(path: Path) -> list[dict]:
    """Returns events in order. Each event: {kind, ...}. kinds: prompt, assistant, tool_use,
    tool_result(step=i), compact."""
    events, uses, usage_tokens, step = [], {}, 0, 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("isSidechain"):
            continue
        t = r.get("type")
        if t == "system" and "compact" in str(r.get("subtype", "")):
            events.append({"kind": "compact"})
        elif t == "assistant":
            m = r.get("message") or {}
            u = m.get("usage")
            if u:
                usage_tokens = int(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                                   + u.get("cache_creation_input_tokens", 0))
            for b in m.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text", "").strip():
                    events.append({"kind": "assistant", "text": b["text"]})
                elif b.get("type") == "tool_use":
                    inp = b.get("input") or {}
                    uses[b["id"]] = (b.get("name", "?"), inp)
                    events.append({"kind": "tool_use", "name": b.get("name", "?"), "input": inp})
        elif t == "user":
            content = (r.get("message") or {}).get("content")
            if isinstance(content, str) and content.strip():
                events.append({"kind": "prompt", "text": content})
            elif isinstance(content, list):
                res = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                if res:
                    name, inp = uses.get(res[0].get("tool_use_id"), ("?", {}))
                    txt = "\n".join(_text([b]) if False else _text(b.get("content", "")) for b in res)
                    events.append({"kind": "tool_result", "step": step, "tokens": usage_tokens, "ts": r.get("timestamp"),
                                   "uuid": r.get("uuid"), "chars": sum(len(json.dumps(b.get("content", ""))) for b in res),
                                   "error": any(b.get("is_error") for b in res), "name": name, "key": _key(name, inp),
                                   "text": txt})
                    step += 1
                else:
                    t2 = _text(content)
                    if t2.strip():
                        events.append({"kind": "prompt", "text": t2})
    return events


def steps_with_estimates(events: list[dict]) -> list[dict]:
    """Per-step dicts with tokens/error/chars plus redundancy_est and stale_est in [0,1]."""
    out, live, seen = [], [], set()  # live: [step, chars, key, referenced]
    since_use_text = []
    for ev in events:
        k = ev["kind"]
        if k == "compact":
            live.clear(); seen.clear()
        elif k == "tool_use":
            blob = json.dumps(ev["input"])
            for item in live:
                if not item[3] and item[4] and item[4] in blob:
                    item[3] = True
        elif k == "tool_result":
            dup = ev["key"] in seen
            seen.add(ev["key"])
            base = os.path.basename(ev["key"].split(":", 1)[1]) if ev["name"] in ("Read", "Edit", "Write", "Grep", "Glob") else ""
            live.append([ev["step"], ev["chars"], ev["key"], False, base, dup])
            total = sum(i[1] for i in live) or 1
            red = sum(i[1] for i in live if i[5]) / total
            stale = sum(i[1] for i in live if not i[3] and ev["step"] - i[0] >= STALE_AFTER) / total
            out.append({"step": ev["step"], "ts": ev["ts"], "uuid": ev["uuid"], "tokens": ev["tokens"], "error": ev["error"],
                        "chars": ev["chars"], "compact_before": False, "redundancy_est": round(red, 4),
                        "stale_est": round(stale, 4)})
    # compact_before flag
    ci = 0
    seen_compact = False
    for ev in events:
        if ev["kind"] == "compact":
            seen_compact = True
        elif ev["kind"] == "tool_result":
            out[ci]["compact_before"] = seen_compact
            seen_compact = False
            ci += 1
    return out


def render(events: list[dict], max_result=240, max_text=500, max_prompt=1500) -> str:
    """Condensed transcript with [STEP n | tokens=..] markers after each tool result."""
    lines = []
    for ev in events:
        k = ev["kind"]
        if k == "prompt":
            lines.append(f"USER: {redact(ev['text'])[:max_prompt]}")
        elif k == "assistant":
            lines.append(f"ASSISTANT: {redact(ev['text'])[:max_text]}")
        elif k == "tool_use":
            lines.append(f"TOOL_CALL {ev['name']}: {redact(json.dumps(ev['input']))[:200]}")
        elif k == "compact":
            lines.append("--- /compact happened here (context reset to summary) ---")
        elif k == "tool_result":
            flag = "ERROR " if ev["error"] else ""
            lines.append(f"RESULT {flag}({ev['chars']} chars): {redact(ev['text'])[:max_result]!r}")
            lines.append(f"[STEP {ev['step']} | context_tokens={ev['tokens']}]")
    return "\n".join(lines)
