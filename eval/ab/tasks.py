"""Synthetic multi-turn coding task for the context-management A/B test.

One task = a small repo with 10 buggy modules, a verbose check tool (bulky tool output fills the
context), and a 12-turn script. Setup facts that exist only in turn 1 (an error-message convention
and six constants) are checked by HIDDEN tests, so a context policy that loses early information
scores lower.

make_task(seed, root) writes the repo and returns {"turns": [...], "instructions": str, "hidden": Path}.
score(root, hidden) runs the hidden tests and returns per-category pass counts.
"""
from __future__ import annotations

import random
import re
import subprocess
import sys
import textwrap
from pathlib import Path

N_MODULES = 10
CONSTS = {"SCALE": 13, "RETRY_LIMIT": 7, "MAX_ITEMS": 42, "OFFSET": 5, "FACTOR": 9, "TAG": "zq"}
# module -> (constant, description of correct behaviour, buggy body, hidden asserts)
RECALL = {
    3: ("SCALE", "x * SCALE", "x", [(2, 26), (5, 65)]),
    5: ("RETRY_LIMIT", "min(x, RETRY_LIMIT)", "x", [(100, 7), (3, 3)]),
    7: ("MAX_ITEMS", "min(x, MAX_ITEMS)", "x", [(1000, 42), (10, 10)]),
    8: ("OFFSET", "x + OFFSET", "x", [(1, 6), (10, 15)]),
    9: ("FACTOR", "x * FACTOR + 1", "x * 2 + 1", [(2, 19), (0, 1)]),
    10: ("TAG", 'f"{TAG}-{x}"', 'f"{x}"', [(7, "zq-7"), (0, "zq-0")]),
}

INSTRUCTIONS = f"""You are fixing a small Python repo in the current directory, one module at a time over many messages.

Rules that apply to ALL your work in this task (I will not repeat them):
1. Every ValueError you raise or modify must have a message starting with "ERR[<module>]: " where <module> is the module file name without .py, for example "ERR[m3]: negative input".
2. Setup constants (not stored in any file, remember them exactly): SCALE = {CONSTS['SCALE']}, RETRY_LIMIT = {CONSTS['RETRY_LIMIT']}, MAX_ITEMS = {CONSTS['MAX_ITEMS']}, OFFSET = {CONSTS['OFFSET']}, FACTOR = {CONSTS['FACTOR']}, TAG = "{CONSTS['TAG']}".
3. Keep fixes minimal. Do not edit anything under tests/ or tools/.
Run `python tools/check.py` to verify a fix; it prints a lot of output, the verdict is on its last lines."""


def _module(k: int, rnd: random.Random) -> tuple[str, str, list[tuple]]:
    """(source_with_bug, visible_test, hidden_asserts[(arg, expected)])"""
    if k in RECALL:
        const, correct, buggy, asserts = RECALL[k]
        src = (f'def f{k}(x):\n    """Return {correct} (the constant is a setup constant from the task rules)."""\n'
               f'    if x is None or x < 0:\n        raise ValueError("bad input")\n    return {buggy}  # BUG\n')
        vis = f"from proj.m{k} import f{k}\n\ndef test_runs():\n    assert f{k}(1) is not None\n"
        return src, vis, asserts
    a, b = rnd.randint(2, 9), rnd.randint(2, 9)
    src = (f'def f{k}(x):\n    """Return x * {a} - {b}."""\n    if x is None or x < 0:\n        raise ValueError("bad input")\n'
           f'    return x * {a} + {b}  # BUG: should subtract {b}\n')
    vis = f"from proj.m{k} import f{k}\n\ndef test_basic():\n    assert f{k}(5) == {5 * a - b}\n\ndef test_zero():\n    assert f{k}(0) == {-b}\n"
    return src, vis, [(7, 7 * a - b)]


def _check_tool() -> str:
    return textwrap.dedent('''
        """Verbose project check. The verdict (pytest summary) is on the LAST lines."""
        import random, subprocess, sys
        rnd = random.Random(1234)
        words = "alloc retry flush cache shard index vector queue lease token commit rebase probe sweep".split()
        for i in range(500):  # ~55k chars of low-value diagnostic noise
            print(f"[diag {i:04d}] " + " ".join(rnd.choice(words) for _ in range(14)) + f" id={rnd.randrange(10**8):08d}")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "tests"], capture_output=True, text=True)
        print(r.stdout[-3000:])
        print("CHECK:", "PASS" if r.returncode == 0 else "FAIL")
    ''').lstrip()


def make_task(seed: int, root: Path) -> dict:
    rnd = random.Random(seed)
    for d in ("proj", "tests", "tools", "data"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "proj" / "__init__.py").write_text("")
    hidden = ["import sys, pytest\nsys.path.insert(0, '.')\n"]
    for k in range(1, N_MODULES + 1):
        src, vis, asserts = _module(k, rnd)
        (root / "proj" / f"m{k}.py").write_text(src)
        (root / "tests" / f"test_m{k}.py").write_text(vis)
        body = "".join(f"    assert f{k}({a!r}) == {e!r}\n" for a, e in asserts)
        hidden.append(
            f"from proj.m{k} import f{k}\n"
            f"def test_behavior_m{k}():\n{body}"
            f"def test_convention_m{k}():\n"
            f"    with pytest.raises(ValueError) as e:\n        f{k}(-1)\n"
            f"    assert str(e.value).startswith('ERR[m{k}]: ')\n")
    (root / "tests" / "conftest.py").write_text("import sys\nsys.path.insert(0, '.')\n")
    (root / "tools" / "check.py").write_text(_check_tool())
    words = "alloc retry flush cache shard index vector queue lease token commit rebase probe sweep merge split".split()
    for k in range(1, N_MODULES + 1):  # background logs the agent is asked to read: the bulk that fills the context
        lines = [f"[ctx{k} {i:04d}] " + " ".join(rnd.choice(words) for _ in range(9)) + f" id={rnd.randrange(10**7):07d}"
                 for i in range(420)]
        (root / "data" / f"ctx_{k}.log").write_text("\n".join(lines))
    hid_path = root.parent / f"{root.name}_hidden.py"
    hid_path.write_text("\n".join(hidden))
    plain = [k for k in range(1, N_MODULES + 1) if k not in RECALL]
    def bg(k: int) -> str:
        return f" Before you start, read data/ctx_{k}.log in full (background material for the team; do not summarise it)."

    turns = [INSTRUCTIONS + "\n\nStart with m1: tests/test_m1.py fails." + bg(1)
             + " Fix proj/m1.py, run `python tools/check.py`, and report in one sentence."]
    for k in range(2, N_MODULES + 1):
        if k in RECALL:
            c = RECALL[k][0]
            turns.append(f"Next: module m{k} (proj/m{k}.py) is wrong." + bg(k) + f" It should use the {c} setup constant from the rules I gave you at the start. Fix it, run `python tools/check.py`, report in one sentence.")
        else:
            turns.append(f"Next: module m{k} (proj/m{k}.py, tests/test_m{k}.py) is wrong." + bg(k) + " Fix it, run `python tools/check.py`, report in one sentence.")
    turns.append("Re-run `python tools/check.py` once more to confirm the whole repo is green, and report in one sentence.")
    turns.append("Final audit: confirm that every module's ValueError messages follow the ERR[<module>]: rule from the start of this task, fix any that do not, run `python tools/check.py`, and finish.")
    return {"turns": turns, "instructions": INSTRUCTIONS, "hidden": hid_path, "plain": plain}


def score(root: Path, hidden: Path) -> dict:
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "-rA", str(hidden)],
                         cwd=root, capture_output=True, text=True, timeout=180).stdout
    res = {"behavior": 0, "convention": 0, "recall": 0}
    for line in out.splitlines():
        if not line.startswith("PASSED"):
            continue
        name = line.split("::")[-1]
        m = re.match(r"test_(behavior|convention)_m(\d+)", name)
        if not m:
            continue
        kind, k = m.group(1), int(m.group(2))
        if kind == "convention":
            res["convention"] += 1
        elif k in RECALL:
            res["recall"] += 1
        else:
            res["behavior"] += 1
    res["total"] = sum(res.values())
    res["max"] = 2 * N_MODULES
    return res
