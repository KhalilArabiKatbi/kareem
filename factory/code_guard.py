"""Static gate for Worker-generated model.py, applied before the code ever executes.

The simulator runs the code in a subprocess with a hard timeout. This gate blocks the obvious
escape hatches (filesystem, network, processes, dynamic code) so a drifting LLM cannot touch the
machine or the harness state.
"""
from __future__ import annotations

import ast

from .errors import WorkerCodeError

MAX_SOURCE_CHARS = 60_000
ALLOWED_IMPORTS = {
    "__future__", "torch", "math", "typing", "collections", "dataclasses", "functools",
    "itertools", "numbers", "statistics", "numpy", "enum", "abc",
}
BANNED_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals",
    "vars", "delattr", "memoryview",
}
BANNED_ATTRS = {
    "__builtins__", "__subclasses__", "__globals__", "__code__", "__closure__", "__bases__",
    "__mro__", "__loader__", "__spec__", "system", "popen", "cuda", "load", "save", "hub",
    "cpp_extension", "compile", "from_file", "manual_seed", "set_num_threads",
    "use_deterministic_algorithms",
}
REQUIRED = ("FEATURE_NAMES", "featurize", "build_model")


def check_model_source(src: str) -> dict:
    if len(src) > MAX_SOURCE_CHARS:
        raise WorkerCodeError(f"model.py is {len(src)} chars; limit is {MAX_SOURCE_CHARS}")
    try:
        tree = ast.parse(src, filename="model.py")
        compile(src, "model.py", "exec")
    except SyntaxError as e:
        raise WorkerCodeError(f"model.py does not compile: {e.msg} (line {e.lineno})") from e

    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    problems.append(f"line {node.lineno}: import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                problems.append(f"line {node.lineno}: 'from {node.module} import' is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            problems.append(f"line {node.lineno}: call to '{node.func.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            problems.append(f"line {node.lineno}: attribute '.{node.attr}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in {"__builtins__", "__import__"}:
            problems.append(f"line {node.lineno}: name '{node.id}' is not allowed")
        elif isinstance(node, (ast.While,)) and isinstance(node.test, ast.Constant) and node.test.value:
            problems.append(f"line {node.lineno}: unbounded 'while True' loop is not allowed")

    top_level = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            top_level.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    top_level.add(t.id)
    for name in REQUIRED:
        if name not in top_level:
            problems.append(f"model.py must define top-level '{name}'")

    if problems:
        raise WorkerCodeError("model.py rejected by static gate:\n- " + "\n- ".join(problems[:20]))
    return {"chars": len(src), "lines": src.count("\n") + 1, "top_level": sorted(top_level)}
