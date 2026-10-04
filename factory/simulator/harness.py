"""Parent side of the simulator: spawn sim_runner in a subprocess with a hard 60 s timeout, then
validate everything it wrote. Pure Python, no LLM."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..errors import SimulatorError, SimulatorTimeout
from . import dataset

TURN_RE = re.compile(r"^TURN (\d{3})/050 \| ", re.M)


@dataclass
class SimResult:
    metrics: dict
    log_text: str
    seconds: float


def ensure_cache() -> Path:
    return dataset.build_cache(config.CACHE_DIR)


def run_simulator(exp_dir: Path, canary: str, timeout_s: int = config.SIM_TIMEOUT_S) -> SimResult:
    for name in ("simulation.log", "sim_metrics.json", "sim_error.txt"):
        (exp_dir / name).unlink(missing_ok=True)
    env = dict(os.environ)
    env.update({"PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                "PYTHONDONTWRITEBYTECODE": "1"})
    cmd = [config.PYTHON_EXE, "-m", "factory.simulator.sim_runner", str(exp_dir), str(config.CACHE_DIR)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=config.PROJECT_ROOT, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise SimulatorTimeout(f"SimulatorTimeout: model.py simulation exceeded {timeout_s}s and was killed") from e
    seconds = time.time() - t0

    if proc.returncode != 0:
        err_file = exp_dir / "sim_error.txt"
        detail = err_file.read_text(encoding="utf-8") if err_file.exists() else (proc.stderr or proc.stdout)
        raise SimulatorError(f"simulator exit code {proc.returncode}:\n{detail[-3000:]}")

    log_path, metrics_path = exp_dir / "simulation.log", exp_dir / "sim_metrics.json"
    if not log_path.exists() or not metrics_path.exists():
        raise SimulatorError("simulator finished without writing simulation.log and sim_metrics.json")
    log_text = log_path.read_text(encoding="utf-8")
    turns = TURN_RE.findall(log_text)
    if len(turns) != config.SIM_EXPECTED_TURNS or turns != [f"{i:03d}" for i in range(1, 51)]:
        raise SimulatorError(f"simulation.log must contain exactly 50 ordered TURN lines, found {len(turns)}")
    if f"canary={canary}" not in log_text:
        raise SimulatorError("simulation.log canary does not match the experiment")
    if "\nSUMMARY | sim_score=" not in log_text:
        raise SimulatorError("simulation.log has no SUMMARY line")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SimulatorError(f"sim_metrics.json invalid: {e}") from e
    if metrics.get("turns") != 50 or not 0 <= float(metrics.get("sim_score", -1)) <= 100:
        raise SimulatorError("sim_metrics.json failed validation")
    metrics["wall_seconds"] = round(seconds, 2)
    return SimResult(metrics=metrics, log_text=log_text, seconds=seconds)


def run_fingerprint(exp_dir: Path, canary: str, timeout_s: int = config.SIM_TIMEOUT_S) -> dict:
    """Functional fingerprint of model.py (features, architecture, initialisation, effective training
    configuration, loss). Runs outside the simulation budget; also applies the validity gates, so a
    silently broken featurize fails here, before any simulation time is spent."""
    out = exp_dir / "fingerprint.json"
    out.unlink(missing_ok=True)
    (exp_dir / "sim_error.txt").unlink(missing_ok=True)
    env = dict(os.environ)
    env.update({"PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                "PYTHONDONTWRITEBYTECODE": "1"})
    cmd = [config.PYTHON_EXE, "-m", "factory.simulator.sim_runner", str(exp_dir), str(config.CACHE_DIR), "--fingerprint"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=config.PROJECT_ROOT, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise SimulatorTimeout(f"SimulatorTimeout: fingerprinting model.py exceeded {timeout_s}s") from e
    if proc.returncode != 0 or not out.exists():
        err_file = exp_dir / "sim_error.txt"
        detail = err_file.read_text(encoding="utf-8") if err_file.exists() else (proc.stderr or proc.stdout)
        raise SimulatorError(f"model.py failed the pre-simulation checks (exit {proc.returncode}):\n{detail[-3000:]}")
    fp = json.loads(out.read_text(encoding="utf-8"))
    fp["seconds"] = round(time.time() - t0, 2)
    return fp


def run_confirmation(exp_dir: Path, timeout_s: int = config.SIM_TIMEOUT_S) -> dict:
    """Retrain model.py and evaluate it on the hidden confirmation set (never used for selection)."""
    out = exp_dir / "confirm_metrics.json"
    out.unlink(missing_ok=True)
    env = dict(os.environ)
    env.update({"PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                "PYTHONDONTWRITEBYTECODE": "1"})
    cmd = [config.PYTHON_EXE, "-m", "factory.simulator.sim_runner", str(exp_dir), str(config.CACHE_DIR), "--confirm"]
    try:
        proc = subprocess.run(cmd, cwd=config.PROJECT_ROOT, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise SimulatorTimeout(f"confirmation re-score exceeded {timeout_s}s") from e
    if proc.returncode != 0 or not out.exists():
        raise SimulatorError(f"confirmation re-score failed (exit {proc.returncode}): {(proc.stderr or '')[-1500:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def warm_up() -> float:
    """Import torch and load the cache once so the first real simulation does not pay a cold start."""
    t0 = time.time()
    subprocess.run([config.PYTHON_EXE, "-c", "import torch; from factory.simulator import dataset as D; "
                    f"D.load_cache(__import__('pathlib').Path(r'{config.CACHE_DIR}'))"],
                   cwd=config.PROJECT_ROOT, capture_output=True, timeout=300)
    return time.time() - t0


def summary_lines(log_text: str) -> str:
    """The simulation summary handed to the Doc agent (header + aggregate lines, no per-turn rows)."""
    keep = [ln for ln in log_text.splitlines() if not ln.startswith("TURN ")]
    return "\n".join(keep)
