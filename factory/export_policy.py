"""Export a trained intervention MLP as a deployable package.

    python -m factory.export_policy [iteration] [--out workspace/deploy]

Default iteration: the confirmed best from workspace/confirmation.json (else the spec best). The
model is re-trained with its exact deterministic simulator protocol on a scratch copy (experiment
artefacts are untouched), and the package gets:
    policy_model.py          featurize + network definition (the Worker's model.py)
    intervention_policy.pt   trained weights of every ensemble member + input normalisation
    policy_runtime.py        torch-only loader: InterventionPolicy.load(dir).decide(history)
    policy_card.json         provenance, configuration, measured scores, parity check
A parity check replays the evaluation scenarios through the exported runtime and requires the same
mean return as the simulator.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from factory import config
from factory.state_store import read_json


def pick_iteration() -> int:
    conf = read_json(config.WORKING_DIR / "confirmation.json", None)
    if conf and conf.get("confirmed_best") is not None:
        return int(conf["confirmed_best"])
    return int(read_json(config.FACTORY_STATE_PATH)["best"]["iteration"])


def parity(out: Path) -> dict:
    sys.path.insert(0, str(out))
    try:
        from policy_runtime import ACTIONS, InterventionPolicy  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    from factory.simulator import env as E

    pol = InterventionPolicy.load(out)
    scs = E.eval_scenarios()
    states = [E.EnvState() for _ in scs]
    hists = [[] for _ in scs]
    single_agree = checks = 0
    for t in range(E.HORIZON):
        for i, sc in enumerate(scs):
            hists[i].append(E.observe(states[i], sc))
        acts = pol.decide_batch(hists)
        if t % 8 == 0:  # the one-history API must agree with the batched one
            for i in range(0, len(scs), 10):
                single_agree += int(pol.decide(hists[i]) == acts[i])
                checks += 1
        for i, sc in enumerate(scs):
            E.step(states[i], ACTIONS.index(acts[i]), sc)
    per_turn = [round(sum(s.reward for s in states[k: k + E.SEEDS_PER_TURN]) / E.SEEDS_PER_TURN, 4)
                for k in range(0, len(states), E.SEEDS_PER_TURN)]
    return {"mean_return": sum(s.reward for s in states) / len(states), "turn_returns": per_turn,
            "single_vs_batch_agreement": single_agree / max(1, checks)}


def main(argv: list[str]) -> int:
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else config.WORKING_DIR / "deploy"
    nums = [a for a in argv if a.isdigit()]
    it = int(nums[0]) if nums else pick_iteration()
    src = config.exp_dir(it)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "exp"
        work.mkdir()
        shutil.copy(src / "model.py", work / "model.py")
        shutil.copy(src / "config.json", work / "config.json")
        proc = subprocess.run([config.PYTHON_EXE, "-m", "factory.simulator.sim_runner", str(work), str(config.CACHE_DIR),
                               "--export", str(out)], cwd=config.PROJECT_ROOT, capture_output=True, text=True,
                              timeout=300)
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
            return 1
        metrics = json.loads((work / "sim_metrics.json").read_text(encoding="utf-8"))
    shutil.copy(src / "model.py", out / "policy_model.py")
    shutil.copy(config.TEMPLATES_DIR / "policy_runtime.py", out / "policy_runtime.py")
    sys.dont_write_bytecode = True
    check = parity(out)
    shutil.rmtree(out / "__pycache__", ignore_errors=True)
    ok = check["turn_returns"] == metrics["turn_returns"]  # all 50 turns, at the simulator's stored precision
    conf = read_json(config.WORKING_DIR / "confirmation.json", {}) or {}
    crow = next((r for r in conf.get("rows", []) if r["iteration"] == it), {})
    card = {
        "iteration": it,
        "evaluation_sim_score": metrics["sim_score"],
        "confirmation_sim_score": crow.get("confirm_sim"),
        "mean_return_eval": metrics["mean_return"],
        "effective_config": metrics["effective_config"],
        "features": metrics["training"]["feature_names"],
        "actions": ["NOOP", "COMPACT", "PRUNE_TOOLS", "REINJECT_INSTRUCTIONS", "CHECKPOINT_RESET", "RETRIEVE_MEMORY"],
        "parity": {"simulator_mean_return": metrics["mean_return"], "runtime_mean_return": round(check["mean_return"], 4),
                   "all_50_turns_identical": ok, "single_vs_batch_agreement": check["single_vs_batch_agreement"]},
        "trained_in": "deterministic context-health simulator (synthetic telemetry); see README.md for sim-to-real caveats",
    }
    (out / "policy_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps({k: card[k] for k in ("iteration", "evaluation_sim_score", "confirmation_sim_score", "parity")}, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
