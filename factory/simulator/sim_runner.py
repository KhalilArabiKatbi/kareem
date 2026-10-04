"""Simulator subprocess entry point. Pure Python, no LLM.

    python -m factory.simulator.sim_runner <exp_dir> <cache_dir> [--fingerprint]

Reads exp_dir/model.py, trains it on the cached synthetic data (targets from a selectable teacher,
optionally on behaviour + on-policy states, optionally as a seed ensemble), evaluates it closed-loop
on 50 turns x 4 seeds, and writes exp_dir/simulation.log plus exp_dir/sim_metrics.json.

`--fingerprint` stops before training: it applies the validity gates and writes
exp_dir/fingerprint.json, a hash of everything that determines the trained model (effective
configuration, standardised features, initial parameters, initial logits, loss). Two model.py files
with the same fingerprint train to the same policy.

Every budget is counted (optimizer steps, parameter-samples), never timed, so results do not depend
on machine load. Wall-clock limits exist only as hard failures.

Exit codes: 0 ok, 2 model error (details in exp_dir/sim_error.txt), 3 harness/cache error.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

T0 = time.perf_counter()

import torch  # noqa: E402

from . import dataset as D  # noqa: E402
from . import env as E  # noqa: E402

T_IMPORT = time.perf_counter() - T0
EXPECTED_TURNS = 50
MAX_FEATURES = 1024
MAX_OPT_STEPS = 10_000              # deterministic training budget per model: optimizer steps
MAX_PARAM_SAMPLES = 1.0e11          # deterministic training budget per model: parameters x samples seen
SAFETY_BRAKE_S = 30.0               # wall-clock guard for supervised training (hard failure)
FEATURIZE_BUDGET_S = 15.0
AGREEMENT_EVERY = 8                 # teacher-agreement probe every 8th step (cost control)
SEED = int(os.environ.get("FACTORY_SIM_SEED", "1234"))  # override only for seed-robustness benchmarks

TRAIN_LIMITS = {
    "lr": (1e-5, 0.1, 1e-3),
    "epochs": (1, 300, 40),
    "batch_size": (16, 2048, 256),
    "weight_decay": (0.0, 0.1, 0.0),
    "label_smoothing": (0.0, 0.3, 0.0),
    "grad_clip": (0.0, 100.0, 0.0),
    "target_temperature": (0.01, 10.0, 0.5),
    "ensemble_seeds": (1, 5, 3),
    "self_episodes": (28, 126, 84),
}
INT_KEYS = {"epochs", "batch_size", "ensemble_seeds", "self_episodes"}
OPTIMIZERS = {"adam", "adamw", "sgd", "rmsprop"}
CLASS_WEIGHTING = {"none", "balanced", "sqrt_balanced"}
TARGETS = {"hard", "soft", "regret"}
DATA = {"behaviour", "behaviour+onpolicy", "behaviour+self"}
EARLY_STOP = {"none", "val_regret"}
DAGGER_SEED = 20_000                # training-side seeds for per-experiment DAgger rollouts
LABEL_TIMEOUT_S = 40
LABEL_WORKERS = max(1, min(6, (os.cpu_count() or 2) // 2))
# Every TRAIN_CONFIG key the simulator reads (the model.py API). The hard-coded-state-space audit
# derives its exemption list from this, so the API and the audit cannot drift apart.
CONTRACT_KEYS = frozenset(TRAIN_LIMITS) | {"teacher", "target", "data", "optimizer", "class_weighting",
                                           "standardize", "early_stop"}


class ModelContractError(Exception):
    pass


def _clamp(name, value, notes):
    lo, hi, default = TRAIN_LIMITS[name]
    try:
        v = float(value)
    except (TypeError, ValueError):
        notes.append(f"{name}={value!r} invalid, used {default}")
        return default
    if math.isnan(v):
        notes.append(f"{name} NaN, used {default}")
        return default
    c = min(hi, max(lo, v))
    if c != v:
        notes.append(f"{name}={v} clamped to {c}")
    return int(c) if name in INT_KEYS else c


def load_model_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("FEATURE_NAMES", "featurize", "build_model"):
        if not hasattr(mod, name):
            raise ModelContractError(f"model.py must define {name}")
    names = list(mod.FEATURE_NAMES)
    if not names or len(names) > MAX_FEATURES:
        raise ModelContractError(f"FEATURE_NAMES must have 1..{MAX_FEATURES} entries, got {len(names)}")
    return mod, names


def _feat(mod, history, nf):
    v = mod.featurize(history)
    if hasattr(v, "tolist"):
        v = v.tolist()
    v = [float(x) for x in v]
    if len(v) != nf:
        raise ModelContractError(f"featurize returned {len(v)} values, FEATURE_NAMES has {nf}")
    return v


def featurize_all(mod, histories, nf, deadline):
    rows = []
    for h in histories:
        rows.append(_feat(mod, h, nf))
        if time.perf_counter() > deadline:
            raise ModelContractError("featurize is too slow (featurization budget exceeded)")
    x = torch.tensor(rows, dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ModelContractError("featurize produced NaN or Inf")
    return x


def run_lockstep(mod, model, mu, sd, nf, scenarios, agree_every=0):
    """Closed-loop rollout of many scenarios in lockstep with one batched forward per step."""
    n = len(scenarios)
    states = [E.EnvState() for _ in range(n)]
    hists = [[] for _ in range(n)]
    agree = [0] * n
    agree_n = [0] * n
    for t in range(E.HORIZON):
        rows = []
        for i in range(n):
            hists[i].append(E.observe(states[i], scenarios[i]))
            rows.append(_feat(mod, hists[i], nf))
        x = (torch.tensor(rows, dtype=torch.float32) - mu) / sd
        if not torch.isfinite(x).all():
            raise ModelContractError("featurize returned invalid values during a rollout")
        with torch.no_grad():
            logits = model(x)
        if tuple(logits.shape) != (n, E.N_ACTIONS):
            raise ModelContractError(f"model output shape {tuple(logits.shape)} != ({n}, {E.N_ACTIONS})")
        acts = logits.argmax(1).tolist()
        for i in range(n):
            if agree_every and t % agree_every == 0:
                agree[i] += int(acts[i] == E.oracle_action(states[i], scenarios[i], "v1"))
                agree_n[i] += 1
            E.step(states[i], acts[i], scenarios[i])
    return states, agree, agree_n


def rollout_record(mod, model, mu, sd, nf, scenarios):
    """Closed-loop rollout (lockstep, batched) that records every observation history and action."""
    n = len(scenarios)
    states = [E.EnvState() for _ in range(n)]
    hists = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    for _t in range(E.HORIZON):
        rows = []
        for i in range(n):
            hists[i].append(E.observe(states[i], scenarios[i]))
            rows.append(_feat(mod, hists[i], nf))
        x = (torch.tensor(rows, dtype=torch.float32) - mu) / sd
        with torch.no_grad():
            a_batch = model(x).argmax(1).tolist()
        for i in range(n):
            acts[i].append(a_batch[i])
            E.step(states[i], a_batch[i], scenarios[i])
    return hists, acts


def _episode_returns(scenarios, actions):
    out = []
    for sc, acts in zip(scenarios, actions):
        s = E.EnvState()
        for a in acts:
            E.step(s, a, sc)
        out.append(s.reward)
    return out


def label_with_teacher(exp_dir: Path, teacher: str, scenarios, actions):
    """Teacher values for every visited state, computed by the labeler subprocess (process pool)."""
    jobs_path, out_path = exp_dir / "_dagger_jobs.pkl", exp_dir / "_dagger_labels.pkl"
    with open(jobs_path, "wb") as f:
        pickle.dump({"teacher": teacher, "workers": LABEL_WORKERS,
                     "episodes": [(sc.sid, sc.family, sc.seed, a) for sc, a in zip(scenarios, actions)]}, f)
    try:
        proc = subprocess.run([sys.executable, "-m", "factory.simulator.labeler", str(jobs_path), str(out_path)],
                              capture_output=True, text=True, timeout=LABEL_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise ModelContractError(f"DAgger labelling exceeded {LABEL_TIMEOUT_S}s; lower self_episodes") from e
    if proc.returncode != 0:
        raise RuntimeError(f"labeler failed: {proc.stderr[-1500:]}")
    with open(out_path, "rb") as f:
        labels = pickle.load(f)
    jobs_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)
    return labels


class Ensemble(torch.nn.Module):
    """Average of the members' logits (harness-owned wrapper)."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members]).mean(0)


def _hash_tensor(h, t: torch.Tensor) -> None:
    t = t.detach().contiguous().to(torch.float32)
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())


def main(exp_dir: Path, cache_dir: Path, fingerprint_only: bool = False, confirm: bool = False,
         export_dir: Path | None = None) -> int:
    cfg = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    experiment, canary = cfg["experiment"], cfg["canary"]
    try:
        cache = D.load_cache(cache_dir)
    except Exception as e:  # harness problem, not a model problem
        print(f"CACHE ERROR: {e}", file=sys.stderr)
        return 3

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    timers = {"import_s": round(T_IMPORT, 2)}

    notes: list[str] = []
    try:
        mod, feat_names = load_model_module(exp_dir / "model.py")
        nf = len(feat_names)
        tc = dict(getattr(mod, "TRAIN_CONFIG", {}) or {})
        teacher = str(tc.get("teacher", "v1")).lower()
        if teacher not in E.TEACHERS:
            notes.append(f"teacher={teacher!r} unknown, used v1")
            teacher = "v1"
        target = str(tc.get("target", "hard")).lower()
        if target not in TARGETS:
            notes.append(f"target={target!r} unsupported, used hard")
            target = "hard"
        data = str(tc.get("data", "behaviour")).lower()
        if data not in DATA:
            notes.append(f"data={data!r} unsupported, used behaviour")
            data = "behaviour"
        hp = {k: _clamp(k, tc.get(k, TRAIN_LIMITS[k][2]), notes) for k in TRAIN_LIMITS}
        opt_name = str(tc.get("optimizer", "adam")).lower()
        if opt_name not in OPTIMIZERS:
            notes.append(f"optimizer={opt_name!r} unsupported, used adam")
            opt_name = "adam"
        early_stop = str(tc.get("early_stop", "none")).lower()
        if early_stop not in EARLY_STOP:
            notes.append(f"early_stop={early_stop!r} unsupported, used none")
            early_stop = "none"
        cw_mode = str(tc.get("class_weighting", "none")).lower()
        if cw_mode not in CLASS_WEIGHTING:
            notes.append(f"class_weighting={cw_mode!r} unsupported, used none")
            cw_mode = "none"
        custom_loss = getattr(mod, "loss_fn", None)
        loss_arity = 0
        if custom_loss is not None:
            try:
                loss_arity = len(inspect.signature(custom_loss).parameters)
            except (TypeError, ValueError):
                loss_arity = 2
        # settings that have no effect are reported, so a coordinate cannot silently test nothing
        if custom_loss is not None:
            notes.append("ignored: loss_fn replaces the built-in target (target, class_weighting, label_smoothing, "
                         "target_temperature have no effect)")
        else:
            if target != "hard" and cw_mode != "none":
                notes.append(f"ignored: class_weighting={cw_mode} has no effect with target={target}")
            if target != "hard" and hp["label_smoothing"] > 0:
                notes.append(f"ignored: label_smoothing has no effect with target={target}")
            if target != "soft" and "target_temperature" in tc:
                notes.append(f"ignored: target_temperature has no effect with target={target}")
        if hasattr(mod, "FINETUNE"):
            notes.append("ignored: FINETUNE is no longer supported (it did not improve held-out return)")
        standardize = bool(tc.get("standardize", True))
        effective = {
            "teacher": teacher, "target": target, "data": data, "ensemble_seeds": hp["ensemble_seeds"],
            "target_temperature": hp["target_temperature"] if target == "soft" and custom_loss is None else None,
            "optimizer": opt_name, "lr": hp["lr"], "batch_size": hp["batch_size"], "epochs": hp["epochs"],
            "weight_decay": hp["weight_decay"], "grad_clip": hp["grad_clip"], "standardize": standardize,
            "class_weighting": cw_mode if target == "hard" and custom_loss is None else None,
            "label_smoothing": hp["label_smoothing"] if target == "hard" and custom_loss is None else None,
            "custom_loss": None if custom_loss is None else f"arity{loss_arity}",
            "self_episodes": hp["self_episodes"] if data == "behaviour+self" else None,
            "early_stop": early_stop,
        }

        # ---- training data ----
        t_feat = time.perf_counter()
        deadline = t_feat + FEATURIZE_BUDGET_S
        tr_h, tr_y, tr_q, va_h, va_y, va_q = [], [], [], [], [], []
        sources = [("behaviour", cache["train"])]
        if data == "behaviour+onpolicy":
            sources.append(("onpolicy", cache["onpolicy"]))
        for src, eps in sources:
            for ei, ep in enumerate(eps):
                is_val = ei % 7 == 0
                if is_val and src == "onpolicy":
                    continue  # validation stays on behaviour states, comparable across data settings
                H, Y, Q = ep["history"], ep["labels"][teacher], ep["q"][teacher]
                dh, dy, dq = (va_h, va_y, va_q) if is_val else (tr_h, tr_y, tr_q)
                for t in range(len(H)):
                    dh.append(H[: t + 1])
                    dy.append(Y[t])
                    dq.append(Q[t])
        x_tr = featurize_all(mod, tr_h, nf, deadline)
        x_va = featurize_all(mod, va_h, nf, deadline)
        y_tr, y_va = torch.tensor(tr_y, dtype=torch.long), torch.tensor(va_y, dtype=torch.long)
        q_tr, q_va = torch.tensor(tr_q, dtype=torch.float32), torch.tensor(va_q, dtype=torch.float32)
        timers["featurize_s"] = round(time.perf_counter() - t_feat, 2)

        # validity gate: a silently failing featurize produces constant columns
        const = int((x_tr.std(0) < 1e-9).sum())
        limit = max(3, int(0.10 * nf))
        if const > limit:
            raise ModelContractError(
                f"{const} of {nf} features are constant over all {x_tr.shape[0]} training states (limit {limit}); "
                f"featurize is probably swallowing an error and returning defaults")

        if standardize:
            mu, sd = x_tr.mean(0), x_tr.std(0).clamp_min(1e-6)
        else:
            mu, sd = torch.zeros(nf), torch.ones(nf)
        x_tr_n, x_va_n = (x_tr - mu) / sd, (x_va - mu) / sd

        # ---- models (seed ensemble) ----
        k_ens = hp["ensemble_seeds"]
        members = []
        for k in range(k_ens):
            torch.manual_seed(SEED + k)
            m = mod.build_model(nf, E.N_ACTIONS)
            if not isinstance(m, torch.nn.Module):
                raise ModelContractError("build_model must return a torch.nn.Module")
            members.append(m)
        with torch.no_grad():
            probe = members[0](x_tr_n[:2])
        if tuple(probe.shape) != (2, E.N_ACTIONS):
            raise ModelContractError(f"model output shape {tuple(probe.shape)} != (batch, {E.N_ACTIONS})")
        n_params = sum(p.numel() for p in members[0].parameters())
        if n_params == 0 or n_params * k_ens > 5_000_000:
            raise ModelContractError(f"parameter count {n_params} x {k_ens} members outside 1..5,000,000")

        n = x_tr_n.shape[0]
        bs = hp["batch_size"]
        steps_per_epoch = math.ceil(n / bs)
        epochs_allowed = min(hp["epochs"], max(1, (MAX_OPT_STEPS // k_ens) // steps_per_epoch),
                             max(1, int(MAX_PARAM_SAMPLES / k_ens // (n_params * n))))
        budget_hit = epochs_allowed < hp["epochs"]
        if budget_hit:
            notes.append(f"epochs capped {hp['epochs']} -> {epochs_allowed} by the deterministic compute budget "
                         f"(<= {MAX_OPT_STEPS} optimizer steps and <= {MAX_PARAM_SAMPLES:.0e} parameter-samples, "
                         f"shared by {k_ens} ensemble member(s))")
        effective["epochs"] = epochs_allowed

        counts = torch.bincount(y_tr, minlength=E.N_ACTIONS).float().clamp_min(1.0)
        if cw_mode == "balanced":
            weight = counts.sum() / (E.N_ACTIONS * counts)
        elif cw_mode == "sqrt_balanced":
            weight = (counts.sum() / (E.N_ACTIONS * counts)).sqrt()
        else:
            weight = None
        ce = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=hp["label_smoothing"])
        tau = hp["target_temperature"]

        def compute_loss(logits, y, q):
            if custom_loss is not None:
                return custom_loss(logits, y, q) if loss_arity >= 3 else custom_loss(logits, y)
            if target == "hard":
                return ce(logits, y)
            logp = torch.log_softmax(logits, 1)
            if target == "soft":
                return -(torch.softmax(q / tau, 1) * logp).sum(1).mean()
            regret = q.max(1, keepdim=True).values - q
            return (logp.exp() * regret).sum(1).mean()

        # ---- functional fingerprint (before any training) ----
        if fingerprint_only:
            parts = {}
            hc = hashlib.sha256(json.dumps(effective, sort_keys=True).encode())
            parts["config"] = hc.hexdigest()[:16]
            for name, t in (("x_train", x_tr_n), ("x_val", x_va_n)):
                hx = hashlib.sha256()
                _hash_tensor(hx, t)
                parts[name] = hx.hexdigest()[:16]
            hi = hashlib.sha256()
            for t in members[0].state_dict().values():  # values in registration order; names are cosmetic
                _hash_tensor(hi, t)
            parts["init_params"] = hi.hexdigest()[:16]
            hl = hashlib.sha256()
            with torch.no_grad():
                _hash_tensor(hl, members[0](x_tr_n[:1024]))
            parts["init_logits"] = hl.hexdigest()[:16]
            if custom_loss is not None:
                with torch.no_grad():
                    lv = compute_loss(members[0](x_tr_n[:256]), y_tr[:256], q_tr[:256])
                parts["custom_loss"] = f"{float(lv):.6e}"
            h = hashlib.sha256(json.dumps(parts, sort_keys=True).encode())
            (exp_dir / "fingerprint.json").write_text(json.dumps({
                "fingerprint": h.hexdigest(), "parts": parts, "effective_config": effective, "features": nf,
                "constant_features": const, "params": n_params, "notes": notes,
            }, indent=2), encoding="utf-8")
            return 0

        # ---- training ----
        def make_opt(params):
            if opt_name == "adam":
                return torch.optim.Adam(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
            if opt_name == "adamw":
                return torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
            if opt_name == "sgd":
                return torch.optim.SGD(params, lr=hp["lr"], momentum=0.9, weight_decay=hp["weight_decay"])
            return torch.optim.RMSprop(params, lr=hp["lr"], weight_decay=hp["weight_decay"])

        def val_regret_of(model):
            with torch.no_grad():
                pv = model(x_va_n).argmax(1)
            return float((q_va.max(1).values - q_va.gather(1, pv[:, None]).squeeze(1)).mean())

        best_epochs: list[int] = []

        def fit(models, xn, y, q, epochs, record=True):
            last = float("nan")
            n_rows = xn.shape[0]
            t_fit = time.perf_counter()  # the brake bounds ONE supervised fit, not rollouts/labelling
            for k, model in enumerate(models):
                params = [p for p in model.parameters() if p.requires_grad]
                opt = make_opt(params)
                gen = torch.Generator().manual_seed(SEED + k)
                best = (float("inf"), None, 0)
                for _ep in range(epochs):
                    model.train()
                    perm = torch.randperm(n_rows, generator=gen)
                    for i in range(0, n_rows, bs):
                        idx = perm[i: i + bs]
                        loss = compute_loss(model(xn[idx]), y[idx], q[idx])
                        if not torch.isfinite(loss):
                            raise ModelContractError("training loss became NaN/Inf")
                        opt.zero_grad()
                        loss.backward()
                        if hp["grad_clip"] > 0:
                            torch.nn.utils.clip_grad_norm_(params, hp["grad_clip"])
                        opt.step()
                        last = float(loss.detach())
                    if time.perf_counter() - t_fit > SAFETY_BRAKE_S:
                        raise ModelContractError(f"supervised training exceeded {SAFETY_BRAKE_S}s wall-clock; reduce "
                                                 f"the model size, epochs, batch count, ensemble_seeds or self_episodes")
                    if early_stop == "val_regret":
                        model.eval()
                        reg = val_regret_of(model)
                        if reg < best[0] - 1e-12:  # keep the checkpoint with the lowest validation regret
                            best = (reg, {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}, _ep + 1)
                if early_stop == "val_regret" and best[1] is not None:
                    model.load_state_dict(best[1])
                    if record:
                        best_epochs.append(best[2])
                elif record:
                    best_epochs.append(epochs)
                model.eval()
            return last

        def epoch_cap(n_rows, n_models):
            spe = math.ceil(n_rows / bs)
            return min(hp["epochs"], max(1, (MAX_OPT_STEPS // n_models) // spe),
                       max(1, int(MAX_PARAM_SAMPLES / n_models // (n_params * n_rows))))

        t_train = time.perf_counter()
        dagger = None
        if data == "behaviour+self":
            # Round 1: one seed on behaviour states. Its OWN closed-loop states are then labelled by the
            # teacher (DAgger is policy-specific: a shared reference student's states do not help).
            torch.manual_seed(SEED)
            r1 = mod.build_model(nf, E.N_ACTIONS)
            fit([r1], x_tr_n, y_tr, q_tr, epoch_cap(n, 1), record=False)
            r1_s = time.perf_counter() - t_train
            scs = [E.Scenario(f"S-DG-{k:03d}", E.FAMILIES[k % len(E.FAMILIES)], DAGGER_SEED + k)
                   for k in range(hp["self_episodes"])]
            hists, acts = rollout_record(mod, r1, mu, sd, nf, scs)
            t_lab = time.perf_counter()
            labels = label_with_teacher(exp_dir, teacher, scs, acts)
            label_s = time.perf_counter() - t_lab
            s_h, s_y, s_q = [], [], []
            for H, Q in zip(hists, labels):
                for t in range(len(H)):
                    s_h.append(H[: t + 1])
                    s_q.append(Q[t])
                    s_y.append(E.argmax_first(Q[t]))
            x_self = featurize_all(mod, s_h, nf, time.perf_counter() + FEATURIZE_BUDGET_S)
            x_tr = torch.cat([x_tr, x_self])
            y_tr = torch.cat([y_tr, torch.tensor(s_y, dtype=torch.long)])
            q_tr = torch.cat([q_tr, torch.tensor(s_q, dtype=torch.float32)])
            if standardize:
                mu, sd = x_tr.mean(0), x_tr.std(0).clamp_min(1e-6)
            x_tr_n, x_va_n = (x_tr - mu) / sd, (x_va - mu) / sd
            n = x_tr_n.shape[0]
            epochs_allowed = epoch_cap(n, k_ens)
            budget_hit = epochs_allowed < hp["epochs"]
            effective["epochs"] = epochs_allowed
            dagger = {"self_episodes": len(scs), "self_states": len(s_h), "teacher": teacher,
                      "round1_seconds": round(r1_s, 2), "label_seconds": round(label_s, 2),
                      "label_workers": LABEL_WORKERS,
                      "round1_mean_return_on_self_episodes": round(
                          sum(E_reward for E_reward in _episode_returns(scs, acts)) / len(scs), 3)}
        last_loss = fit(members, x_tr_n, y_tr, q_tr, epochs_allowed)
        policy = members[0] if k_ens == 1 else Ensemble(members).eval()
        if export_dir is not None:  # deployable artefact: exact trained weights + input normalisation
            export_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format": "context-health-intervention-policy/1",
                "feature_names": feat_names,
                "actions": list(E.ACTIONS),
                "mu": mu.clone(), "sd": sd.clone(),
                "members": [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in members],
                "effective_config": effective,
                "member_seeds": [SEED + k for k in range(k_ens)],
            }, export_dir / "intervention_policy.pt")
        timers["train_s"] = round(time.perf_counter() - t_train, 2)

        with torch.no_grad():
            pred_tr = policy(x_tr_n).argmax(1)
            pred_va = policy(x_va_n).argmax(1)
        train_acc = float((pred_tr == y_tr).float().mean())
        val_acc = float((pred_va == y_va).float().mean())
        val_regret = float((q_va.max(1).values - q_va.gather(1, pred_va[:, None]).squeeze(1)).mean())

        # ---- closed-loop evaluation: 50 turns x 4 seeds, lockstep ----
        t_eval = time.perf_counter()
        turns_sc = E.confirmation_turns() if confirm else E.eval_turns()
        flat = [sc for turn in turns_sc for sc in turn]
        states, agree, agree_n = run_lockstep(mod, policy, mu, sd, nf, flat, agree_every=AGREEMENT_EVERY)
        timers["eval_s"] = round(time.perf_counter() - t_eval, 2)
    except Exception:
        # confirmation re-scoring never touches the scored experiment's artefacts
        err_name = "confirm_error.txt" if confirm else "sim_error.txt"
        (exp_dir / err_name).write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr)
        return 2

    # ---- aggregate ----
    base = {b["sid"]: b for b in (cache["confirmation"] if confirm else cache["baselines"])}
    turns = []
    k = 0
    for turn in turns_sc:
        rows = []
        for sc in turn:
            s = states[k]
            rows.append((s, base[sc.sid], agree[k], agree_n[k]))
            k += 1
        m = len(rows)
        turns.append({
            "family": turn[0].family,
            "sid": turn[0].sid.rsplit("-", 1)[0],
            "return": sum(r[0].reward for r in rows) / m,
            "noop": sum(r[1]["noop"] for r in rows) / m,
            "heuristic": sum(r[1]["heuristic"] for r in rows) / m,
            "oracle": sum(r[1]["oracle"] for r in rows) / m,
            "teacher_v2": sum(r[1]["teacher_v2"] for r in rows) / m,
            "clairvoyant": sum(r[1]["clairvoyant"] for r in rows) / m,
            "success": sum(int(r[0].progress >= E.TARGET) for r in rows) / m,
            "progress": sum(r[0].progress for r in rows) / m,
            "overflows": sum(r[0].overflows for r in rows),
            "overflowed": sum(1 for r in rows if r[0].overflows),
            "mean_health": sum(r[0].health_sum / E.HORIZON for r in rows) / m,
            "oracle_agree": sum(r[2] for r in rows) / max(1, sum(r[3] for r in rows)),
            "actions": [sum(r[0].action_counts[a] for r in rows) for a in range(E.N_ACTIONS)],
        })

    n_t = len(turns)
    mean = lambda key: sum(t[key] for t in turns) / n_t  # noqa: E731
    mr, mn, mh, mo, m2, mc = (mean(x) for x in ("return", "noop", "heuristic", "oracle", "teacher_v2", "clairvoyant"))
    sim_score = D.anchored_score(mr, mn, mh, mc)
    tot_actions = [sum(t["actions"][a] for t in turns) for a in range(E.N_ACTIONS)]
    ta = sum(tot_actions)
    entropy = -sum((c / ta) * math.log(c / ta) for c in tot_actions if c) / math.log(E.N_ACTIONS)
    n_scen = len(flat)
    fam = {}
    for t in turns:
        fam.setdefault(t["family"], []).append(t)
    family_stats = {
        name: {key: round(sum(t[key] for t in ts) / len(ts), 3)
               for key in ("return", "heuristic", "oracle", "teacher_v2", "clairvoyant", "success")}
        for name, ts in fam.items()
    }
    runtime = time.perf_counter() - T0
    timers["total_s"] = round(runtime, 2)
    metrics = {
        "experiment": experiment,
        "sim_score": round(sim_score, 2),
        "mean_return": round(mr, 3),
        "noop_return": round(mn, 3),
        "heuristic_return": round(mh, 3),
        "oracle_return": round(mo, 3),
        "teacher_v2_return": round(m2, 3),
        "clairvoyant_return": round(mc, 3),
        "success_rate": round(mean("success"), 4),
        "overflow_rate": round(sum(t["overflowed"] for t in turns) / n_scen, 4),
        "mean_health": round(mean("mean_health"), 4),
        "oracle_agreement": round(sum(agree) / max(1, sum(agree_n)), 4),
        "action_distribution": {E.ACTIONS[a]: tot_actions[a] for a in range(E.N_ACTIONS)},
        "noop_share": round(tot_actions[0] / ta, 4),
        "action_entropy": round(entropy, 4),
        "turn_returns": [round(t["return"], 4) for t in turns],
        "family": family_stats,
        "effective_config": effective,
        "training": {
            "samples": int(x_tr.shape[0]), "val_samples": int(x_va.shape[0]), "features": nf,
            "feature_names": feat_names, "params": n_params, "ensemble_seeds": k_ens, "teacher": teacher,
            "target": target, "data": data, "target_temperature": effective["target_temperature"],
            "epochs_requested": hp["epochs"], "epochs_run": epochs_allowed, "budget_hit": budget_hit,
            "optimizer": opt_name, "lr": hp["lr"], "batch_size": bs, "class_weighting": cw_mode,
            "custom_loss": custom_loss is not None, "custom_loss_uses_q": custom_loss is not None and loss_arity >= 3,
            "early_stop": early_stop, "best_epochs": best_epochs,
            "constant_features": const, "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
            "val_regret": round(val_regret, 4), "final_loss": round(last_loss, 5),
            "featurize_seconds": timers["featurize_s"], "train_seconds": timers["train_s"],
        },
        "finetune": None,
        "dagger": dagger,
        "timers": timers,
        "notes": notes,
        "turns": n_t,
        "scenarios": n_scen,
        "runtime_seconds": round(runtime, 2),
    }

    lines = [
        "# CONTEXT HEALTH SIMULATION LOG v3",
        f"# experiment={experiment} canary={canary}",
        "# policy: argmax of the trained intervention MLP (ensembles average logits), closed-loop, deterministic seeds",
        f"# evaluation: {n_t} turns x {E.SEEDS_PER_TURN} seeds of one scenario family = {n_scen} scenarios; TURN values are per-turn means",
        "# score anchors (mean return, piecewise linear): NOOP policy = 0, threshold heuristic = 50, clairvoyant bound = 100",
        "# oracle = teacher v1 (one-step lookahead over imagined futures); teacher_v2 = two-step lookahead; clairvoyant = search over the true future (unreachable)",
        f"# actions: {', '.join(f'{E.ACTION_ABBR[i]}={E.ACTIONS[i]}' for i in range(E.N_ACTIONS))}",
    ]
    tr = metrics["training"]
    lines.append(
        f"TRAINING | samples={tr['samples']} | val_samples={tr['val_samples']} | features={tr['features']} | "
        f"params={tr['params']} | ensemble_seeds={k_ens} | teacher={teacher} | target={target} | data={data} | "
        f"tau={effective['target_temperature']} | optimizer={tr['optimizer']} | lr={tr['lr']:g} | batch={tr['batch_size']} | "
        f"epochs_run={tr['epochs_run']}/{tr['epochs_requested']} | class_weighting={effective['class_weighting']} | "
        f"early_stop={early_stop} | kept_epochs={','.join(str(e) for e in best_epochs)} | "
        f"custom_loss={int(tr['custom_loss'])} | loss_uses_q={int(tr['custom_loss_uses_q'])} | "
        f"constant_features={const} | train_acc={tr['train_acc']:.4f} | val_acc={tr['val_acc']:.4f} | "
        f"val_regret={tr['val_regret']:.4f} | final_loss={tr['final_loss']:.5f}"
    )
    if dagger:
        lines.append(
            f"DAGGER | self_episodes={dagger['self_episodes']} | self_states={dagger['self_states']} | "
            f"teacher={dagger['teacher']} | round1_return_on_self_episodes={dagger['round1_mean_return_on_self_episodes']:+.3f} | "
            f"round1_s={dagger['round1_seconds']:.2f} | label_s={dagger['label_seconds']:.2f}"
        )
    for note in notes:
        lines.append(f"NOTE | {note}")
    for i, t in enumerate(turns, 1):
        acts = " ".join(f"{E.ACTION_ABBR[a]}{t['actions'][a]}" for a in range(E.N_ACTIONS))
        lines.append(
            f"TURN {i:03d}/{EXPECTED_TURNS:03d} | scenario={t['sid']} | family={t['family']} | "
            f"return={t['return']:+.2f} | noop={t['noop']:+.2f} | heuristic={t['heuristic']:+.2f} | "
            f"oracle={t['oracle']:+.2f} | teacher_v2={t['teacher_v2']:+.2f} | clairvoyant={t['clairvoyant']:+.2f} | "
            f"success={t['success']:.2f} | progress={t['progress']:.1f}/{E.HORIZON} | overflows={t['overflows']} | "
            f"mean_health={t['mean_health']:.3f} | oracle_agree={t['oracle_agree']:.3f} | actions={acts}"
        )
    for name in E.FAMILIES:
        if name not in family_stats:
            continue
        f = family_stats[name]
        ts = fam[name]
        lines.append(
            f"FAMILY | {name} | turns={len(ts)} | return={f['return']:+.2f} | heuristic={f['heuristic']:+.2f} | "
            f"oracle={f['oracle']:+.2f} | teacher_v2={f['teacher_v2']:+.2f} | clairvoyant={f['clairvoyant']:+.2f} | "
            f"success={f['success']:.2f} | overflow_rate={sum(t['overflowed'] for t in ts) / (len(ts) * E.SEEDS_PER_TURN):.2f}"
        )
    dist = " ".join(f"{E.ACTION_ABBR[a]}={tot_actions[a]}" for a in range(E.N_ACTIONS))
    lines.append(
        f"SUMMARY | sim_score={metrics['sim_score']:.2f} | mean_return={mr:+.3f} | noop_return={mn:+.3f} | "
        f"heuristic_return={mh:+.3f} | oracle_return={mo:+.3f} | teacher_v2_return={m2:+.3f} | "
        f"clairvoyant_return={mc:+.3f} | success_rate={metrics['success_rate']:.3f} | "
        f"overflow_rate={metrics['overflow_rate']:.3f} | mean_health={metrics['mean_health']:.3f} | "
        f"oracle_agreement={metrics['oracle_agreement']:.3f} | noop_share={metrics['noop_share']:.3f} | "
        f"action_entropy={entropy:.3f} | actions={dist}"
    )
    lines.append("RUNTIME | " + " | ".join(f"{k}={v:.2f}" for k, v in timers.items()))
    if confirm:
        keep = ("sim_score", "mean_return", "noop_return", "heuristic_return", "clairvoyant_return", "success_rate",
                "overflow_rate", "turn_returns", "family", "effective_config", "runtime_seconds")
        out = {k: metrics[k] for k in keep}
        out.update({"set": "confirmation", "scenarios": n_scen})
        (exp_dir / "confirm_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return 0
    (exp_dir / "simulation.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (exp_dir / "sim_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    _args = sys.argv[3:]
    _export = Path(_args[_args.index("--export") + 1]) if "--export" in _args else None
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), fingerprint_only="--fingerprint" in _args,
                  confirm="--confirm" in _args, export_dir=_export))
