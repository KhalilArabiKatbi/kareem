"""Simulator protocol tests: determinism, 50 TURN lines x 4 seeds, teacher/target/data selection,
loss_fn with per-action values, seed ensembles, ignored-setting notes, the functional fingerprint and
the constant-feature validity gate."""
from __future__ import annotations

import json
import re

import pytest

from factory.errors import SimulatorError
from factory.simulator import harness
from factory.tests.mock_llm import GOOD_MODEL

CANARY = "CANARY-exp_950-0badc0de"
NL = "\n"


def make_exp(tmp_path, name, extra=""):
    d = tmp_path / name
    d.mkdir()
    (d / "model.py").write_text(GOOD_MODEL + NL + extra, encoding="utf-8")
    (d / "config.json").write_text(json.dumps({"experiment": "exp_950", "canary": CANARY}), encoding="utf-8")
    return d


def stable(metrics: dict) -> dict:
    m = json.loads(json.dumps(metrics))
    for k in ("runtime_seconds", "wall_seconds", "timers"):
        m.pop(k, None)
    for k in ("featurize_seconds", "train_seconds"):
        m["training"].pop(k, None)
    return m


def test_soft_targets_q_loss_data_and_ensembles_are_deterministic(tmp_path):
    harness.ensure_cache()
    extra = NL.join([
        'TRAIN_CONFIG = dict(TRAIN_CONFIG, teacher="v2", data="behaviour+onpolicy", ensemble_seeds=2)',
        "def loss_fn(logits, targets, q):",
        "    import torch",
        "    return -(torch.softmax(q / 0.5, 1) * torch.log_softmax(logits, 1)).sum(1).mean()",
        'FINETUNE = {"params": "output_bias"}',
        "",
    ])
    a = harness.run_simulator(make_exp(tmp_path, "a", extra), CANARY)
    b = harness.run_simulator(make_exp(tmp_path, "b", extra), CANARY)
    assert stable(a.metrics) == stable(b.metrics)
    t = a.metrics["training"]
    assert t["teacher"] == "v2" and t["data"] == "behaviour+onpolicy" and t["ensemble_seeds"] == 2
    assert t["custom_loss_uses_q"] and t["samples"] > 15000
    assert any("FINETUNE is no longer supported" in n for n in a.metrics["notes"])
    assert any("loss_fn replaces the built-in target" in n for n in a.metrics["notes"])
    assert len(re.findall(r"(?m)^TURN \d{3}/050 \| ", a.log_text)) == 50
    assert a.metrics["scenarios"] == 200 and len(a.metrics["turn_returns"]) == 50
    strip = lambda s: re.sub(r"(?m)^RUNTIME \|.*$", "", s)  # noqa: E731
    assert strip(a.log_text) == strip(b.log_text)


def test_ignored_settings_are_reported(tmp_path):
    extra = 'TRAIN_CONFIG = dict(TRAIN_CONFIG, target="soft", class_weighting="balanced", label_smoothing=0.1)' + NL
    m = harness.run_simulator(make_exp(tmp_path, "ign", extra), CANARY).metrics
    assert any("class_weighting=balanced has no effect" in n for n in m["notes"])
    assert any("label_smoothing has no effect" in n for n in m["notes"])
    assert m["effective_config"]["class_weighting"] is None


def test_builtin_targets_and_teachers_change_training(tmp_path):
    runs = {}
    variants = {"hard_v1": "", "soft_v1": 'TRAIN_CONFIG = dict(TRAIN_CONFIG, target="soft")' + NL,
                "hard_v2": 'TRAIN_CONFIG = dict(TRAIN_CONFIG, teacher="v2")' + NL}
    for name, extra in variants.items():
        runs[name] = harness.run_simulator(make_exp(tmp_path, name, extra), CANARY).metrics
    assert runs["hard_v1"]["training"]["target"] == "hard" and runs["soft_v1"]["training"]["target"] == "soft"
    assert runs["hard_v2"]["training"]["teacher"] == "v2"
    assert len({r["training"]["final_loss"] for r in runs.values()}) == 3


def test_fingerprint_ignores_cosmetic_changes_and_tracks_real_ones(tmp_path):
    base = harness.run_fingerprint(make_exp(tmp_path, "fa"), CANARY)["fingerprint"]
    cosmetic = "# a comment only" + NL + "TRAIN_CONFIG = dict(TRAIN_CONFIG)" + NL
    assert harness.run_fingerprint(make_exp(tmp_path, "fb", cosmetic), CANARY)["fingerprint"] == base
    noop = 'TRAIN_CONFIG = dict(TRAIN_CONFIG, target="hard", target_temperature=3.0)' + NL  # tau ignored under hard
    assert harness.run_fingerprint(make_exp(tmp_path, "fc", noop), CANARY)["fingerprint"] == base
    real = "TRAIN_CONFIG = dict(TRAIN_CONFIG, lr=0.01)" + NL
    assert harness.run_fingerprint(make_exp(tmp_path, "fd", real), CANARY)["fingerprint"] != base


def test_constant_features_fail_the_validity_gate(tmp_path):
    broken = NL.join([
        "_orig_featurize = featurize",
        "def featurize(history):",
        "    v = _orig_featurize(history)",
        "    return [v[0]] + [0.0] * (len(v) - 1)",
        "",
    ])
    with pytest.raises(SimulatorError, match="features are constant"):
        harness.run_fingerprint(make_exp(tmp_path, "const", broken), CANARY)


def test_self_dagger_is_deterministic_and_reports(tmp_path):
    extra = 'TRAIN_CONFIG = dict(TRAIN_CONFIG, teacher="v2", target="soft", data="behaviour+self", self_episodes=28)' + NL
    a = harness.run_simulator(make_exp(tmp_path, "da", extra), CANARY)
    b = harness.run_simulator(make_exp(tmp_path, "db", extra), CANARY)
    sa, sb = stable(a.metrics), stable(b.metrics)
    for m in (sa, sb):
        for k in ("round1_seconds", "label_seconds", "label_workers"):
            m["dagger"].pop(k, None)
    assert sa == sb
    d = a.metrics["dagger"]
    assert d["self_episodes"] == 28 and d["self_states"] == 28 * 40 and d["teacher"] == "v2"
    assert a.metrics["training"]["samples"] == 8640 + 28 * 40
    assert "DAGGER | self_episodes=28" in a.log_text
    assert not list((tmp_path / "da").glob("_dagger_*.pkl"))  # temp files cleaned up


def test_early_stop_keeps_best_validation_checkpoint(tmp_path):
    extra = 'TRAIN_CONFIG = dict(TRAIN_CONFIG, teacher="v2", target="soft", epochs=30, early_stop="val_regret")' + NL
    a = harness.run_simulator(make_exp(tmp_path, "ea", extra), CANARY)
    b = harness.run_simulator(make_exp(tmp_path, "eb", extra), CANARY)
    assert stable(a.metrics) == stable(b.metrics)
    t = a.metrics["training"]
    assert t["early_stop"] == "val_regret" and len(t["best_epochs"]) == 1 and 1 <= t["best_epochs"][0] <= 30
    assert a.metrics["effective_config"]["early_stop"] == "val_regret"
    assert "early_stop=val_regret | kept_epochs=" in a.log_text
