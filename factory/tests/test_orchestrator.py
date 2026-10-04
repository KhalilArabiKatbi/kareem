"""End-to-end tests of the deterministic state machine with a scripted mock LLM.

The real simulator runs (subprocess, 60 s timeout); only the LLM is mocked. Workspace and
experiment paths are redirected to a temporary directory so the real run is never touched.
"""
from __future__ import annotations

import json
import re

import pytest

from factory import config, verify_run
from factory.errors import InfraUnavailable, IsolationViolation
from factory.init_workspace import initialize
from factory.router import ModelRouter
from factory.run_factory import FactoryOrchestrator
from factory.state_store import read_json
from factory.tests.mock_llm import MockLLM

PATH_ATTRS = {
    "WORKING_DIR": "workspace", "EXPERIMENTS_DIR": "experiments", "SANDBOX_DIR": "sandbox",
    "STATE_SPACE_PATH": "workspace/state_space.json", "FACTORY_STATE_PATH": "workspace/factory_state.json",
    "RESEARCH_PAPER_PATH": "workspace/research_paper.md", "BEST_MODEL_PATH": "workspace/best_model.py",
    "BEST_CONFIG_PATH": "workspace/best_config.json", "LAST_JUDGE_PATH": "workspace/last_judge.json",
    "FACTORY_LOG_PATH": "workspace/factory.log", "FACTORY_REPORT_PATH": "workspace/factory_report.json",
}


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    for attr, rel in PATH_ATTRS.items():
        monkeypatch.setattr(config, attr, tmp_path / rel)
    monkeypatch.setattr(config, "CONFIRM_TOP_K", 0)  # confirmation has its own test (it adds ~20 s per halt)
    initialize(fresh=False, log=lambda *_: None)
    return tmp_path


def make(llm, max_iterations, router=None, sim_timeout=config.SIM_TIMEOUT_S, parallel=1):
    return FactoryOrchestrator(max_iterations=max_iterations, llm=llm, router=router, sim_timeout=sim_timeout,
                               echo=False, parallel=parallel)


def routes(exp_idx, agent, purpose="primary"):
    log = read_json(config.exp_dir(exp_idx) / "router_log.json")
    return [(e["decision"]["model"], e["decision"]["effort"], e["outcome"], e["decision"]["rules"])
            for e in log["decisions"] if e["context"]["agent"] == agent and e["context"]["purpose"] == purpose]


def test_happy_path_then_halt(ws):
    llm = MockLLM()
    assert make(llm, 2).run() == 0
    st = read_json(config.FACTORY_STATE_PATH)
    assert st["halted"]["reason"].startswith("max-iterations reached")
    assert [r["status"] for r in st["iterations"]] == ["scored", "scored"]
    for i in (0, 1):
        chk = verify_run.check_experiment(i)
        assert chk["micro_done"], chk
        turns = re.findall(r"(?m)^TURN ", (config.exp_dir(i) / "simulation.log").read_text(encoding="utf-8"))
        assert len(turns) == 50
    # Fable is used by doc/format only, everything else starts on sonnet-5/standard
    assert {(c["key"], c["model"]) for c in llm.calls if c["model"] == "fable-5.1"} == {("doc/format", "fable-5.1")}
    assert all(c["model"] == "sonnet-5" for c in llm.calls if c["key"] != "doc/format")
    paper = config.RESEARCH_PAPER_PATH.read_text(encoding="utf-8")
    assert "### Iteration 000: Mock study" in paper and "### Iteration 001: Mock study" in paper
    assert "*Status: complete." in paper and "exp_" not in paper
    assert config.BEST_MODEL_PATH.exists()
    space = read_json(config.STATE_SPACE_PATH)
    assert len(space["dimensions"]) == 3  # seed + 2 meta-created
    assert "exp_999" not in json.dumps(space)  # hallucinated id scrubbed at ingestion
    cfg = read_json(config.exp_dir(1) / "config.json")
    assert "exp_123" not in json.dumps(cfg)
    rep = read_json(config.FACTORY_REPORT_PATH)
    assert rep["micro_done_summary"]["scored_all_checks_passed"] == 2
    for name in ("halted_by_meta_done_condition", "best_model_is_highest_scoring", "paper_complete",
                 "state_space_not_hardcoded", "no_raw_experiment_ids_in_paper"):
        assert rep["macro_done"]["checks"][name]["ok"], (name, rep["macro_done"]["checks"][name])
    # A halted factory does nothing on rerun
    assert make(MockLLM(), 2).run() == 0


def test_micro_fail_retries_with_high_effort_and_meta_escalates(ws):
    llm = MockLLM({"worker/primary": ["bad_json"], "meta/primary": ["invalid_proposal", "invalid_proposal"],
                   "judge/primary": ["far_score"]})
    assert make(llm, 1).run() == 0
    w = routes(0, "worker")
    assert w[0][:3] == ("sonnet-5", "standard", "failed") and w[1][:3] == ("sonnet-5", "high", "ok")
    m = routes(0, "meta")
    # two invalid proposals: in-place retry at high effort, then the iteration retries meta at opus-5.5/high (R4)
    assert m[0][:3] == ("sonnet-5", "standard", "failed")
    assert m[1][:3] == ("sonnet-5", "high", "failed")
    assert m[2][:2] == ("opus-5.5", "high") and m[2][2] == "ok"
    assert any(r.startswith("R4") for r in m[2][3])
    j = routes(0, "judge")
    assert j[0][2] == "failed" and j[1][:3] == ("sonnet-5", "high", "ok")
    assert verify_run.check_experiment(0)["micro_done"]


def test_three_simulator_failures_poison_the_coordinate(ws):
    llm = MockLLM({"worker/primary": ["crash_code"] * 3})
    assert make(llm, 2).run() == 0
    st = read_json(config.FACTORY_STATE_PATH)
    assert [r["status"] for r in st["iterations"]] == ["poisoned", "scored"]
    assert st["poisoned"][0]["iteration"] == 0 and st["poisoned"][0]["coordinate"] is not None
    attempts = read_json(config.exp_dir(0) / "attempts.json")
    assert [a["error_type"] for a in attempts] == ["SimulatorError"] * 3
    w = routes(0, "worker")  # each iteration-level failure bumps the worker one ladder level
    assert [x[:2] for x in w] == [("sonnet-5", "standard"), ("sonnet-5", "high"), ("opus-5.5", "standard")]
    chk = verify_run.check_experiment(0)
    assert chk["status"] == "poisoned" and chk["micro_fail_recovered"], chk
    space = read_json(config.STATE_SPACE_PATH)
    assert [e["status"] for e in space["ledger"]] == ["poisoned", "scored"]
    assert "### Iteration 000: Poisoned coordinate" in config.RESEARCH_PAPER_PATH.read_text(encoding="utf-8")


def test_fable_routed_to_worker_is_a_routing_violation(ws):
    llm = MockLLM()
    assert make(llm, 1, router=ModelRouter(overrides={"worker": "fable-5.1"})).run() == 0
    attempts = read_json(config.exp_dir(0) / "attempts.json")
    assert [a["error_type"] for a in attempts] == ["RoutingViolationError"] * 3
    assert not any(c["key"] == "worker/primary" for c in llm.calls)  # the call never left the harness
    st = read_json(config.FACTORY_STATE_PATH)
    assert st["iterations"][0]["status"] == "poisoned"


def test_simulator_timeout_is_a_micro_fail(ws):
    llm = MockLLM()
    assert make(llm, 1, sim_timeout=1).run() == 0
    attempts = read_json(config.exp_dir(0) / "attempts.json")
    assert [a["error_type"] for a in attempts] == ["SimulatorTimeout"] * 3


def test_low_judge_score_escalates_next_worker_and_judge(ws):
    llm = MockLLM({"worker/primary": ["weak"]})
    assert make(llm, 2).run() == 0
    st = read_json(config.FACTORY_STATE_PATH)
    assert st["scores"]["0"] < 60
    assert routes(1, "worker")[0][:2] == ("opus-5.5", "standard")
    assert routes(1, "judge")[0][:2] == ("opus-5.5", "standard")
    assert routes(1, "meta")[0][:2] == ("sonnet-5", "standard")


def test_banned_worker_code_rejected_before_execution(ws):
    llm = MockLLM({"worker/primary": ["banned"]})
    assert make(llm, 1).run() == 0
    w = routes(0, "worker")
    assert w[0][2] == "failed" and w[1][2] == "ok"


def test_stagnant_state_space_halts(ws, monkeypatch):
    monkeypatch.setattr(config, "STAGNATION_WINDOW", 2)
    from factory import done_checker
    monkeypatch.setattr(done_checker.check, "__defaults__", (2, config.MIN_IMPROVEMENT))
    llm = MockLLM({"meta/primary": ["no_mutation"] * 10})
    assert make(llm, 10).run() == 0
    st = read_json(config.FACTORY_STATE_PATH)
    assert st["halted"]["reason"].startswith("state space stagnant")
    assert st["next_iteration"] == 2


def test_isolation_violation_aborts(ws):
    llm = MockLLM()
    orch = make(llm, 1)
    orch.state["canaries"]["exp_007"] = "CANARY-exp_007-00000000"
    # Simulate a harness bug that leaks another experiment's data into the Meta payload.
    config.BEST_CONFIG_PATH.write_text(json.dumps({"leak": "CANARY-exp_007-00000000"}), encoding="utf-8")
    from factory import isolation
    orig = isolation.sanitize_obj
    import factory.run_factory as rf
    rf.sanitize_obj = lambda o, *a: o  # disable the sanitiser to prove the guard is the last line of defence
    try:
        with pytest.raises(IsolationViolation):
            orch.run()
    finally:
        rf.sanitize_obj = orig


def test_resume_after_infra_outage(ws):
    llm = MockLLM({"doc/draft": ["ok", "infra"]})
    with pytest.raises(InfraUnavailable):
        make(llm, 3).run()
    st = read_json(config.FACTORY_STATE_PATH)
    assert st["next_iteration"] == 1 and config.exp_dir(1).exists()
    assert make(MockLLM(), 3).run() == 0
    st = read_json(config.FACTORY_STATE_PATH)
    assert [r["iteration"] for r in st["iterations"]] == [0, 1, 2]
    assert list((config.EXPERIMENTS_DIR / "_aborted").iterdir())
    space = read_json(config.STATE_SPACE_PATH)
    assert [e["iteration"] for e in space["ledger"]] == [0, 1, 2]
    for i in range(3):
        assert verify_run.check_experiment(i)["micro_done"]


# ---- parallel waves ---------------------------------------------------------------------------
def _outcome():
    st = read_json(config.FACTORY_STATE_PATH)
    space = read_json(config.STATE_SPACE_PATH)
    return (st["scores"], [(e["iteration"], e["coordinate_hash"], e["status"]) for e in space["ledger"]],
            [(r["iteration"], r["status"], r["wave"]) for r in st["iterations"]])


def test_parallel_waves_are_deterministic_and_complete(ws, tmp_path, monkeypatch):
    assert make(MockLLM(), 6, parallel=4).run() == 0
    st = read_json(config.FACTORY_STATE_PATH)
    assert [(r["iteration"], r["wave"]) for r in st["iterations"]] == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1)]
    assert st["halted"]["reason"].startswith("max-iterations reached (6/6)")
    for i in range(6):
        assert verify_run.check_experiment(i)["micro_done"], i
    # every member's router log carries the shared Meta decision
    for i in range(4):
        assert routes(i, "meta")[0][:3] == ("sonnet-5", "standard", "ok")
    first = _outcome()
    an = read_json(config.STATE_SPACE_PATH)["analytics"]
    assert [r["iteration"] for r in an["rows"]] == list(range(6))
    assert an["marginals"] and "mockdim_01" in an["marginals"]
    assert an["paired_vs_best"] and all("se_sim" in p for p in an["paired_vs_best"])
    assert "exp_" not in json.dumps(an)
    paper = config.RESEARCH_PAPER_PATH.read_text(encoding="utf-8")
    assert all(f"### Iteration {i:03d}: Mock study" in paper for i in range(6))

    # same inputs, second run in a fresh workspace: identical scores, ledger and wave layout
    other = tmp_path / "again"
    for attr, rel in PATH_ATTRS.items():
        monkeypatch.setattr(config, attr, other / rel)
    initialize(fresh=False, log=lambda *_: None)
    assert make(MockLLM(), 6, parallel=4).run() == 0
    assert _outcome() == first


def test_parallel_member_failure_is_isolated(ws):
    llm = MockLLM({"worker/primary@exp_001": ["crash_code"] * 3, "judge/primary@exp_002": ["far_score"]})
    assert make(llm, 3, parallel=3).run() == 0
    st = read_json(config.FACTORY_STATE_PATH)
    assert [r["status"] for r in st["iterations"]] == ["scored", "poisoned", "scored"]
    # exp_001's escalations stayed inside exp_001
    assert [x[:2] for x in routes(1, "worker")] == [("sonnet-5", "standard"), ("sonnet-5", "high"), ("opus-5.5", "standard")]
    assert [x[:2] for x in routes(0, "worker")] == [("sonnet-5", "standard")]
    assert [x[:3] for x in routes(2, "judge")] == [("sonnet-5", "standard", "failed"), ("sonnet-5", "high", "ok")]
    assert verify_run.check_experiment(1)["micro_fail_recovered"]


def test_quality_escalation_uses_last_score_of_previous_wave(ws):
    llm = MockLLM({"worker/primary@exp_001": ["weak"]})
    assert make(llm, 4, parallel=2).run() == 0
    st = read_json(config.FACTORY_STATE_PATH)
    assert st["scores"]["1"] < 60 <= st["scores"]["0"]
    for i in (2, 3):
        assert routes(i, "worker")[0][:2] == ("opus-5.5", "standard")
        assert routes(i, "judge")[0][:2] == ("opus-5.5", "standard")


def test_parallel_resume_after_outage(ws):
    llm = MockLLM({"doc/draft@exp_003": ["infra"]})
    with pytest.raises(InfraUnavailable):
        make(llm, 4, parallel=2).run()
    st = read_json(config.FACTORY_STATE_PATH)
    assert st["next_iteration"] == 2
    assert make(MockLLM(), 4, parallel=2).run() == 0
    st = read_json(config.FACTORY_STATE_PATH)
    assert [r["iteration"] for r in st["iterations"]] == [0, 1, 2, 3]
    for i in range(4):
        assert verify_run.check_experiment(i)["micro_done"], i


# ---- infrastructure faults must pause, never poison -------------------------------------------
def test_process_launch_oserror_pauses_without_poisoning(ws):
    llm = MockLLM({"meta/primary": ["ok", "oserror"]})
    with pytest.raises(InfraUnavailable):
        make(llm, 4, parallel=1).run()
    st = read_json(config.FACTORY_STATE_PATH)
    assert [r["status"] for r in st["iterations"]] == ["scored"] and not st["poisoned"]
    assert make(MockLLM(), 2, parallel=1).run() == 0  # resumes cleanly
    assert [r["status"] for r in read_json(config.FACTORY_STATE_PATH)["iterations"]] == ["scored", "scored"]


def test_circuit_breaker_pauses_after_two_fully_poisoned_waves(ws):
    llm = MockLLM({"worker/primary": ["crash_code"] * 12})
    with pytest.raises(InfraUnavailable, match="circuit breaker"):
        make(llm, 10, parallel=1).run()
    st = read_json(config.FACTORY_STATE_PATH)
    assert [r["status"] for r in st["iterations"]] == ["poisoned", "poisoned"]


def test_llm_client_oserror_becomes_infra_unavailable(monkeypatch):
    import subprocess
    from factory import llm_client
    from factory.router import ModelRouter, RouteContext
    monkeypatch.setattr(config, "INFRA_BACKOFF_S", (0,))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError(216, "bad exe")))
    client = llm_client.ClaudeCLIClient(log=lambda *_: None)
    d = ModelRouter().route(RouteContext(iteration=0, agent="judge"))
    with pytest.raises(InfraUnavailable, match="cannot be launched"):
        client.call(d, "sys", "user", None)


# ---- functional duplicates --------------------------------------------------------------------
def test_duplicate_models_are_refilled_then_marked(ws):
    # every Worker call returns the same model: iteration 1 duplicates iteration 0
    llm = MockLLM({"worker/primary": ["same"] * 20})
    assert make(llm, 3, parallel=1).run() == 0
    st = read_json(config.FACTORY_STATE_PATH)
    rows = {r["iteration"]: r for r in st["iterations"]}
    assert rows[0]["duplicate_of"] is None
    assert len(rows[1]["refills"]) == 2 and rows[1]["duplicate_of"] == 0  # refill cap 2, then recorded
    assert (config.exp_dir(1) / "next_coordinate.rejected_1.json").exists()
    assert any("Refill request" in p.read_text(encoding="utf-8") for p in (config.exp_dir(1) / "prompts").glob("meta_*.md"))
    an = read_json(config.STATE_SPACE_PATH)["analytics"]
    assert {"iteration": 1, "same_model_as": 0} in an["duplicates"] and 1 in an["invalid_iterations"]
    assert verify_run.check_experiment(1)["micro_done"]


# ---- hidden confirmation at halt ---------------------------------------------------------------
def test_hidden_confirmation_rescores_top_models(ws, monkeypatch):
    monkeypatch.setattr(config, "CONFIRM_TOP_K", 2)
    assert make(MockLLM(), 2, parallel=2).run() == 0
    conf = read_json(config.WORKING_DIR / "confirmation.json")
    assert [r["iteration"] for r in conf["rows"]] and len(conf["rows"]) == 2
    assert all("confirm_sim" in r for r in conf["rows"])
    assert conf["confirmed_best"] in (0, 1) and conf["selected_best"] in (0, 1)
    assert (config.WORKING_DIR / "best_model_confirmed.py").exists()
    for i in (0, 1):  # the scored artefacts are untouched; micro-done still holds
        assert (config.exp_dir(i) / "confirm_metrics.json").exists()
        assert verify_run.check_experiment(i)["micro_done"]
    paper = config.RESEARCH_PAPER_PATH.read_text(encoding="utf-8")
    assert "Confirmation sim_score" in paper and "exp_" not in paper
