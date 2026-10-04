"""Unit tests for the deterministic pieces: router, audit, done checker, isolation, state space, code gate."""
from __future__ import annotations

import itertools

import pytest

from factory import done_checker
from factory.code_guard import check_model_source
from factory.errors import IsolationViolation, MetaProposalInvalid, RoutingViolationError, WorkerCodeError
from factory.isolation import IsolationGuard, paperize, sanitize_obj, scrub_foreign
from factory.router import EscalationTracker, ModelRouter, RouteContext
from factory.router_audit import expected_route
from factory.state_space import StateSpace
from factory.tests.mock_llm import BANNED_MODEL, GOOD_MODEL


# ---- router ---------------------------------------------------------------------------------
def route(agent, purpose="primary", **kw):
    return ModelRouter().route(RouteContext(iteration=0, agent=agent, purpose=purpose, **kw))


@pytest.mark.parametrize("agent,purpose", [("meta", "primary"), ("worker", "primary"), ("judge", "primary"),
                                           ("doc", "draft"), ("doc", "final")])
def test_rule1_baseline(agent, purpose):
    d = route(agent, purpose)
    assert (d.model, d.effort) == ("sonnet-5", "standard")


def test_rule2_micro_retry_forces_high_effort():
    d = route("worker", step_retry=1)
    assert (d.model, d.effort) == ("sonnet-5", "high")
    assert "R2 micro-retry" in d.rules


@pytest.mark.parametrize("agent", ["worker", "judge"])
def test_rule3_quality_escalation(agent):
    assert (route(agent, last_judge_score=59).model, route(agent, last_judge_score=59).effort) == ("opus-5.5", "standard")
    assert route(agent, last_judge_score=60).model == "sonnet-5"
    assert route("meta", last_judge_score=10).model == "sonnet-5"  # rule 3 only covers worker + judge


def test_rule4_complexity_escalation():
    assert route("meta", meta_invalid_streak=1).model == "sonnet-5"
    d = route("meta", meta_invalid_streak=2)
    assert (d.model, d.effort) == ("opus-5.5", "high")


def test_rule5_fable_only_for_doc_format():
    d = route("doc", "format")
    assert d.model == "fable-5.1"
    for agent in ("meta", "worker", "judge"):
        with pytest.raises(RoutingViolationError):
            ModelRouter(overrides={agent: "fable-5.1"}).route(RouteContext(iteration=0, agent=agent))
    with pytest.raises(RoutingViolationError):
        ModelRouter(overrides={"doc": "fable-5.1"}).route(RouteContext(iteration=0, agent="doc", purpose="draft"))
    with pytest.raises(RoutingViolationError):
        ModelRouter().route(RouteContext(iteration=0, agent="simulator"))


def test_rule6_iteration_escalation_caps_at_opus_high_and_is_per_iteration():
    t, other = EscalationTracker(), EscalationTracker()
    levels = [t.escalate("worker") for _ in range(5)]
    assert levels == [1, 2, 3, 3, 3]
    d = ModelRouter().route(RouteContext(iteration=0, agent="worker", escalation=t.escalation("worker")))
    assert (d.model, d.effort) == ("opus-5.5", "high")
    assert other.escalation("worker") == 0  # parallel iterations never share escalation state


def test_audit_matches_router_on_full_grid():
    r = ModelRouter()
    grid = itertools.product(
        [("meta", "primary"), ("worker", "primary"), ("judge", "primary"), ("doc", "draft"), ("doc", "format"),
         ("doc", "final")],
        [0, 1], [0, 1, 2, 3], [None, 30, 59, 60, 95], [0, 1, 2, 5],
    )
    n = 0
    for (agent, purpose), retry, esc, score, streak in grid:
        ctx = RouteContext(iteration=1, agent=agent, purpose=purpose, step_retry=retry, escalation=esc,
                           last_judge_score=score, meta_invalid_streak=streak)
        d = r.route(ctx)
        assert (d.model, d.effort) == expected_route(ctx), ctx
        n += 1
    assert n == 6 * 2 * 4 * 5 * 4


# ---- done checker ---------------------------------------------------------------------------
def state(n, scores=None, last_added=-1):
    return {"next_iteration": n, "scores": {str(k): v for k, v in (scores or {}).items()},
            "last_dimension_added_iteration": last_added}


def test_done_max_iterations():
    assert done_checker.check(state(5), 5).startswith("max-iterations")
    assert done_checker.check(state(4, last_added=3), 5) is None


def test_done_state_space_stagnant():
    assert done_checker.check(state(10, last_added=-1), 50).startswith("state space stagnant")
    assert done_checker.check(state(10, last_added=0), 50) is None
    assert done_checker.check(state(15, {i: i for i in range(15)}, last_added=4), 50).startswith("state space stagnant")


def test_done_score_plateau():
    flat = {i: 70 for i in range(12)}
    flat[11] = 72  # +2.0 is not "> 2.0"
    assert done_checker.check(state(12, flat, last_added=11), 50).startswith("score plateau")
    better = dict(flat)
    better[11] = 72.5
    assert done_checker.check(state(12, better, last_added=11), 50) is None
    # no scores at all in the window counts as no improvement
    assert done_checker.check(state(12, {0: 50}, last_added=11), 50).startswith("score plateau")


# ---- isolation ------------------------------------------------------------------------------
def test_guard_blocks_foreign_experiment_ids_and_canaries():
    g = IsolationGuard({"exp_002": "CANARY-exp_002-aaaaaaaa", "exp_005": "CANARY-exp_005-bbbbbbbb"})
    ok = g.check("worker", "exp_005", ["exp/next_coordinate.json"], "exp_005 CANARY-exp_005-bbbbbbbb")
    assert ok["verdict"] == "ISOLATED"
    with pytest.raises(IsolationViolation):
        g.check("worker", "exp_005", ["exp/next_coordinate.json"], "data from exp_002")
    with pytest.raises(IsolationViolation):
        g.check("judge", "exp_005", ["exp/simulation.log"], "CANARY-exp_002-aaaaaaaa")
    with pytest.raises(IsolationViolation):
        g.check("worker", "exp_005", ["experiments/exp_002/model.py"], "clean text")
    with pytest.raises(IsolationViolation):
        g.check("doc_final", None, ["workspace/research_paper.md"], "exp_000")


def test_sanitizers():
    obj = {"experiment": "exp_003", "canary": "CANARY-exp_003-12345678", "note": "beats exp_001", "n": 3}
    s = sanitize_obj(obj)
    assert "experiment" not in s and "canary" not in s and "exp_" not in s["note"]
    assert scrub_foreign({"a": "exp_004 vs exp_009"}, "exp_004") == {"a": "exp_004 vs another experiment"}
    assert paperize("see exp_007") == "see Iteration 007"


# ---- state space ----------------------------------------------------------------------------
SEED = {"dimensions": {"seed_dim": {"type": "categorical", "values": [[64, 64]], "description": "seed dimension x"}},
        "ledger": []}


def test_state_space_add_and_coordinate():
    sp = StateSpace(SEED)
    prop = {"mutations": [{"op": "add_dimension", "name": "width_mult",
                           "spec": {"type": "float", "min": 0.5, "max": 2.0, "description": "width multiplier xx"}}],
            "next_coordinate": {"seed_dim": [64, 64], "width_mult": 1.5}}
    new, coord, rep = sp.apply_proposal(prop, 0)
    assert rep["dimensions_added"] == ["width_mult"] and coord == {"seed_dim": [64, 64], "width_mult": 1.5}
    assert "width_mult" not in sp.dimensions  # original untouched
    new.record(0, coord, "scored", 70)
    with pytest.raises(MetaProposalInvalid, match="already evaluated"):
        new.apply_proposal({"mutations": [], "next_coordinate": coord}, 1)


@pytest.mark.parametrize("prop,msg", [
    ({"mutations": [{"op": "add_dimension", "name": "Bad Name", "spec": {}}], "next_coordinate": {}}, "name must match"),
    ({"mutations": [], "next_coordinate": {"seed_dim": [32]}}, "outside its domain"),
    ({"mutations": [], "next_coordinate": {}}, "missing"),
    ({"mutations": [{"op": "prune_dimension", "name": "seed_dim"}], "next_coordinate": {}}, "last active"),
    ({"mutations": [{"op": "explode", "name": "seed_dim"}], "next_coordinate": {}}, "unknown op"),
])
def test_state_space_rejects_invalid(prop, msg):
    with pytest.raises(MetaProposalInvalid, match=msg):
        StateSpace(SEED).apply_proposal(prop, 0)


def test_poisoned_coordinate_rejected():
    sp = StateSpace({**SEED, "ledger": []})
    sp.record(0, {"seed_dim": [64, 64]}, "poisoned", None)
    with pytest.raises(MetaProposalInvalid, match="poisoned"):
        sp.apply_proposal({"mutations": [], "next_coordinate": {"seed_dim": [64, 64]}}, 1)


# ---- code gate ------------------------------------------------------------------------------
def test_code_gate():
    assert "featurize" in check_model_source(GOOD_MODEL)["top_level"]
    with pytest.raises(WorkerCodeError, match="import of 'os'"):
        check_model_source(BANNED_MODEL)
    with pytest.raises(WorkerCodeError, match="call to 'open'"):
        check_model_source(GOOD_MODEL + "\nx = open('f')\n")
    with pytest.raises(WorkerCodeError, match="build_model"):
        check_model_source(GOOD_MODEL.replace("def build_model", "def make_model"))
    with pytest.raises(WorkerCodeError, match="does not compile"):
        check_model_source(GOOD_MODEL + "\ndef broken(:\n")


def test_wave_proposal_needs_exact_distinct_candidates():
    sp = StateSpace(SEED)
    add = [{"op": "add_dimension", "name": "act", "spec": {"type": "categorical", "values": ["relu", "gelu", "silu"],
                                                           "description": "activation function"}}]
    ok = {"mutations": add, "candidates": [{"next_coordinate": {"seed_dim": [64, 64], "act": a}, "hypothesis": "h"}
                                           for a in ("relu", "gelu", "silu")]}
    new, coords, rep = sp.apply_wave_proposal(ok, [4, 5, 6])
    assert [c["act"] for c in coords] == ["relu", "gelu", "silu"] and len(set(rep["coordinate_hashes"])) == 3
    assert new.data["last_dimension_added_iteration"] == 4
    with pytest.raises(MetaProposalInvalid, match="exactly 2"):
        sp.apply_wave_proposal(ok, [4, 5])
    dup = {"mutations": add, "candidates": [ok["candidates"][0], ok["candidates"][0], ok["candidates"][1]]}
    with pytest.raises(MetaProposalInvalid, match="duplicates"):
        sp.apply_wave_proposal(dup, [4, 5, 6])
