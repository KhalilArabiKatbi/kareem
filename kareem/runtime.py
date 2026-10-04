"""Context-health intervention policy runtime (needs only PyTorch).

Seeded from the factory's export (workspace/deploy/policy_runtime.py) and extended
with a typed Decision result and confidence gating.

    from kareem.runtime import InterventionPolicy
    policy = InterventionPolicy.load("path/to/weights")
    decision = policy.decide(history)                 # Decision(action, confidence, ...)
    action = decision.action                          # Action enum
    probs = policy.probabilities(history)             # {"NOOP": 0.71, ...}

`history` is the list of observation dicts for the current agent session, oldest first,
with history[-1] describing "now" (schema in kareem/observe.py). Call once per agent
step, after appending the newest observation, and apply the returned intervention
before the agent's next step.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .actions import ACTION_NAMES, Action


@dataclass
class Decision:
    """One intervention recommendation for the current step.

    `action` is the argmax recommendation. `confidence` is the softmax probability of
    that action. `deferred` is True when `confidence` fell below the `min_confidence`
    passed to decide(); the recommended action is still reported, and the caller is
    expected to fall back to NOOP (or a human/default rule) instead.
    """

    action: Action
    confidence: float
    probabilities: dict[str, float] = field(repr=False)
    deferred: bool = False

    @property
    def action_name(self) -> str:
        return self.action.name


class InterventionPolicy:
    def __init__(self, module, members, mu, sd, meta):
        self.module = module
        self.members = members
        self.mu, self.sd = mu, sd
        self.meta = meta
        self.feature_names = list(module.FEATURE_NAMES)

    @classmethod
    def load(cls, directory: str | Path) -> "InterventionPolicy":
        directory = Path(directory)
        spec = importlib.util.spec_from_file_location("policy_model", directory / "policy_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        blob = torch.load(directory / "intervention_policy.pt", map_location="cpu", weights_only=False)
        if blob.get("format") != "context-health-intervention-policy/1":
            raise ValueError("unknown policy file format")
        if list(module.FEATURE_NAMES) != blob["feature_names"]:
            raise ValueError("policy_model.py does not match the saved weights (feature names differ)")
        members = []
        for state in blob["members"]:
            net = module.build_model(len(blob["feature_names"]), len(ACTION_NAMES))
            net.load_state_dict(state)
            net.eval()
            members.append(net)
        meta = {k: v for k, v in blob.items() if k not in ("members", "mu", "sd")}
        card_path = directory / "policy_card.json"
        if card_path.exists():
            # card carries provenance and measured scores; blob fields win on conflict
            meta = {**json.loads(card_path.read_text(encoding="utf-8")), **meta}
        return cls(module, members, blob["mu"], blob["sd"], meta)

    def logits(self, histories: list[list[dict]]) -> torch.Tensor:
        rows = [[float(v) for v in self.module.featurize(h)] for h in histories]
        x = (torch.tensor(rows, dtype=torch.float32) - self.mu) / self.sd
        with torch.no_grad():
            return torch.stack([m(x) for m in self.members]).mean(0)

    def decide_batch(self, histories: list[list[dict]]) -> list[str]:
        return [ACTION_NAMES[i] for i in self.logits(histories).argmax(1).tolist()]

    def decide(self, history: list[dict], min_confidence: float = 0.0) -> Decision:
        """Recommend an intervention for the current step.

        With min_confidence > 0, the decision is marked `deferred` when the policy's
        confidence in its argmax action falls below the threshold — the analog of
        escalating to a human/default rule instead of acting on a weak signal.
        """
        probs = torch.softmax(self.logits([history])[0], 0)
        conf, idx = probs.max(0)
        conf = float(conf)
        probabilities = {a: round(float(p), 4) for a, p in zip(ACTION_NAMES, probs.tolist())}
        return Decision(
            action=Action(int(idx)),
            confidence=round(conf, 4),
            probabilities=probabilities,
            deferred=conf < min_confidence,
        )

    def probabilities(self, history: list[dict]) -> dict[str, float]:
        p = torch.softmax(self.logits([history])[0], 0).tolist()
        return {a: round(v, 4) for a, v in zip(ACTION_NAMES, p)}
