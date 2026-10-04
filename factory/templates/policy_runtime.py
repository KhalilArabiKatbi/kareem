"""Context-health intervention policy: standalone runtime (needs only PyTorch).

    from policy_runtime import InterventionPolicy
    policy = InterventionPolicy.load("path/to/deploy")
    action = policy.decide(history)            # "NOOP", "COMPACT", ...
    probs = policy.probabilities(history)      # {"NOOP": 0.71, ...}

`history` is the list of observation dicts for the current agent session, oldest first, with
history[-1] describing "now" (schema in README.md). Call once per agent step, after appending the
newest observation, and apply the returned intervention before the agent's next step.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ACTIONS = ("NOOP", "COMPACT", "PRUNE_TOOLS", "REINJECT_INSTRUCTIONS", "CHECKPOINT_RESET", "RETRIEVE_MEMORY")


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
            net = module.build_model(len(blob["feature_names"]), len(ACTIONS))
            net.load_state_dict(state)
            net.eval()
            members.append(net)
        meta = {k: v for k, v in blob.items() if k not in ("members", "mu", "sd")}
        return cls(module, members, blob["mu"], blob["sd"], meta)

    def logits(self, histories: list[list[dict]]) -> torch.Tensor:
        rows = [[float(v) for v in self.module.featurize(h)] for h in histories]
        x = (torch.tensor(rows, dtype=torch.float32) - self.mu) / self.sd
        with torch.no_grad():
            return torch.stack([m(x) for m in self.members]).mean(0)

    def decide_batch(self, histories: list[list[dict]]) -> list[str]:
        return [ACTIONS[i] for i in self.logits(histories).argmax(1).tolist()]

    def decide(self, history: list[dict]) -> str:
        return self.decide_batch([history])[0]

    def probabilities(self, history: list[dict]) -> dict[str, float]:
        p = torch.softmax(self.logits([history])[0], 0).tolist()
        return {a: round(v, 4) for a, v in zip(ACTIONS, p)}
