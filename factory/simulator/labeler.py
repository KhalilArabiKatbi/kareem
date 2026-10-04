"""Teacher labelling for per-experiment DAgger, run as its own light subprocess.

    python -m factory.simulator.labeler <jobs.pkl> <out.pkl>

jobs.pkl = {"teacher": "v2", "workers": 6, "episodes": [(sid, family, seed, [a_0, a_1, ...]), ...]}
out.pkl  = list (one per episode, same order) of per-step teacher value vectors.

The states are rebuilt by replaying each episode's actions from its seed, so only (seed, actions)
cross the process boundary. The module imports nothing heavier than the environment, so the worker
processes start fast and stay small. Results are returned in job order: deterministic regardless of
scheduling.
"""
from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

from . import env as E


def _label_episode(args):
    sid, family, seed, actions, teacher = args
    sc = E.Scenario(sid, family, seed)
    s = E.EnvState()
    out = []
    for a in actions:
        out.append(E.oracle_values(s, sc, teacher))
        E.step(s, a, sc)
    return out


def main(jobs_path: str, out_path: str) -> int:
    with open(jobs_path, "rb") as f:
        jobs = pickle.load(f)
    teacher = jobs["teacher"]
    args = [(sid, fam, seed, acts, teacher) for sid, fam, seed, acts in jobs["episodes"]]
    workers = max(1, int(jobs.get("workers", 4)))
    if workers == 1:
        result = [_label_episode(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            result = list(ex.map(_label_episode, args, chunksize=2))
    with open(out_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
