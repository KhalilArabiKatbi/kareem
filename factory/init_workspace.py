"""Step 4: initialise WORKING_DIR with a one-dimension seed state space and the paper header."""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import time

from . import config
from .state_store import new_factory_state, read_json, write_json, write_text


def ai_docs_checksum() -> str:
    h = hashlib.sha256()
    for p in sorted(config.AI_DOCS_DIR.glob("**/*")):
        if p.is_file():
            h.update(p.relative_to(config.AI_DOCS_DIR).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def protect_ai_docs() -> None:
    for p in config.AI_DOCS_DIR.glob("**/*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def is_initialized() -> bool:
    return config.FACTORY_STATE_PATH.exists() and config.STATE_SPACE_PATH.exists()


def initialize(fresh: bool = False, log=print) -> None:
    config.WORKING_DIR.mkdir(parents=True, exist_ok=True)
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if fresh:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        archive = config.EXPERIMENTS_DIR / "_archived" / stamp
        moved = False
        for d in sorted(config.EXPERIMENTS_DIR.glob("exp_*")):
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(archive / d.name))
            moved = True
        ws_items = [p for p in config.WORKING_DIR.iterdir() if p.name != ".snapshots"]
        if ws_items:
            (archive / "workspace").mkdir(parents=True, exist_ok=True)
            for p in ws_items:
                shutil.move(str(p), str(archive / "workspace" / p.name))
            moved = True
        shutil.rmtree(config.WORKING_DIR / ".snapshots", ignore_errors=True)
        if moved:
            log(f"[init] previous run archived to {archive}")
    elif is_initialized():
        log("[init] workspace already initialised; keeping existing state")
        return

    seed = read_json(config.TEMPLATES_DIR / "state_space_seed.json")
    if len(seed["dimensions"]) != 1:
        raise RuntimeError("the seed state space must contain exactly one seed dimension")
    write_json(config.STATE_SPACE_PATH, seed)
    write_text(config.RESEARCH_PAPER_PATH, (config.TEMPLATES_DIR / "research_paper_header.md").read_text(encoding="utf-8"))
    write_json(config.BEST_CONFIG_PATH, {"status": "no experiment has been scored yet"})
    write_json(config.LAST_JUDGE_PATH, {"status": "no judge verdict yet"})
    protect_ai_docs()
    write_json(config.FACTORY_STATE_PATH, new_factory_state(ai_docs_checksum()))
    log("[init] workspace initialised: seed state space (1 dimension), paper header, factory state")


if __name__ == "__main__":
    import sys

    initialize(fresh="--fresh" in sys.argv)
