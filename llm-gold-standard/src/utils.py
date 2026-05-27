"""Shared helpers: logging, seeding, IO."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def ensure_dirs(output_dir: str) -> dict:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    return {"base": base}


def save_json(obj, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


class FailureLog:
    """Appends failed/skipped doc records to outputs/scoring_log.txt."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # start fresh each run header, but keep appending within the run
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("--- run start ---\n")

    def record(self, doc_id: str, reason: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{doc_id}\t{reason}\n")
