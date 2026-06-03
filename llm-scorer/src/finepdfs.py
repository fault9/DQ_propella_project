"""Shared FinePDFs helpers: id normalization, a streaming single-pass scan, and atomic IO.

Used by the sample builders (`build_extreme_sample*.py`). Kept in `src/` so callers
import from a library module rather than from a CLI script.
"""
from __future__ import annotations

import pathlib

import pandas as pd
from datasets import load_dataset

from src.config import FINEPDFS_DATASET, MAX_TEXT_CHARS
from src.utils import log

PROGRESS_EVERY = 20_000
CHECKPOINT_EVERY_FOUND = 50


# --- IO helpers -----------------------------------------------------------
def _atomic_write(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Atomically write parquet (tmp+rename). Used for resumable checkpoints."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)



# --- ID normalization -----------------------------------------------------
def _normalize_id(s) -> str:
    """Strip the `<urn:uuid:...>` envelope to bare UUID; pass everything else through.

    Both Propella and FinePDFs store some doc ids wrapped (`<urn:uuid:X>`) and some
    bare (`X`). When a target list (e.g. a colleague's CSV) uses one form and the
    other side uses the other, the in-set match silently misses. Normalizing both
    sides to bare UUID before matching makes the match robust to that disagreement.
    UUID collisions across documents are astronomically unlikely (122 random bits),
    so stripping the envelope never crosses unrelated rows in practice.
    """
    s = str(s).strip()
    if s.startswith("<urn:uuid:") and s.endswith(">"):
        return s[len("<urn:uuid:"):-1]
    return s


# --- Core FinePDFs scan (single source of truth, reused by all callers) ---
def _scan_finepdfs(target_norm: set,
                   language: str,
                   max_scan: int | None = None,
                   columns: list | None = None,
                   checkpoint_path: pathlib.Path | None = None) -> tuple:
    """Stream FinePDFs once; return `({normalized_id -> row_dict}, scanned_count)`.

    Matches on the *normalized* id, so `<urn:uuid:X>` in one source and bare `X` in
    the other resolve to the same row. The stored row dict preserves FinePDFs'
    original `id` string verbatim. Short-circuits the moment `target_norm` is empty.
    If `checkpoint_path` is given, atomically writes the current found rows there
    every CHECKPOINT_EVERY_FOUND hits so a kill keeps progress.
    """
    target = set(target_norm)
    keep_cols = None if columns is None else list({"id", *columns})
    n_target = len(target)

    sample_targets = list(target)[:3]
    log(f"Streaming {FINEPDFS_DATASET} split='{language}' for {n_target:,} target IDs "
        f"(this can take many minutes — finepdfs is huge and scattered).")
    log(f"  target id sample (normalized): {sample_targets}")
    if keep_cols is not None:
        log(f"  filtering to columns: {sorted(keep_cols)}")

    ds = load_dataset(FINEPDFS_DATASET, language, split="train", streaming=True)
    found: dict = {}
    scanned, new_since_ckpt = 0, 0
    first_row_logged = False
    first_match_logged = False
    try:
        for row in ds:
            if not first_row_logged:
                row_keys = sorted(row.keys())
                log(f"  FinePDFs columns: {row_keys}")
                log(f"  first-row id: {row.get('id')!r}  "
                    f"-> normalized: {_normalize_id(row.get('id'))!r}")
                if columns is not None:
                    missing = [c for c in columns if c not in row]
                    if missing:
                        raise ValueError(
                            f"Requested columns not in FinePDFs: "
                            f"{missing}. Available: {row_keys}")
                first_row_logged = True
            scanned += 1
            rid_norm = _normalize_id(row.get("id"))
            if rid_norm in target:
                d = dict(row) if keep_cols is None else {k: row.get(k) for k in keep_cols}
                if "text" in d:
                    d["text"] = (d.get("text") or "")[:MAX_TEXT_CHARS]
                found[rid_norm] = d
                target.discard(rid_norm)
                if not first_match_logged:
                    log(f"  first match at scan #{scanned:,}: id={row.get('id')!r}  "
                        f"(format normalized for matching)")
                    first_match_logged = True
                new_since_ckpt += 1
                if checkpoint_path is not None and new_since_ckpt >= CHECKPOINT_EVERY_FOUND:
                    _atomic_write(pd.DataFrame(list(found.values())), checkpoint_path)
                    new_since_ckpt = 0
                    log(f"  checkpoint: {len(found):,} found, {len(target):,} remaining "
                        f"(scanned {scanned:,})")
                if not target:
                    break  # early termination: all target ids found
            elif scanned % PROGRESS_EVERY == 0:
                log(f"  FinePDFs: scanned {scanned:,}, found {len(found):,}/{n_target:,}")
            if max_scan is not None and scanned >= max_scan:
                log(f"Reached --max_scan {max_scan:,}; {len(target):,} still missing.")
                break
    finally:
        if checkpoint_path is not None:
            _atomic_write(pd.DataFrame(list(found.values())), checkpoint_path)

    return found, scanned
