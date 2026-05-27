"""Fetch raw document text from FinePDFs for a set of doc IDs, with a checkpointed cache.

NOTE: Propella's `finepdfs` ordering does NOT match FinePDFs `swe_Latn` ordering, so
the sampled IDs are scattered across the whole corpus — fetching streams a large
fraction of FinePDFs (slow, minutes) and ~5% of IDs may not be present at all.

The cache is written INCREMENTALLY (atomic tmp+rename) as IDs are found, so an
interrupted fetch keeps its progress. Re-running loads what's already cached and
only streams for the still-missing IDs. Pass `stream=False` to use the cache only
(no streaming), e.g. for --skip_text_fetch.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset  # imported at top so tests can monkeypatch it

from src.config import FINEPDFS_DATASET, MAX_TEXT_CHARS
from src.utils import log

PROGRESS_EVERY = 20_000        # log a progress line every N scanned rows
CHECKPOINT_EVERY_FOUND = 10    # rewrite the cache after this many newly-found docs


def _cache_path(output_dir: str, language: str) -> Path:
    return Path(output_dir) / f"raw_texts_{language}.parquet"


def _write_cache(found: dict, cache: Path) -> None:
    """Atomically write the cache (tmp file + rename) so a kill can't corrupt it."""
    tmp = cache.with_suffix(".parquet.tmp")
    pd.DataFrame({"id": list(found.keys()), "text": list(found.values())}).to_parquet(tmp, index=False)
    tmp.replace(cache)


def _load_cache(cache: Path, target: set) -> dict:
    if not cache.exists():
        return {}
    cached = pd.read_parquet(cache)
    m = dict(zip(cached["id"].astype(str), cached["text"]))
    return {d: m[d] for d in target if d in m}


def fetch_texts(doc_ids, language: str, output_dir: str = "outputs",
                stream: bool = True, max_scan: int | None = None) -> dict:
    """Return {doc_id -> raw_text}, resuming from and checkpointing into a parquet cache.

    stream=True (default) streams FinePDFs for any IDs not already cached.
    stream=False returns only what's cached (no streaming).
    """
    doc_ids = [str(d) for d in doc_ids]
    target = set(doc_ids)
    cache = _cache_path(output_dir, language)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Resume: start from whatever is already cached for these IDs.
    found = _load_cache(cache, target)
    if found:
        log(f"Loaded {len(found):,} previously-cached texts (resuming).")
    remaining = target - set(found)

    if not stream or not remaining:
        if remaining and not stream:
            log(f"stream=False: returning {len(found):,} cached texts, {len(remaining):,} still missing.")
        else:
            log(f"All {len(target):,} texts available ({len(found):,} cached); no streaming needed.")
        return {d: found[d] for d in doc_ids if d in found}

    log(f"Streaming {FINEPDFS_DATASET} split='{language}' for {len(remaining):,} remaining "
        f"texts ({len(found):,} already cached)...")
    ds = load_dataset(FINEPDFS_DATASET, language, split="train", streaming=True)

    scanned, new_since_ckpt, first = 0, 0, True
    try:
        for row in ds:
            if first:
                log(f"FinePDFs columns: {sorted(row.keys())}")
                first = False
            scanned += 1
            rid = str(row.get("id"))
            if rid in remaining:
                found[rid] = (row.get("text") or "")[:MAX_TEXT_CHARS]
                remaining.discard(rid)
                new_since_ckpt += 1
                if new_since_ckpt >= CHECKPOINT_EVERY_FOUND:
                    _write_cache(found, cache)
                    new_since_ckpt = 0
                    log(f"  checkpoint: {len(found):,} cached, {len(remaining):,} remaining "
                        f"(scanned {scanned:,})")
                if not remaining:
                    break
            elif scanned % PROGRESS_EVERY == 0:
                log(f"  scanned {scanned:,} rows, found {len(found):,}/{len(target):,}...")
            if max_scan is not None and scanned >= max_scan:
                log(f"Reached --max_scan {max_scan:,}; stopping with {len(remaining):,} still missing.")
                break
    finally:
        _write_cache(found, cache)  # always persist progress on exit

    avg = (sum(len(t) for t in found.values()) / len(found)) if found else 0
    log(f"Fetched {len(found):,}/{len(target):,}; missing {len(remaining):,}; "
        f"avg length {avg:,.0f} chars (scanned {scanned:,} rows). Cache: {cache}")
    return {d: found[d] for d in doc_ids if d in found}
