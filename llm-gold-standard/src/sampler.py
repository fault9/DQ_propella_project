"""Stratified sampling of document IDs from Propella annotations (finepdfs subset).

Streams a pool into memory, buckets docs by (educational_value, content_quality)
ordinal pair (<=25 strata), and samples so every non-empty stratum is represented
(>= MIN_PER_STRATUM) with the remaining budget spread proportionally to size.
"""
from __future__ import annotations

from itertools import islice

import pandas as pd
from datasets import load_dataset

from src.config import (
    PROPELLA_DATASET, PROPELLA_SUBSET, POOL_SIZE, ORDINAL_MAPS, STRATA_FEATURES,
    MAX_STRATA, MIN_PER_STRATUM, PREFERRED_LANGUAGES,
)
from src.utils import log


def resolve_language(language: str) -> str:
    try:
        from datasets import get_dataset_split_names
        splits = list(get_dataset_split_names(PROPELLA_DATASET, PROPELLA_SUBSET))
    except Exception as e:
        log(f"Could not list splits for {PROPELLA_DATASET}/{PROPELLA_SUBSET}: {e!r}")
        return language
    if language in splits:
        return language
    log(f"Split '{language}' not in {PROPELLA_SUBSET}. Available (first 20): {splits[:20]}")
    for cand in PREFERRED_LANGUAGES:
        if cand in splits:
            log(f"Falling back to '{cand}'.")
            return cand
    log(f"Falling back to first available split '{splits[0]}'.")
    return splits[0]


def _to_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feat in STRATA_FEATURES:
        col = "edu_ord" if feat == "educational_value" else "quality_ord"
        labels = df[feat].astype("string").str.strip().str.lower()
        df[col] = labels.map(ORDINAL_MAPS[feat])
    return df


def _allocate_proportional(sizes: dict, sample_size: int, min_per: int = MIN_PER_STRATUM) -> dict:
    """min_per per non-empty stratum, remainder ∝ leftover capacity (mirrors the corpus)."""
    strata = list(sizes)
    sample_size = min(sample_size, sum(sizes.values()))
    alloc = {s: 0 for s in strata}

    for s in sorted(strata, key=lambda k: -sizes[k]):
        if sum(alloc.values()) >= sample_size:
            break
        alloc[s] = min(min_per, sizes[s], sample_size - sum(alloc.values()))

    remaining = sample_size - sum(alloc.values())
    while remaining > 0:
        caps = {s: sizes[s] - alloc[s] for s in strata if sizes[s] - alloc[s] > 0}
        if not caps:
            break
        total_cap = sum(caps.values())
        added = 0
        for s, cap in caps.items():
            if remaining - added <= 0:
                break
            give = min(cap, int(remaining * cap / total_cap))
            alloc[s] += give
            added += give
        remaining = sample_size - sum(alloc.values())
        if added == 0:
            for s in sorted(caps, key=lambda k: -caps[k]):
                if remaining <= 0:
                    break
                alloc[s] += 1
                remaining -= 1
    return alloc


def _allocate_uniform(sizes: dict, sample_size: int) -> dict:
    """Equal docs per stratum, capped by availability; leftover from small strata
    is redistributed to strata that still have capacity. Best for a ranking gold
    standard — covers the rare high/low-quality tails."""
    strata = list(sizes)
    sample_size = min(sample_size, sum(sizes.values()))
    alloc = {s: 0 for s in strata}
    active = set(strata)
    remaining = sample_size
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for s in list(active):
            if remaining <= 0:
                break
            give = min(share, sizes[s] - alloc[s], remaining)
            if give > 0:
                alloc[s] += give
                remaining -= give
                progressed = True
            if alloc[s] >= sizes[s]:
                active.discard(s)
        if not progressed:
            break
    return alloc


def stratified_sample(pool: pd.DataFrame, sample_size: int, seed: int, mode: str = "uniform"):
    """Stratified sampler over a pool with edu_ord/quality_ord.

    Returns (sampled_df, stratum_stats). The df gets a `stratum_weight` column =
    inverse-propensity weight (corpus_frac / sample_frac) so the sample can be
    re-weighted back to the corpus prior. mode: 'uniform' (balanced) or 'proportional'.
    """
    pool = pool.copy()
    pool["strat"] = list(zip(pool["edu_ord"].astype(int), pool["quality_ord"].astype(int)))
    sizes = pool["strat"].value_counts().to_dict()

    alloc = (_allocate_uniform(sizes, sample_size) if mode == "uniform"
             else _allocate_proportional(sizes, sample_size, MIN_PER_STRATUM))

    total_pool = sum(sizes.values())
    total_samp = sum(v for v in alloc.values() if v > 0)
    weights = {s: (sizes[s] / total_pool) / (alloc[s] / total_samp)
               for s in alloc if alloc[s] > 0}

    parts = []
    for strat, n in alloc.items():
        if n <= 0:
            continue
        grp = pool[pool["strat"] == strat]
        parts.append(grp.sample(n=min(n, len(grp)), random_state=seed))
    sampled = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    sampled["stratum_weight"] = sampled["strat"].map(weights)

    stats = [{
        "edu_ord": s[0], "quality_ord": s[1],
        "pool_size": sizes[s], "sampled": alloc[s],
        "pop_frac": sizes[s] / total_pool, "sample_frac": alloc[s] / total_samp,
        "stratum_weight": weights[s],
    } for s in alloc if alloc[s] > 0]

    n_strata = sum(1 for c in sizes.values() if c > 0)
    per = sampled["strat"].value_counts()
    log(f"Stratified [{mode}]: {n_strata}/{MAX_STRATA} non-empty strata; "
        f"docs/stratum min={per.min()}, max={per.max()}, mean={per.mean():.1f}; "
        f"reweighting factors {min(weights.values()):.2f}–{max(weights.values()):.2f}.")
    return sampled.drop(columns=["strat"]), stats


def sample_documents(language: str, sample_size: int, seed: int,
                     mode: str = "uniform", pool_size: int = POOL_SIZE):
    """Stream a Propella pool, convert to ordinals, stratified-sample. Returns (df, stats)."""
    lang = resolve_language(language)
    log(f"Streaming up to {pool_size:,} Propella rows from {PROPELLA_DATASET}/{PROPELLA_SUBSET} "
        f"split='{lang}'...")
    ds = load_dataset(PROPELLA_DATASET, PROPELLA_SUBSET, split=lang, streaming=True)

    rows, first = [], True
    for row in islice(ds, pool_size):
        if first:
            log(f"Propella columns: {sorted(row.keys())}")
            first = False
        rows.append({
            "id": row.get("id"),
            "educational_value": row.get("educational_value"),
            "content_quality": row.get("content_quality"),
        })
    pool = pd.DataFrame(rows)
    log(f"Loaded pool of {len(pool):,} docs.")

    pool = pool.dropna(subset=["id"] + STRATA_FEATURES)
    pool = _to_ordinals(pool).dropna(subset=["edu_ord", "quality_ord"])
    pool["edu_ord"] = pool["edu_ord"].astype(int)
    pool["quality_ord"] = pool["quality_ord"].astype(int)
    pool["id"] = pool["id"].astype("string")
    if len(pool) == 0:
        raise RuntimeError(f"No usable Propella rows for '{lang}'.")

    sampled, stats = stratified_sample(pool, sample_size, seed, mode=mode)
    sampled["language"] = lang
    log(f"Sampled {len(sampled):,} doc IDs (requested {sample_size:,}, mode={mode}).")
    return sampled, stats
