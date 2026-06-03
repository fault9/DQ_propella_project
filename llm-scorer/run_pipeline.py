"""
LLM gold-standard quality scoring — main entry point.

    python run_pipeline.py --sample_size 100 --seed 42                    # quick test
    python run_pipeline.py --sample_size 2000                             # full run
    python run_pipeline.py --provider berget --model <berget-model> \\
        --api_key_env BERGET_API_KEY                                      # Berget (OpenAI-compatible)

Deliverable: outputs/gold_standard_{lang}.parquet/.csv = (id, quality_score) only.
quality_score in [0,1] is an unweighted arithmetic mean of two LLM axes
(educational_value + content_quality) / 2. The per-axis scores are kept alongside in
outputs/gold_standard_{lang}_axes.parquet/.csv (doc_id, educational_value, content_quality,
quality_score) for analysis/baselines, not in the deliverable.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.config import (
    CANONICAL_COLUMNS,
    DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, DEFAULT_LANGUAGE, DEFAULT_PROVIDER,
    DEFAULT_BATCH_SIZE, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_FILE, DEFAULT_MAX_CHARS, DEFAULT_SAMPLING_MODE, POOL_SIZE,
    MAX_TEXT_CHARS, PROVIDER_PRESETS, FINEPDFS_META, build_api_config, default_model_for,
)
from src.utils import set_seed, ensure_dirs, save_json, log, FailureLog
from src.sampler import resolve_language, sample_documents
from src.text_fetcher import fetch_texts, _cache_path
from src.llm_scorer import score_documents, estimate_cost, _resolve_key


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM gold-standard quality scoring for FinePDFs/Propella docs.")
    p.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--language", type=str, default=DEFAULT_LANGUAGE)
    p.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, choices=list(PROVIDER_PRESETS))
    p.add_argument("--anthropic", action="store_true",
                   help="shortcut for --provider anthropic (uses Claude; needs ANTHROPIC_API_KEY)")
    p.add_argument("--model", type=str, default=None,
                   help="defaults per provider (Gemma-4-31B for berget, Claude for anthropic)")
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="parallel API requests")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--skip_scoring", action="store_true", help="use existing gold_standard file, skip API calls")
    p.add_argument("--skip_text_fetch", action="store_true", help="require cached raw_texts, skip streaming")
    p.add_argument("--prompt_file", type=str, default=DEFAULT_PROMPT_FILE)
    # Provider flexibility (needed for Berget / any OpenAI-compatible endpoint):
    p.add_argument("--base_url", type=str, default=None, help="override OpenAI-compatible base URL")
    p.add_argument("--api_key_env", type=str, default=None, help="override env var holding the API key")
    p.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    # Budget controls:
    p.add_argument("--max_chars", type=int, default=DEFAULT_MAX_CHARS,
                   help="chars of doc text sent to the LLM (cost lever; -1 = no extra cap)")
    p.add_argument("--max_scan", type=int, default=None,
                   help="cap FinePDFs rows scanned when fetching text (None = until all found)")
    p.add_argument("--estimate_only", action="store_true",
                   help="sample + fetch + print a free cost estimate, then STOP (no API spend)")
    p.add_argument("--max_budget", type=float, default=None,
                   help="abort before scoring if the estimated cost (USD) exceeds this")
    p.add_argument("--price_in", type=float, default=None,
                   help="USD per 1M input tokens (override price table, e.g. for Berget)")
    p.add_argument("--price_out", type=float, default=None,
                   help="USD per 1M output tokens (override price table, e.g. for Berget)")
    # Sampling:
    p.add_argument("--sampling_mode", choices=["uniform", "proportional"], default=DEFAULT_SAMPLING_MODE,
                   help="uniform = balanced across edu×quality strata (best for a ranking gold "
                        "standard); proportional = mirror the corpus distribution")
    p.add_argument("--pool_size", type=int, default=POOL_SIZE,
                   help="Propella rows streamed into memory before stratified sampling")
    return p


def _summary(gold: pd.DataFrame, gold_path: Path) -> None:
    scored = gold["quality_score"].dropna()
    n_fail = len(gold) - len(scored)
    mean = scored.mean() if len(scored) else float("nan")
    median = scored.median() if len(scored) else float("nan")
    print("\n" + "=" * 70)
    print(f"Done. Scored {len(scored):,} documents ({n_fail:,} failed/null).")
    print(f"Combined quality_score — mean: {mean:.4f}   median: {median:.4f}")
    for axis in ("educational_value", "content_quality"):
        if axis in gold.columns:
            a = gold[axis].dropna()
            if len(a):
                print(f"  {axis}: mean {a.mean():.4f}   median {a.median():.4f}")
    print(f"Saved to {gold_path}")
    print("=" * 70)


def _write_outputs(gold: pd.DataFrame, gold_path: Path, gold_csv: Path,
                   axes_path: Path, axes_csv: Path,
                   manifest: pd.DataFrame | None = None,
                   combined_path: Path | None = None) -> None:
    """Project the scored frame into output files and write parquet + csv for each.

    Pure projection — no scoring. Failed docs (quality_score / axes are null) are PRESERVED
    as rows in both files, never dropped.

    - Deliverable (collaborator contract): EXACTLY (id, quality_score); doc_id -> id here only.
    - Axes (analysis/baselines): (doc_id, educational_value, content_quality, quality_score),
      so educational_value can serve as a single-axis baseline vs. the combined score.
    - Combined (if manifest provided): canonical 24-col schema (id, quality_score, 18 Propella
      features, 4 FinePDFs metadata) — replaces the old manual join step.
    """
    axes = gold[["doc_id", "educational_value", "content_quality", "quality_score"]]
    axes.to_parquet(axes_path, index=False)
    axes.to_csv(axes_csv, index=False)

    deliverable = gold[["doc_id", "quality_score"]].rename(columns={"doc_id": "id"})
    deliverable.to_parquet(gold_path, index=False)
    deliverable.to_csv(gold_csv, index=False)

    if manifest is not None and combined_path is not None:
        merged = manifest.merge(
            gold[["doc_id", "quality_score"]], on="doc_id", how="left",
        )
        merged = merged.rename(columns={"doc_id": "id"})
        # Project to canonical 24-col schema; add any missing cols as null.
        for c in CANONICAL_COLUMNS:
            if c not in merged.columns:
                merged[c] = pd.NA
        merged = merged[CANONICAL_COLUMNS]
        merged.to_csv(combined_path, index=False)
        log(f"Combined 24-col file -> {combined_path.name} ({len(merged):,} rows)")


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    dirs = ensure_dirs(args.output_dir)
    base = dirs["base"]

    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")
    # --anthropic is a friendly alias for --provider anthropic; --model (if given) always wins,
    # otherwise the default model is chosen per provider.
    provider = "anthropic" if args.anthropic else args.provider
    model = args.model or default_model_for(provider)
    api_config = build_api_config(
        provider=provider, model=model,
        max_tokens=args.max_tokens, temperature=args.temperature,
        base_url=args.base_url, api_key_env=args.api_key_env,
    )

    lang = resolve_language(args.language)
    gold_path = base / f"gold_standard_{lang}.parquet"
    gold_csv = base / f"gold_standard_{lang}.csv"
    axes_path = base / f"gold_standard_{lang}_axes.parquet"   # analysis: per-axis scores
    axes_csv = base / f"gold_standard_{lang}_axes.csv"

    config = {
        "sample_size": args.sample_size, "seed": args.seed, "language": lang,
        "provider": api_config["provider"], "model": api_config["model"],
        "base_url": api_config["base_url"], "api_key_env": api_config["api_key_env"],
        "batch_size": args.batch_size, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "pool_size": args.pool_size,
        "sampling_mode": args.sampling_mode,
        "max_text_chars": MAX_TEXT_CHARS, "prompt_file": args.prompt_file,
        "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
    }
    save_json(config, base / "config.json")
    log(f"Config: provider={api_config['provider']} model={api_config['model']} "
        f"base_url={api_config['base_url']} lang={lang}")

    # Fast path: reuse an existing gold standard.
    if args.skip_scoring:
        if gold_path.exists():
            log(f"--skip_scoring: loading existing {gold_path}")
            # Prefer the axes file for a richer summary; fall back to the lean deliverable.
            summary_df = pd.read_parquet(axes_path) if axes_path.exists() else pd.read_parquet(gold_path)
            _summary(summary_df, gold_path)
            return
        raise FileNotFoundError(f"--skip_scoring set but {gold_path} does not exist.")

    max_chars = None if (args.max_chars is not None and args.max_chars < 0) else args.max_chars

    # Step 3: stratified sample of doc IDs (+ per-stratum re-weighting stats).
    sampled, strat_stats = sample_documents(lang, args.sample_size, args.seed,
                                            mode=args.sampling_mode, pool_size=args.pool_size)
    doc_ids = sampled["id"].astype(str).tolist()

    # Persist the sample manifest with ALL Propella columns (doc_id + every Propella field
    # the row carried + edu_ord/quality_ord/stratum_weight/language) and the per-stratum
    # table, so the collaborator can join on doc_id without re-scanning the HF dataset.
    manifest_path = base / f"sample_{lang}.parquet"
    manifest = sampled.rename(columns={"id": "doc_id"})
    _priority = ["doc_id", "edu_ord", "quality_ord", "stratum_weight", "language"]
    manifest = manifest[_priority + [c for c in manifest.columns if c not in _priority]]
    manifest.to_parquet(manifest_path, index=False)
    pd.DataFrame(strat_stats).to_csv(base / "sampling_stratification.csv", index=False)
    log(f"Sample manifest -> sample_{lang}.parquet; stratification -> sampling_stratification.csv")

    # Step 4: fetch raw texts (cache-first).
    if args.skip_text_fetch and not _cache_path(args.output_dir, lang).exists():
        raise FileNotFoundError(
            f"--skip_text_fetch set but no cache at {_cache_path(args.output_dir, lang)}. "
            f"Run once without the flag to build it."
        )
    texts = fetch_texts(doc_ids, lang, output_dir=args.output_dir,
                        stream=not args.skip_text_fetch, max_scan=args.max_scan)
    if not texts:
        raise RuntimeError("No raw texts fetched — cannot score. Check the FinePDFs split/ids.")

    # Enrich the sample manifest with the FinePDFs metadata now in the raw-text cache, so it
    # carries all 18 Propella features AND the FinePDFs metadata in one file (easy extraction).
    cache_df = pd.read_parquet(_cache_path(args.output_dir, lang))
    meta_cols = [c for c in FINEPDFS_META if c in cache_df.columns]
    if meta_cols:
        meta = cache_df[["id", *meta_cols]].copy()
        meta["id"] = meta["id"].astype(str)
        manifest = manifest.merge(meta, left_on="doc_id", right_on="id", how="left").drop(columns=["id"])
        manifest.to_parquet(manifest_path, index=False)
        log(f"Enriched sample manifest with FinePDFs metadata: {meta_cols}")

    # Step 4.5: FREE pre-flight cost estimate (no API calls).
    price_override = (args.price_in, args.price_out) if (args.price_in is not None and args.price_out is not None) else None
    est = estimate_cost(texts, prompt_template, api_config, max_chars, price_override=price_override)
    note = "" if est["pricing_known"] else "  [model not in price table — fallback rate]"
    log(f"COST ESTIMATE: {est['n']} docs, ~{est['input_tokens']:,} in + {est['output_tokens']:,} out "
        f"tokens @ ${est['price_in_per_m']}/{est['price_out_per_m']} per 1M -> "
        f"~${est['est_cost_usd']:.2f}{note}  (approx; verify pricing)")
    if args.estimate_only:
        log("--estimate_only: stopping before any API spend.")
        return
    if args.max_budget is not None and est["est_cost_usd"] > args.max_budget:
        log(f"ABORT: estimated ~${est['est_cost_usd']:.2f} exceeds --max_budget ${args.max_budget:.2f}. "
            f"Lower --sample_size or --max_chars (or raise --max_budget) and retry.")
        return

    # Fail fast on a missing API key (only when about to score).
    _resolve_key(api_config)

    # Step 5: LLM scoring (resume-aware).
    failure_log = FailureLog(base / "scoring_log.txt")
    gold = score_documents(texts, prompt_template, api_config,
                           output_dir=args.output_dir, language=lang,
                           batch_size=args.batch_size, failure_log=failure_log, max_chars=max_chars)

    # Step 6: save outputs. Three files:
    #   deliverable -> (id, quality_score) ONLY               (collaborator contract)
    #   axes        -> (doc_id, educational_value, content_quality, quality_score)  (analysis)
    #   combined    -> canonical 24-col schema (id, quality_score, Propella + FinePDFs metadata)
    # Null rows (failed docs) are preserved in all files. See _write_outputs.
    combined_csv = base / f"LLM_scoring_finepdf_propella_combined_{lang}.csv"
    _write_outputs(gold, gold_path, gold_csv, axes_path, axes_csv,
                   manifest=manifest, combined_path=combined_csv)
    _summary(gold, gold_path)


if __name__ == "__main__":
    main()
