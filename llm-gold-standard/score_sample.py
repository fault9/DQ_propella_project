"""Pilot: score N already-cached texts and print results next to Propella's labels.

Uses the existing raw_texts cache (no fetching) and the real LLM call, so you can
eyeball quality/calibration before committing to the full run. Needs the provider
API key in the environment (e.g. BERGET_API_KEY).

    ../.venv/bin/python score_sample.py --n 5
    ../.venv/bin/python score_sample.py --n 10 --max_chars 16000
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from src.config import build_api_config, DEFAULT_PROVIDER, DEFAULT_MAX_TOKENS, default_model_for
from src.llm_scorer import score_one


def main() -> None:
    p = argparse.ArgumentParser(description="Score N cached texts as a pilot.")
    p.add_argument("--language", default="swe_Latn")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_chars", type=int, default=16000)
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--anthropic", action="store_true",
                   help="shortcut for --provider anthropic (uses Claude; needs ANTHROPIC_API_KEY)")
    p.add_argument("--model", default=None,
                   help="defaults per provider (Llama-70B for berget, Claude for anthropic)")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--prompt_file", default=None,
                   help="override prompt file (defaults to prompts/quality_prompt.txt)")
    args = p.parse_args()

    cache = pathlib.Path(args.output_dir) / f"raw_texts_{args.language}.parquet"
    if not cache.exists():
        raise FileNotFoundError(f"No cache at {cache}. Run a fetch first.")
    df = pd.read_parquet(cache)

    # Optional: Propella's own ordinal labels for comparison.
    labels = {}
    man = pathlib.Path(args.output_dir) / f"sample_{args.language}.parquet"
    if man.exists():
        m = pd.read_parquet(man)
        labels = {str(r.doc_id): (int(r.edu_ord), int(r.quality_ord)) for r in m.itertuples()}

    prompt = pathlib.Path(args.prompt_file or "prompts/quality_prompt.txt").read_text(encoding="utf-8")
    provider = "anthropic" if args.anthropic else args.provider
    model = args.model or default_model_for(provider)
    cfg = build_api_config(provider, model, DEFAULT_MAX_TOKENS, 0.0)
    print(f"Scoring {args.n} cached texts via {cfg['provider']}/{cfg['model']} "
          f"(max_chars={args.max_chars})...\n")

    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    scores = []
    for r in sample.itertuples():
        did, text = str(r.id), (r.text or "")
        _, edu, quality, combined, raw = score_one(did, text[:args.max_chars], prompt, cfg)
        scores.append(combined)
        edu_s = f"{edu:.2f}" if edu is not None else "None"
        qual_s = f"{quality:.2f}" if quality is not None else "None"
        comb_s = f"{combined:.2f}" if combined is not None else "None"
        print(f"id={did}")
        print(f"  propella(edu,quality)={labels.get(did)}   "
              f"LLM edu={edu_s} qual={qual_s} -> quality_score={comb_s}")
        print(f"  len={len(text):,} chars | text[:160]={text[:160]!r}")
        print(f"  raw={raw[:160]!r}\n")

    ok = [s for s in scores if s is not None]
    if ok:
        print(f"Scored {len(ok)}/{len(scores)} | min={min(ok):.2f} max={max(ok):.2f} "
              f"mean={sum(ok)/len(ok):.2f}")


if __name__ == "__main__":
    main()
