"""LLM-judge scoring: call an LLM per document, parse two 0.0–1.0 axis scores.

The judge rates two INDEPENDENT axes — educational_value and content_quality —
which are combined in code (see combine_scores) into the final quality_score.
Scoring the axes separately (then combining) avoids a halo/blend effect where one
strong axis inflates the whole score, and keeps each axis joinable/validatable
against Propella's two ordinals.

One OpenAI-compatible path (provider "openai" / "berget" / any --base_url) plus a
native Anthropic path. Robust JSON parsing, retry+backoff, parallel requests,
checkpointing, and resume-from-partial.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.config import (MAX_RETRIES, BACKOFF_BASE, CHECKPOINT_EVERY, COMBINE_MODE,
                        EDU_WEIGHT, QUAL_WEIGHT, get_pricing)
from src.utils import log

AXIS_KEYS = ("educational_value", "content_quality")
OUTPUT_COLUMNS = ["doc_id", "educational_value", "content_quality", "quality_score", "raw_response"]
CHARS_PER_TOKEN = 4          # rough estimate for cost preview
EST_OUTPUT_TOKENS = 30       # `{"id": "...", "educational_value": 0.7, "content_quality": 0.4}`


def estimate_cost(texts: dict, prompt_template: str, api_config: dict,
                  max_chars: int | None = None, price_override: tuple | None = None) -> dict:
    """Free, pre-flight cost estimate (no API calls). Token counts are approximate.

    `price_override` = (input_per_M, output_per_M) lets you plug in a provider's real
    rate (e.g. Berget's published price) so the estimate and --max_budget are accurate.
    """
    rubric_chars = len(prompt_template.replace("{text}", ""))
    n = len(texts)
    in_chars = sum(rubric_chars + len((t[:max_chars] if max_chars else t)) for t in texts.values())
    in_tok = in_chars / CHARS_PER_TOKEN
    out_tok = n * EST_OUTPUT_TOKENS
    if price_override is not None:
        p_in, p_out, known = price_override[0], price_override[1], True
    else:
        p_in, p_out, known = get_pricing(api_config["model"])
    cost = in_tok / 1e6 * p_in + out_tok / 1e6 * p_out
    return {"n": n, "input_tokens": int(in_tok), "output_tokens": int(out_tok),
            "price_in_per_m": p_in, "price_out_per_m": p_out, "pricing_known": known,
            "est_cost_usd": cost}


# --- parsing (pure, no API) ------------------------------------------------
def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _extract_key(obj: dict, key: str):
    """Clamped float for `key` in a parsed JSON object, else None."""
    if isinstance(obj, dict) and key in obj:
        try:
            return _clamp(float(obj[key]))
        except (TypeError, ValueError):
            return None
    return None


def parse_scores(raw: str | None):
    """Extract (educational_value, content_quality), each clamped to [0,1] or None.

    Accepts a clean JSON object, a JSON object embedded in extra text, or
    `educational_value: <num>` / `content_quality: <num>` patterns. Each axis is
    parsed independently, so a partial response still yields whatever it carried.
    """
    if not raw:
        return None, None
    text = str(raw).strip()
    found = {k: None for k in AXIS_KEYS}

    # 1. JSON object(s) embedded anywhere in the text.
    for obj_str in re.findall(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            obj = json.loads(obj_str)
        except Exception:
            continue
        for k in AXIS_KEYS:
            if found[k] is None:
                found[k] = _extract_key(obj, k)

    # 2. Whole response as JSON.
    if any(found[k] is None for k in AXIS_KEYS):
        try:
            obj = json.loads(text)
            for k in AXIS_KEYS:
                if found[k] is None:
                    found[k] = _extract_key(obj, k)
        except Exception:
            pass

    # 3. `key: 0.7` style (non-JSON) fallback, per axis.
    for k in AXIS_KEYS:
        if found[k] is None:
            m = re.search(rf"{k.replace('_', '[_\\s]*')}['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
                          text, re.IGNORECASE)
            if m:
                found[k] = _clamp(float(m.group(1)))

    return found["educational_value"], found["content_quality"]


def combine_scores(edu, quality, mode: str = COMBINE_MODE):
    """Combine the two axes into the final quality_score; None if either is missing.

    'weighted' (default): EDU_WEIGHT*edu + QUAL_WEIGHT*qual — an educational-value-weighted
    arithmetic mean (edu is the primary curation signal, so it drives the score while writing
    quality still contributes); no zero-collapse, no corruption gate needed. Alternatives:
    'geometric' (sqrt(edu*quality), AND-like), 'mean' (plain 0.5/0.5), 'min'. The combine is
    recomputable in code without re-running the LLM, since both raw axes are stored.
    """
    if edu is None or quality is None:
        return None
    if mode == "weighted":
        return EDU_WEIGHT * edu + QUAL_WEIGHT * quality
    if mode == "geometric":
        return math.sqrt(edu * quality)
    if mode == "mean":
        return (edu + quality) / 2.0
    if mode == "min":
        return min(edu, quality)
    raise ValueError(f"Unknown combine mode '{mode}' (use weighted|geometric|mean|min).")


# --- API calls -------------------------------------------------------------
def _resolve_key(api_config: dict) -> str:
    key = os.environ.get(api_config["api_key_env"])
    if not key:
        raise RuntimeError(
            f"Missing API key. Set the {api_config['api_key_env']} environment variable "
            f"(provider='{api_config['provider']}', model='{api_config['model']}')."
        )
    return key


def _call_api(prompt: str, api_config: dict) -> str:
    """Single LLM call -> raw text. Lazily imports the SDK so tests don't need it."""
    key = _resolve_key(api_config)
    if api_config["openai_compatible"]:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=api_config.get("base_url"))
        resp = client.chat.completions.create(
            model=api_config["model"],
            max_tokens=api_config["max_tokens"],
            temperature=api_config["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=api_config["model"],
            max_tokens=api_config["max_tokens"],
            temperature=api_config["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def score_one(doc_id: str, text: str, prompt_template: str, api_config: dict):
    """Score one document with retries+backoff.

    Returns (doc_id, educational_value, content_quality, quality_score, raw). A
    response is valid only if BOTH axes parsed (so the combined score exists);
    otherwise it retries, and finally returns Nones with the last raw/error.
    """
    # str.replace (not .format) — the prompt contains literal JSON braces.
    # Inject {id} first, then {text} (so doc text can't accidentally absorb a placeholder).
    prompt = prompt_template.replace("{id}", str(doc_id)).replace("{text}", text)
    last_raw, last_err = None, None
    for attempt in range(MAX_RETRIES):
        try:
            raw = _call_api(prompt, api_config)
            edu, quality = parse_scores(raw)
            combined = combine_scores(edu, quality)
            if combined is not None:
                return doc_id, edu, quality, combined, raw
            last_raw = raw  # missing an axis -> retry
        except Exception as e:  # rate limit / network / etc.
            last_err = e
        time.sleep(BACKOFF_BASE ** attempt + random.random())
    return doc_id, None, None, None, last_raw if last_raw is not None else f"ERROR: {last_err}"


# --- orchestration ---------------------------------------------------------
def _results_to_df(results: dict) -> pd.DataFrame:
    rows = [{"doc_id": d, "educational_value": e, "content_quality": q,
             "quality_score": c, "raw_response": r}
            for d, (e, q, c, r) in results.items()]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def score_documents(texts: dict, prompt_template: str, api_config: dict,
                    output_dir: str = "outputs", language: str = "swe_Latn",
                    batch_size: int = 5, failure_log=None, max_chars: int | None = None) -> pd.DataFrame:
    """Score all docs in `texts` (resume-aware, checkpointed). Returns the full DataFrame.

    `max_chars` truncates the text sent to the LLM (a cost lever, applied on top of
    the fetch-time cap); None = send the full cached text.
    """
    partial_path = Path(output_dir) / f"gold_standard_{language}_partial.parquet"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results: dict = {}
    if partial_path.exists():
        prev = pd.read_parquet(partial_path)
        for r in prev.itertuples():
            edu = getattr(r, "educational_value", None)
            quality = getattr(r, "content_quality", None)
            results[str(r.doc_id)] = (edu, quality, r.quality_score, r.raw_response)
        log(f"Resuming: {len(results):,} docs already scored; they will be skipped.")

    pending = [(d, (t[:max_chars] if max_chars else t)) for d, t in texts.items() if d not in results]
    total = len(texts)
    log(f"Scoring {len(pending):,} docs ({len(results):,} already done) via "
        f"{api_config['provider']}/{api_config['model']} (batch_size={batch_size}).")

    if not pending:
        return _results_to_df(results)

    scores_seen = [c for _, _, c, _ in results.values() if c is not None]
    start = time.time()
    completed = 0

    def flush():
        _results_to_df(results).to_parquet(partial_path, index=False)

    with ThreadPoolExecutor(max_workers=max(1, batch_size)) as ex:
        futures = {ex.submit(score_one, d, t, prompt_template, api_config): d for d, t in pending}
        for fut in as_completed(futures):
            doc_id, edu, quality, combined, raw = fut.result()
            results[doc_id] = (edu, quality, combined, raw)
            completed += 1
            if combined is None:
                if failure_log:
                    failure_log.record(doc_id, "failed or malformed after retries")
            else:
                scores_seen.append(combined)

            if completed % CHECKPOINT_EVERY == 0 or completed == len(pending):
                flush()
                done_total = len(results)
                avg = (sum(scores_seen) / len(scores_seen)) if scores_seen else float("nan")
                rate = completed / max(time.time() - start, 1e-6)
                remaining_min = ((len(pending) - completed) / rate / 60) if rate > 0 else 0
                log(f"Scored {done_total}/{total} ({100 * done_total / total:.1f}%), "
                    f"avg score: {avg:.3f}, est. remaining: {remaining_min:.1f} min")

    return _results_to_df(results)
