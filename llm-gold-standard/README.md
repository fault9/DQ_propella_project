# LLM Gold-Standard Quality Scoring

Produces a **gold standard** of LLM-judged document quality for the data-curation
ranking project: `(doc_id, quality_score)` where `quality_score ∈ [0.0, 1.0]`. The score
combines two independently-judged axes — **educational value** and **content quality** —
into one number (see [`RESEARCH.md`](RESEARCH.md) for the rubric, scale, and combine-rule
rationale).

A collaborator uses this gold standard to train/validate L1 (logistic regression)
and L2 (LightGBM reranker) models that predict quality from Propella features.
**This pipeline only produces the scores** — no feature joining, training, or evaluation.

> Why an external LLM and not just Propella? Propella's features train/validate the
> rankers; using a Propella-derived score to also supervise them would be circular.
> A separate LLM judge (scoring from **raw text only**, never shown Propella labels)
> is an *independent* signal — an abundant proxy for the scarce human A/B labels.

## Pipeline

1. **Sample** (`src/sampler.py`) — stream a ~50k Propella pool (`finepdfs` subset), bucket by `(educational_value, content_quality)` ordinals (≤25 strata), and **stratified-sample** so every non-empty stratum is represented.
2. **Fetch text** (`src/text_fetcher.py`) — stream `HuggingFaceFW/finepdfs`, match the sampled `id`s, truncate to 50k chars, **cache** to parquet.
3. **Score** (`src/llm_scorer.py`) — call an LLM per doc with the editable prompt, parse **two independent 0–1 axes** (`educational_value`, `content_quality`), **combine** them into `quality_score` (geometric mean; `COMBINE_MODE` in `src/config.py`), **retry+backoff**, **checkpoint every 50**, **resume from partial**.

## Install & run

```bash
pip install -r requirements.txt

python run_pipeline.py --sample_size 100 --seed 42      # quick test
python run_pipeline.py --sample_size 2000               # full run
python -m pytest tests/ -v                              # tests (no API key needed)
```

## LLM providers

One OpenAI-compatible path (the **default** judge runs on **Berget AI**, EU) plus a native
**Anthropic** path used as a different-family second judge. Set the API key in the
environment, then choose a provider:

```bash
# Berget AI (default — EU, OpenAI-compatible; Llama-3.3-70B judge)
export BERGET_API_KEY=...
python run_pipeline.py --sample_size 2000                                       # meta-llama/Llama-3.3-70B-Instruct

# Anthropic (different-family second judge / cross-validation)
export ANTHROPIC_API_KEY=sk-ant-...
python run_pipeline.py --anthropic --sample_size 2000                           # claude-sonnet-4-6
python run_pipeline.py --anthropic --model claude-opus-4-7 --sample_size 2000   # max quality

# OpenAI
export OPENAI_API_KEY=sk-...
python run_pipeline.py --provider openai --model gpt-4o-mini --sample_size 2000
```

Any OpenAI-compatible endpoint also works via `--base_url ... --api_key_env MY_KEY`.

> **Model note:** the default judge is `--provider berget --model meta-llama/Llama-3.3-70B-Instruct`
> — the same model class FineWeb-Edu used as its educational-quality annotator (see
> [`RESEARCH.md`](RESEARCH.md) §H). The `--anthropic` shortcut runs Claude (`claude-sonnet-4-6`)
> as a *different-family* second judge for cross-validation; use `claude-opus-4-7` for max
> quality or `claude-haiku-4-5-20251001` to cut cost. At ~2000 docs any of these is inexpensive.
>
> **Berget:** base URL is `https://api.berget.ai/v1` (override with `--base_url`).

### Key CLI flags
| Flag | Default | Meaning |
|---|---|---|
| `--sample_size` | 2000 | docs to score |
| `--seed` | 42 | reproducible sampling |
| `--language` | `swe_Latn` | dataset split (auto-falls back if missing) |
| `--provider` | `berget` | `berget` \| `anthropic` \| `openai` |
| `--model` | `Llama-3.3-70B-Instruct` | model id (per-provider default) |
| `--anthropic` | off | shortcut for `--provider anthropic` (Claude) |
| `--max_chars` | 3000 | chars of doc text sent per doc (cost lever; `-1` = no cap) |
| `--batch_size` | 5 | parallel API requests |
| `--base_url` / `--api_key_env` | preset | OpenAI-compatible endpoint overrides |
| `--skip_text_fetch` | off | require the raw-text cache, skip streaming |
| `--skip_scoring` | off | reuse existing gold standard, skip API calls |
| `--prompt_file` | `prompts/quality_prompt.txt` | editable scoring prompt |

## Outputs

| File | Description |
|---|---|
| `outputs/gold_standard_{lang}.parquet` | **the deliverable**: `doc_id, quality_score` only |
| `outputs/gold_standard_{lang}.csv` | same two columns, for inspection |
| `outputs/gold_standard_{lang}_axes.parquet` | per-axis diagnostics: `doc_id, educational_value, content_quality, quality_score` (kept out of the deliverable) |
| `outputs/gold_standard_{lang}_partial.parquet` | checkpoint for crash/resume (includes `raw_response`) |
| `outputs/raw_texts_{lang}.parquet` | cached raw texts (re-runs skip fetching) |
| `outputs/sample_{lang}.parquet` | sample manifest: `doc_id` + strata + reweighting weight |
| `outputs/sampling_stratification.csv` | per-stratum sampling stats |
| `outputs/config.json` | run config + prompt SHA-256 |
| `outputs/scoring_log.txt` | failed/skipped docs |

## Budget / cost control

LLM scoring is the only thing that costs money (text fetch is just time). The input
text dominates cost, so the levers are **model**, **sample_size**, and **`--max_chars`**
(how much text each doc sends). Get a **free estimate first** — it never calls the API:

```bash
python run_pipeline.py --sample_size 2000 --max_chars 8000 --estimate_only
```

Approx pricing (per 1M tokens, **verify current rates**): Berget Llama-3.3-70B ≈ $0.97/$0.97;
Claude Haiku 4.5 ≈ $1/$5, Sonnet 4.6 ≈ $3/$15, Opus 4.7 ≈ $15/$75. Rough cost for N docs ≈
`N × (rubric + max_chars)/4 / 1e6 × input_rate` (the static rubric is ~1.2k tokens, so it
dominates at small `--max_chars`).

Default is the **Berget Llama-3.3-70B** judge + `--max_chars 3000` — roughly a few dollars for
2000 docs (run `--estimate_only` for the exact figure; Claude Sonnet/Opus cost several× to ~20×
more). To fit a tight budget: lower `--max_chars` and/or `--sample_size`, run `--estimate_only`,
and consider a tiny pilot (`--sample_size 20`) to measure real cost before committing.

## Robustness
- **Resume:** re-running skips doc_ids already in the partial file.
- **Cache:** raw texts are cached; `--skip_text_fetch` reuses them.
- **Retries:** 3× with exponential backoff on API errors *and* malformed responses, then the doc is skipped (score `null`) and logged.
- **Missing API key:** fails fast with a clear message before any sampling.
- **Editable prompt:** change `prompts/quality_prompt.txt` without touching code (the hash is recorded in `config.json`).

The LLM judge sees **raw text only** — never Propella labels.
