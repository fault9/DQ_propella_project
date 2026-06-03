# LLM Quality Scorer

Scores Swedish PDF documents on two axes, **educational value** and **content quality**, using an LLM judge. The combined `quality_score` is the unweighted mean `(edu + cq) / 2`.

The LLM sees raw text only, never Propella labels, providing an independent signal for training the reranker.

## Usage

```bash
pip install -r requirements.txt

# Score a stratified sample (default: Berget/Gemma-4-31B)
export BERGET_API_KEY=...
python run_pipeline.py --sample_size 2000

# Use Anthropic instead
export ANTHROPIC_API_KEY=sk-ant-...
python run_pipeline.py --anthropic --sample_size 2000

# Build extreme-quality training sets
python build_extreme_sample.py

# Run LLM-vs-Human diagnostic
python llm_human_diagnostic.py

# Tests
python -m pytest tests/ -v
```

## How it works

1. **Sample**: stream a ~50k Propella pool, stratified-sample by `(educational_value, content_quality)` ordinals so every non-empty quality stratum is represented.
2. **Fetch text**: stream FinePDFs, match sampled ids, truncate, cache to parquet.
3. **Score**: call the LLM per doc with the scoring prompt (`prompts/quality_prompt.txt`), parse two 0-1 axes, combine into `quality_score`. Retries with backoff on API errors, checkpoints every 50 docs, resumes from partial runs.

The prompt is editable and its SHA-256 hash is recorded in `outputs/config.json` for reproducibility.

## Scripts

| Script | Purpose |
|---|---|
| `run_pipeline.py` | Main pipeline: sample, fetch text, score, write deliverables |
| `build_extreme_sample.py` | Sample extreme docs, score, filter into train top/bottom |
| `llm_human_diagnostic.py` | Compare LLM scores against 130 human-annotated docs |

## Outputs

| File | Description |
|---|---|
| `gold_standard_{lang}.parquet/.csv` | Deliverable: `id, quality_score` |
| `gold_standard_{lang}_axes.parquet/.csv` | Per-axis scores: `doc_id, educational_value, content_quality, quality_score` |
| `LLM_scoring_finepdf_propella_combined_{lang}.csv` | 24-col file: scores + Propella features + FinePDFs metadata |
| `extreme_train_top.csv` / `extreme_train_bottom.csv` | Extreme-quality training sets |
| `human_vs_llm_comparison.png/.txt` | LLM-vs-Human diagnostic |
| `Scoring+Labels.zip` / `train_extremes.zip` | Packaged deliverables |

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--sample_size` | 2000 | Docs to score |
| `--seed` | 42 | Reproducible sampling |
| `--language` | `swe_Latn` | Dataset split |
| `--provider` | `berget` | `berget` / `anthropic` / `openai` |
| `--model` | per-provider | Model id |
| `--anthropic` | off | Shortcut for `--provider anthropic` |
| `--max_chars` | 3000 | Chars of doc text sent per doc |
| `--batch_size` | 5 | Parallel API requests |
| `--skip_text_fetch` | off | Reuse cached raw texts |
| `--skip_scoring` | off | Reuse existing scores |
| `--estimate_only` | off | Print cost estimate without calling API |

## Cost

LLM scoring is the only cost. Use `--estimate_only` to get a token/price estimate before committing:

```bash
python run_pipeline.py --sample_size 2000 --max_chars 8000 --estimate_only
```

The main levers are `--sample_size`, `--max_chars`, and model choice. The default (Berget Gemma-4-31B, 3000 chars) is roughly ~$1 for 2000 docs. Anthropic Claude costs several times more.

## Design note

The LLM judge scores from **raw text only** and never sees Propella annotations. This makes the scores an independent supervision signal, avoiding circularity when Propella features are used to train the reranker.
