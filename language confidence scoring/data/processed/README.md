# Language Confidence Features

Output of `script/language_confidence_test.py`. One row per document, joinable on `id`.

## Columns

| Column | Type | Description |
|---|---|---|
| `id` | string | Document identifier (URN), join key |
| `language` | string | Declared language partition in FinePDF (e.g. `swe_Latn`) |
| `full_doc_lid` | string | Language predicted by LID model on full concatenated document text |
| `full_doc_lid_score` | float | Confidence in `full_doc_lid` prediction (0–1) |
| `page_average_lid` | string | Most common language when running LID per page |
| `page_average_lid_score` | float | Confidence for the page-average prediction (0–1) |
| `language_match_finepdfs` | bool | `True` if `full_doc_lid == language` |
| `low_confidence_finepdfs` | bool | `True` if `full_doc_lid_score < 0.70` |

## Workflow

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Run the extraction script:
   ```bash
   python script/language_confidence_test.py \
     --language swe_Latn \
     --n-rows 1000 \
     --output data/processed/language_confidence_swe_Latn.parquet
   ```

   The script:
   - Streams `n-rows` documents from `openeurollm/propella-annotations`
   - Collects their IDs, then scans `HuggingFaceFW/finepdfs` to find matching rows
   - Derives `language_match_finepdfs` and `low_confidence_finepdfs` flags
   - Saves a lean Parquet with only the language confidence columns (ready to join with other feature tables)

3. Verify the output:
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_parquet('data/processed/language_confidence_swe_Latn.parquet')
   print(df.shape)
   print(df.head())
   "
   ```

## Use in Data Quality Ranking

These features are one component of the full document-level feature table used to train and evaluate a quality ranker for multilingual web corpora. The project compares interpretable heuristic baselines with a LightGBM LambdaMART learned ranker, evaluated with NDCG@k, Precision@k, and cross-language robustness metrics.

**Full feature table composition:**

| Feature group | Source | Workstream |
|---|---|---|
| Content quality, information density, educational value, safety, PII, content integrity | `openeurollm/propella-annotations` | All |
| Language confidence (`full_doc_lid_score`, `language_match`, `low_confidence`) | This file | Language confidence |
| Duplicate cluster features (`duplicate_cluster_id`, `cluster_size`, `max_similarity`) | Computed from embeddings | Duplicate clustering (teammate) |
| Human retention labels (KEEP / MAYBE / REMOVE) | A/B annotation study | User study (teammate) |

**How language confidence is used:**

- `full_doc_lid_score` is a continuous ranker feature — low scores indicate the document may not be in the target language.
- `language_match_finepdfs` and `low_confidence_finepdfs` serve as hard filters (pre-ranking) to remove wrong-language and severely degraded documents before ranking.
- The research question is whether language confidence adds signal beyond what Propella content quality and deduplication already capture — tested via feature ablations in Week 3.

**To assemble the full feature table:**
```python
import pandas as pd

propella = pd.read_parquet("data/processed/propella_swe_Latn.parquet")
lang = pd.read_parquet("data/processed/language_confidence_swe_Latn.parquet")
dupes = pd.read_parquet("data/processed/duplicate_features_swe_Latn.parquet")
human = pd.read_parquet("data/processed/human_judgments_swe_Latn.parquet")

features = propella.merge(lang, on="id").merge(dupes, on="id").merge(human, on="id")
```

## Notes

- `full_doc_lid_score` is the primary signal for the ranker — low score means the document may not actually be in the target language.
- `full_doc_lid` and `page_average_lid` can disagree on documents with heavy structured/numeric content (tables, forms), where full-text LID is less reliable than per-page LID.
- Language codes follow the FLORES-200 format, e.g. `swe_Latn` = Swedish in Latin script.
