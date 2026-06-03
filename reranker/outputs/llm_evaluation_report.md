# Evaluation Report

**Generated:** 2026-05-30T07:37:17.213004+00:00
**Config:** `configs\finepdfs_swe_latn_llm.yaml`
**Training data:** `traindata\LLM_train_extended.csv`

## Trained L1 weights

Linear pairwise ranker feature weights (higher absolute value = stronger influence on `z_l1`).

| Feature | Weight |
| --- | ---: |
| `content_quality` | 0.978835 |
| `information_density` | 0.931593 |
| `educational_value` | 0.885188 |
| `length` | 0.527571 |
| `language_confidence` | 0.165561 |
| `pii_absence` | 0.115698 |
| `safety` | 0.000000 |

**Intercept:** 4.323481


## Holdout test (training data)

| Metric | Value |
| --- | ---: |
| Labels file | `traindata\LLM_train_extended.csv` |
| Rows in labels file | 3172 |
| Labeled + passed hard filter (deduped) | 2758 |
| Evaluation documents | 288 |
| Gold score mean | 0.400 |
| Gold score std | 0.197 |

Split: hash holdout (10% test) from training labels. Hard filter uses `full_doc_lid == expected_lid` from config.

| Method | spearman | recall_at_1pct | recall_at_5pct | recall_at_10pct | recall_at_20pct | recall_at_30pct | ndcg_at_10 | ndcg_at_25 | ndcg_at_50 | ndcg_at_100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random shuffle | -0.001 | 0.000 | 0.000 | 0.069 | 0.121 | 0.264 | 0.156 | 0.173 | 0.188 | 0.322 |
| Descending mean of L1 input features | 0.801 | 0.000 | 0.267 | 0.448 | 0.586 | 0.736 | 0.391 | 0.543 | 0.630 | 0.757 |
| educational_value → content_quality → safety → doc_hash | 0.811 | 0.333 | 0.333 | 0.483 | 0.655 | 0.747 | 0.605 | 0.647 | 0.750 | 0.836 |
| content_quality → educational_value → safety → doc_hash | 0.860 | 0.333 | 0.333 | 0.483 | 0.569 | 0.724 | 0.605 | 0.647 | 0.717 | 0.824 |
| Trained L1 (pairwise linear) | 0.844 | 0.000 | 0.267 | 0.448 | 0.569 | 0.759 | 0.401 | 0.546 | 0.645 | 0.770 |
| Trained L2 (LightGBM LambdaRank) | 0.776 | 0.000 | 0.200 | 0.345 | 0.466 | 0.609 | 0.558 | 0.565 | 0.604 | 0.726 |

## Human train (external)

| Metric | Value |
| --- | ---: |
| Labels file | `traindata\human_train.csv` |
| Rows in labels file | 130 |
| Labeled + passed hard filter (deduped) | 130 |
| Evaluation documents | 130 |
| Gold score mean | 0.516 |
| Gold score std | 0.223 |

Split: all labeled documents (external evaluation). Hard filter uses `full_doc_lid == expected_lid` from config.

| Method | spearman | recall_at_1pct | recall_at_5pct | recall_at_10pct | recall_at_20pct | recall_at_30pct | ndcg_at_10 | ndcg_at_25 | ndcg_at_50 | ndcg_at_100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random shuffle | 0.003 | 0.000 | 0.000 | 0.077 | 0.115 | 0.205 | 0.135 | 0.200 | 0.340 | 0.510 |
| Descending mean of L1 input features | 0.253 | 0.000 | 0.143 | 0.231 | 0.346 | 0.385 | 0.248 | 0.410 | 0.483 | 0.639 |
| educational_value → content_quality → safety → doc_hash | 0.194 | 0.000 | 0.143 | 0.154 | 0.269 | 0.333 | 0.291 | 0.398 | 0.477 | 0.622 |
| content_quality → educational_value → safety → doc_hash | 0.371 | 0.000 | 0.143 | 0.154 | 0.346 | 0.436 | 0.291 | 0.431 | 0.513 | 0.692 |
| Trained L1 (pairwise linear) | 0.278 | 0.000 | 0.143 | 0.231 | 0.346 | 0.385 | 0.248 | 0.397 | 0.483 | 0.648 |
| Trained L2 (LightGBM LambdaRank) | 0.387 | 0.000 | 0.000 | 0.077 | 0.423 | 0.513 | 0.161 | 0.389 | 0.463 | 0.647 |

## Methods

| Key | Description |
| --- | --- |
| `random_shuffle` | Uniform random scores (fixed seed) |
| `feature_mean` | Mean of normalized L1 input features (`*_z`) |
| `edu_quality_safety_hash` | Lexicographic: educational_value, content_quality, safety, doc_hash |
| `quality_edu_safety_hash` | Lexicographic: content_quality, educational_value, safety, doc_hash |
| `l1_trained` | Trained pairwise linear L1 model (`z_l1`) |
| `l2_trained` | Trained LightGBM LambdaRank L2 model |

Recall@p%: fraction of top-p% gold documents captured in the top-p% of predicted scores. NDCG uses 8-bin relevance derived from gold scores in `[0, 1]`.
