# Evaluation Report

**Generated:** 2026-05-29T12:18:41.893690+00:00
**Config:** `configs\finepdfs_swe_latn_local.yaml`
**Labels:** `traindata\Combined_train.csv`

## Dataset

| Metric | Value |
| --- | ---: |
| Rows in labels file | 1978 |
| Labeled + passed hard filter (deduped) | 1976 |
| Holdout test documents | 210 |
| Gold score mean (test) | 0.459 |
| Gold score std (test) | 0.186 |

Split: hash holdout (10% test). Hard filter uses `full_doc_lid == expected_lid` from config.

## Results on holdout test

| Method | spearman | recall_at_1pct | recall_at_5pct | recall_at_10pct | recall_at_20pct | recall_at_30pct | ndcg_at_10 | ndcg_at_25 | ndcg_at_50 | ndcg_at_100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random shuffle | 0.085 | 0.000 | 0.000 | 0.048 | 0.310 | 0.317 | 0.123 | 0.247 | 0.308 | 0.411 |
| Descending mean of L1 input features | 0.637 | 0.000 | 0.182 | 0.333 | 0.452 | 0.667 | 0.323 | 0.427 | 0.564 | 0.642 |
| educational_value → content_quality → safety → doc_hash | 0.656 | 0.000 | 0.273 | 0.333 | 0.595 | 0.651 | 0.317 | 0.446 | 0.603 | 0.655 |
| content_quality → educational_value → safety → doc_hash | 0.736 | 0.000 | 0.273 | 0.381 | 0.405 | 0.571 | 0.317 | 0.449 | 0.490 | 0.668 |
| Trained L1 (pairwise linear) | 0.717 | 0.000 | 0.182 | 0.333 | 0.500 | 0.667 | 0.323 | 0.438 | 0.559 | 0.659 |
| Trained L2 (LightGBM LambdaRank) | 0.735 | 0.000 | 0.455 | 0.381 | 0.476 | 0.619 | 0.437 | 0.477 | 0.568 | 0.674 |

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
