# Learned Reranker

This project builds a train-ready document reranker for Propella-annotated FinePDFs data.
The current prototype targets `finepdfs / swe_Latn`.

The pipeline has three stages:

1. `L0`: configurable hard filtering. The default filter only keeps `full_doc_lid == swe_Latn`.
2. `L1`: candidate generation with a linear pairwise ranker once gold labels exist. Until then, a clearly marked prototype heuristic is available.
3. `L2`: LightGBM LambdaRank reranking over the top `m * k` L1 candidates. It can be disabled, run in prototype mode, or use a trained model via `l2_mode: disabled | prototype | trained`.

## Features

The user priority budget covers seven user-facing features:

- `content_quality`
- `information_density`
- `educational_value`
- `safety`
- `pii_absence`
- `length`
- `language_confidence`

`pii_absence` is the inverse of Propella's `pii_presence`, so higher remains better.

FinePDFs duplication metadata is used only by L2:

- `inverted_cluster_size = 1 - normalize(log(1 + minhash_cluster_size))`
- `inverted_duplicate_count = 1 - normalize(log(1 + duplicate_count))`

The main report keeps only the raw `minhash_cluster_size` and `duplicate_count` columns.

## Usage

Create a config interactively:

```powershell
learned-reranker wizard
```

Run the prototype and write a Parquet report:

```powershell
learned-reranker run --config configs/finepdfs_swe_latn.yaml
```

Preview the top `k` rows as JSON without writing a file:

```powershell
learned-reranker run --config configs/finepdfs_swe_latn.yaml --preview
```

Train L1 when labels arrive:

```powershell
learned-reranker train-l1 --config configs/finepdfs_swe_latn.yaml --labels labels.csv
```

Train L2 when labels arrive:

```powershell
learned-reranker train-l2 --config configs/finepdfs_swe_latn.yaml --labels labels.csv
```

Labels must contain:

```text
id,gold_score
```

where `gold_score` is in `[0.0, 1.0]`.

## Output

The canonical output is Parquet, with optional JSONL support. The report columns are:

- `doc_id`
- `passed_hard_filter`
- `rank_l1`
- `z_l1`
- `rank_l2`
- `score_l2`
- `in_top_k`
- `minhash_cluster_size`
- `duplicate_count`

Training artifacts also save feature means/stds, label distributions, relevance-bin histograms, normalization percentiles, model settings, and metrics.
