# DQ Propella

Data quality scoring and reranking for Swedish PDF documents, using Propella annotations and FinePDFs metadata.

## Components

| Directory | What it does |
|---|---|
| `llm-scorer/` | Scores documents with an LLM judge on two axes (educational value, content quality). Produces gold-standard quality labels for training the reranker. |
| `human_evaluation/` | Pairwise A/B rating tool where human raters compare document quality. Produces human scores used to validate the LLM judge. |
| `reranker/` | Trains L1 (logistic) and L2 (LightGBM) rerankers using the LLM gold-standard labels and Propella features. |

## How they relate

```
human_evaluation/        llm-scorer/
  Human A/B ratings        LLM quality scores
        |                     |
        |   (validate)        |   (train labels)
        v                     v
              reranker/
        Learns to predict quality
        from Propella features
```

1. **`llm-scorer`** scores ~2000 documents using an LLM that sees raw text only (never Propella labels), producing independent quality labels.
2. **`human_evaluation`** collects pairwise human ratings, used to validate how well the LLM judge agrees with humans.
3. **`reranker`** uses the LLM labels as training signal and Propella features as input to learn a fast quality ranker.

The LLM scorer also includes a diagnostic (`llm_human_diagnostic.py`) that compares LLM and human scores on the same 130 documents.

## Quick start

```bash
# Install shared dependencies
pip install -r requirements.txt

# 1. Score documents with LLM
cd llm-scorer
export BERGET_API_KEY=...
python run_pipeline.py --sample_size 2000

# 2. Train reranker on LLM labels
cd ../reranker
learned-reranker train-l1 --config configs/finepdfs_swe_latn.yaml --labels labels.csv

# 3. Run human evaluation server
cd ../human_evaluation
uvicorn app:app --reload
```

See each component's README for full usage details.
