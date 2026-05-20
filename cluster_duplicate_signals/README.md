# Propella Dataset Quality Signal Prototype

This repository contains a prototype system that quantifies dataset quality based on [Propella annotations](https://huggingface.co/datasets/openeurollm/propella-annotations).

## 🚀 Key Features

*   **Systematic decile block sampling**: Pools blocks of consecutive rows spaced evenly across the entire dataset.
*   ** Shannon Entropy metrics**: Calculates normalized Shannon entropy metrics for content length, content type, and business sector tags.
*   **Tag bucket density**: Measures category tag diversity.
*   **FAISS-based semantic duplicate detection**: Precomputes dense description embeddings with `all-MiniLM-L6-v2` and queries a `faiss.IndexFlatIP` range search for fast, exact cosine similarity matching on CPU.
*   **High Throughput**: 100,000 documents are processed and analyzed in under 8 minutes.

## 🛠️ Installation

Ensure you have Python 3.8+ installed, then install the dependencies:

```bash
pip install pandas numpy faiss-cpu sentence-transformers pyarrow huggingface_hub
```

## 📊 Running the Quality Pipeline

Run the end-to-end pipeline from the root directory:

```bash
python run_pipeline.py --block_size 10000 --num_blocks 10
```

### Options

*   `--block_size`: Number of consecutive rows in each block (default: 10,000).
*   `--num_blocks`: Number of blocks to sample (default: 10).
*   `--embedding_model`: Hugging Face model identifier for dense embeddings (default: `all-MiniLM-L6-v2`).
*   `--embedding_backend`: Choose between `sentence-transformers` or `fastembed` (default: `sentence-transformers`).
*   `--similarity_threshold`: Cosine similarity threshold for semantic duplicates (default: `0.9`).

## 🧪 Testing

Run the test suite to verify implementation correctness:

```bash
python -m unittest discover -s tests -p "test_*.py"
```
