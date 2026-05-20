# Text Extractor

Small Streamlit UI for retrieving raw FinePDFs text excerpts from Propella annotation filters.

## Run

From the repository root:

```bash
source .venv/bin/activate
python -m pip install streamlit datasets pandas
streamlit run text_extractor/app.py --server.headless true --browser.gatherUsageStats false
```

## How It Works

1. Create one or more extraction orders with dropdown filters for Propella annotation fields.
2. Queue the orders instead of searching immediately.
3. Run one search across Propella annotations and linked FinePDFs raw text.
4. Export a CSV with IDs, annotation levels, one-sentence descriptions, raw text excerpts, language-confidence metadata, and URLs.

Generated CSV files are saved under:

```text
text_extractor/outputs/
```
