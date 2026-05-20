# Text Extractor

Streamlit UI for finding human-review examples from Propella annotations and
retrieving the matching FinePDFs raw text excerpts.

The tool is meant for creating small, targeted samples for human evaluation.
For example, it can retrieve documents where `content_quality = good` and
`educational_value = none`, or where `content_integrity = fragment` but
`content_quality = excellent`.

Propella remains the source of annotation filters. FinePDFs is only joined by
document `id` to retrieve raw text excerpts and supporting metadata such as URL,
token count, and language-ID scores.

## Run

From the repository root:

```bash
source .venv/bin/activate
python -m pip install streamlit datasets pandas
streamlit run text_extractor/app.py --server.headless true --browser.gatherUsageStats false
```

Then open:

```text
http://127.0.0.1:8501
```

## How It Works

1. Create one or more extraction orders with dropdown filters for Propella
   annotation fields.
2. Each order can request a target number of documents and a raw text excerpt
   length.
3. Queue multiple orders before searching. This makes it possible to ask for
   several different edge-case types in one run.
4. Run one search across Propella annotations and linked FinePDFs text.
5. The app writes matching rows incrementally to CSV, so partial results remain
   available even if a long search is stopped early.
6. View the saved CSV directly in the app, preview raw text by document ID, and
   download the CSV.

## Data Sources

Annotation filters come from:

```text
openeurollm/propella-annotations
config: finepdfs
```

Raw text and metadata are joined from:

```text
HuggingFaceFW/finepdfs
```

The join key is:

```text
id
```

## Typical Use

Examples of useful orders:

```text
Dense but not educational:
  information_density = dense
  educational_value = none/minimal
  content_quality = good/excellent

Complete but thin:
  content_integrity = complete
  content_ratio = complete_content/mostly_content
  information_density = thin/empty

Fragment but high quality:
  content_integrity = fragment/severely_degraded
  content_quality = good/excellent

Educational but low content quality:
  educational_value = high/moderate
  content_quality = poor/unacceptable
```

## Output

Generated CSV files are saved under:

```text
text_extractor/outputs/
```

Default output path:

```text
text_extractor/outputs/text_extractor_results.csv
```

Important output columns:

```text
order_name
selected_preferences
id
language
one_sentence_description
raw_text_excerpt
content_quality
information_density
educational_value
content_safety
pii_presence
content_integrity
content_ratio
content_length
full_doc_lid
full_doc_lid_score
language_match
low_language_confidence
token_count
url
```

`selected_preferences` records the non-`Any` dropdown values used for the order
that produced each row. Older CSVs created before this field existed will not
contain those preferences.

## Notes

- Raw text excerpts are hidden by default in the UI and can be toggled on.
- The CSV viewer is at the bottom of the app.
- Use **Clear Orders And Results** to reset queued orders and delete the current
  output CSV.
- Output CSVs are ignored by git, because they are generated artifacts and may
  contain raw text excerpts.
