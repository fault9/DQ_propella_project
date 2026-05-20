from __future__ import annotations

import csv
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from datasets import load_dataset


PROP_DATASET = "openeurollm/propella-annotations"
PROP_CONFIG = "finepdfs"
TEXT_DATASET = "HuggingFaceFW/finepdfs"

OUTPUT_COLUMNS = [
    "order_name",
    "selected_preferences",
    "id",
    "language",
    "one_sentence_description",
    "raw_text_excerpt",
    "content_quality",
    "information_density",
    "educational_value",
    "content_safety",
    "pii_presence",
    "content_integrity",
    "content_ratio",
    "content_length",
    "commercial_bias",
    "time_sensitivity",
    "audience_level",
    "reasoning_indicators",
    "full_doc_lid",
    "full_doc_lid_score",
    "language_match",
    "low_language_confidence",
    "token_count",
    "url",
]

FIELD_OPTIONS = {
    "content_quality": ["Any", "excellent", "good", "adequate", "poor", "unacceptable"],
    "information_density": ["Any", "dense", "adequate", "moderate", "thin", "empty"],
    "educational_value": ["Any", "high", "moderate", "basic", "minimal", "none"],
    "content_safety": ["Any", "safe", "mild_concerns", "nsfw", "harmful", "illegal"],
    "pii_presence": ["Any", "no_pii", "contains_pii"],
    "content_integrity": ["Any", "complete", "mostly_complete", "fragment", "severely_degraded"],
    "content_ratio": ["Any", "complete_content", "mostly_content", "mixed_content", "mostly_navigation", "minimal_content"],
    "content_length": ["Any", "substantial", "moderate", "brief", "minimal"],
    "commercial_bias": ["Any", "none", "low", "moderate", "high"],
    "time_sensitivity": ["Any", "evergreen", "time_sensitive", "outdated"],
    "audience_level": ["Any", "general", "specialist", "expert", "children"],
    "reasoning_indicators": ["Any", "none", "basic", "moderate", "strong"],
}


def default_order() -> dict[str, Any]:
    return {
        "name": "new_order",
        "language": "swe_Latn",
        "target_count": 2,
        "text_chars": 5_000,
        "filters": {field: ["Any"] for field in FIELD_OPTIONS},
        "require_low_language_confidence": False,
        "require_language_mismatch": False,
    }


def propella_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "one_sentence_description": row.get("one_sentence_description"),
        "content_quality": row.get("content_quality"),
        "information_density": row.get("information_density"),
        "educational_value": row.get("educational_value"),
        "content_safety": row.get("content_safety"),
        "pii_presence": row.get("pii_presence"),
        "content_integrity": row.get("content_integrity"),
        "content_ratio": row.get("content_ratio"),
        "content_length": row.get("content_length"),
        "commercial_bias": row.get("commercial_bias"),
        "time_sensitivity": row.get("time_sensitivity"),
        "audience_level": row.get("audience_level"),
        "reasoning_indicators": row.get("reasoning_indicators"),
    }


def value_matches(value: Any, allowed_values: list[str]) -> bool:
    if not allowed_values or "Any" in allowed_values:
        return True
    return value in allowed_values


def propella_matches(row: dict[str, Any], order: dict[str, Any]) -> bool:
    for field, allowed_values in order["filters"].items():
        if not value_matches(row.get(field), allowed_values):
            return False
    return True


def format_selected_preferences(order: dict[str, Any]) -> str:
    preferences = []

    for field, values in order["filters"].items():
        selected_values = [value for value in values if value != "Any"]
        if selected_values:
            preferences.append(f"{field}={','.join(selected_values)}")

    if order.get("require_low_language_confidence"):
        preferences.append("require_low_language_confidence=True")
    if order.get("require_language_mismatch"):
        preferences.append("require_language_mismatch=True")

    return "; ".join(preferences) if preferences else "Any"


def preferences_for_row(row: pd.Series) -> str | None:
    saved_preferences = row.get("selected_preferences")
    if pd.notna(saved_preferences) and str(saved_preferences).strip():
        return str(saved_preferences)

    order_name = row.get("order_name")
    for order in st.session_state.get("orders", []):
        if order.get("name") == order_name:
            return format_selected_preferences(order)

    return None


def annotation_values_for_row(row: pd.Series) -> str:
    values = []
    for field in FIELD_OPTIONS:
        value = row.get(field)
        if pd.notna(value) and str(value).strip():
            values.append(f"{field}={value}")
    return "; ".join(values) if values else "No annotation values available"


def normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


def build_candidate_orders(orders: list[dict[str, Any]], propella_scan: int) -> dict[str, list[dict[str, Any]]]:
    candidates = {order["name"]: [] for order in orders}

    for language in sorted({order["language"] for order in orders}):
        language_orders = [order for order in orders if order["language"] == language]
        ds = load_dataset(PROP_DATASET, PROP_CONFIG, split=language, streaming=True)

        for row in islice(ds, propella_scan):
            row_dict = propella_row_to_dict(row)

            for order in language_orders:
                if propella_matches(row_dict, order):
                    candidates[order["name"]].append(row_dict)

    return candidates


def extract_texts(
    orders: list[dict[str, Any]],
    propella_scan: int,
    finepdfs_scan: int,
    output_path: Path,
    progress_callback=None,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = build_candidate_orders(orders, propella_scan)
    id_to_candidate_orders = {}

    for order in orders:
        for candidate in candidates[order["name"]]:
            id_to_candidate_orders.setdefault(candidate["id"], []).append((order, candidate))

    rows = []
    seen = set()
    counts = {order["name"]: 0 for order in orders}

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        file.flush()

        for language in sorted({order["language"] for order in orders}):
            ds = load_dataset(TEXT_DATASET, language, split="train", streaming=True)

            for scanned, text_row in enumerate(islice(ds, finepdfs_scan), start=1):
                matches = id_to_candidate_orders.get(text_row["id"])
                if not matches:
                    if progress_callback and scanned % 1_000 == 0:
                        progress_callback(scanned, counts)
                    continue

                full_doc_lid_score = normalize_score(text_row.get("full_doc_lid_score"))
                language_match = text_row.get("language") == text_row.get("full_doc_lid")
                low_language_confidence = full_doc_lid_score is None or full_doc_lid_score < 0.70

                for order, candidate in matches:
                    key = (order["name"], candidate["id"])
                    if key in seen or counts[order["name"]] >= order["target_count"]:
                        continue
                    if order["require_low_language_confidence"] and not low_language_confidence:
                        continue
                    if order["require_language_mismatch"] and language_match:
                        continue

                    row = {
                        "order_name": order["name"],
                        "selected_preferences": format_selected_preferences(order),
                        **candidate,
                        "language": text_row.get("language"),
                        "raw_text_excerpt": str(text_row.get("text", ""))[:order["text_chars"]],
                        "full_doc_lid": text_row.get("full_doc_lid"),
                        "full_doc_lid_score": full_doc_lid_score,
                        "language_match": language_match,
                        "low_language_confidence": low_language_confidence,
                        "token_count": text_row.get("token_count"),
                        "url": text_row.get("url"),
                    }
                    rows.append(row)
                    writer.writerow(row)
                    file.flush()
                    seen.add(key)
                    counts[order["name"]] += 1

                    if progress_callback:
                        progress_callback(scanned, counts)

                if all(counts[order["name"]] >= order["target_count"] for order in orders):
                    break

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def read_existing_output(output_path: Path) -> pd.DataFrame | None:
    if not output_path.exists():
        return None
    try:
        return pd.read_csv(output_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def render_csv_viewer(df: pd.DataFrame, title: str, output_path: Path | None = None) -> None:
    st.subheader(title)

    if df.empty:
        st.info("CSV exists, but it has no result rows yet.")
        return

    if "order_name" in df.columns:
        st.write("Rows by order:")
        st.dataframe(
            df["order_name"].value_counts().rename_axis("order_name").reset_index(name="rows"),
            use_container_width=True,
            hide_index=True,
        )

    default_columns = [
        column for column in [
            "order_name",
            "selected_preferences",
            "id",
            "language",
            "one_sentence_description",
            "content_quality",
            "information_density",
            "educational_value",
            "content_safety",
            "pii_presence",
            "content_integrity",
            "content_ratio",
            "content_length",
            "full_doc_lid",
            "full_doc_lid_score",
            "language_match",
            "low_language_confidence",
            "token_count",
            "url",
        ]
        if column in df.columns
    ]
    selected_columns = st.multiselect(
        "Columns to display",
        list(df.columns),
        default=default_columns,
        key=f"columns_{title}",
    )
    st.dataframe(df[selected_columns], use_container_width=True, hide_index=True)

    if "raw_text_excerpt" in df.columns:
        show_raw_text = st.toggle(
            "Show raw text preview",
            value=False,
            key=f"show_raw_text_{title}",
        )
        if not show_raw_text:
            st.caption("Raw text is hidden. Toggle it on to preview excerpts.")
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv_bytes,
                file_name=output_path.name if output_path else "text_extractor_results.csv",
                mime="text/csv",
                key=f"download_{title}",
            )
            return

        preview_df = df.reset_index(drop=True)
        index_options = list(range(len(preview_df)))
        selected_index = st.selectbox(
            "Preview raw text by ID",
            index_options,
            format_func=lambda index: f"{index}: {preview_df.iloc[index].get('id', '')}",
            key=f"raw_id_{title}",
        )
        selected_row = preview_df.iloc[selected_index]
        selected_id = str(selected_row.get("id", ""))
        st.markdown(f"**Selected ID:** `{selected_id}`")
        selected_preferences = preferences_for_row(selected_row)
        if selected_preferences is None:
            st.markdown("**Selected preferences:** `Not stored in this CSV. Re-run the search to save them.`")
            st.markdown(f"**Selected row annotation values:** `{annotation_values_for_row(selected_row)}`")
        else:
            st.markdown(f"**Selected preferences:** `{selected_preferences}`")
        st.text_area(
            "Raw text excerpt",
            str(selected_row.get("raw_text_excerpt", "")),
            height=600,
            key=f"raw_text_{title}_{selected_index}_{selected_id}",
        )

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv_bytes,
        file_name=output_path.name if output_path else "text_extractor_results.csv",
        mime="text/csv",
        key=f"download_{title}",
    )


st.set_page_config(page_title="Propella Text Extractor", layout="wide")
st.title("Propella Text Extractor")
st.caption("Queue annotation filter orders, then retrieve matching raw FinePDFs text excerpts.")

if "orders" not in st.session_state:
    st.session_state.orders = []

with st.sidebar:
    st.header("Search Settings")
    propella_scan = st.number_input("Propella rows to scan", min_value=100, max_value=1_000_000, value=50_000, step=1_000)
    finepdfs_scan = st.number_input("FinePDFs rows to scan", min_value=100, max_value=1_000_000, value=120_000, step=1_000)
    output_path = Path(st.text_input("Output CSV", "text_extractor/outputs/text_extractor_results.csv"))
    st.caption(f"Resolved output: `{output_path}`")

st.subheader("Create Order")
with st.form("new_order_form"):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        name = st.text_input("Order name", value=f"order_{len(st.session_state.orders) + 1}")
    with col_b:
        language = st.selectbox("Language", ["swe_Latn", "eng_Latn", "deu_Latn", "fra_Latn", "dan_Latn", "nob_Latn"])
    with col_c:
        target_count = st.number_input("How many?", min_value=1, max_value=100, value=2, step=1)
    with col_d:
        text_chars = st.number_input("Raw text chars", min_value=500, max_value=20_000, value=5_000, step=500)

    filters = {}
    filter_cols = st.columns(3)
    for index, (field, options) in enumerate(FIELD_OPTIONS.items()):
        with filter_cols[index % 3]:
            filters[field] = st.multiselect(field, options, default=["Any"])

    col_e, col_f = st.columns(2)
    with col_e:
        require_low_language_confidence = st.checkbox("Require low language confidence")
    with col_f:
        require_language_mismatch = st.checkbox("Require language mismatch")

    submitted = st.form_submit_button("Add Order")

    if submitted:
        order = {
            "name": name.strip() or f"order_{len(st.session_state.orders) + 1}",
            "language": language,
            "target_count": int(target_count),
            "text_chars": int(text_chars),
            "filters": filters,
            "require_low_language_confidence": require_low_language_confidence,
            "require_language_mismatch": require_language_mismatch,
        }
        st.session_state.orders.append(order)
        st.success(f"Added order: {order['name']}")

st.subheader("Queued Orders")
if not st.session_state.orders:
    st.info("No orders queued yet.")
else:
    for index, order in enumerate(st.session_state.orders):
        with st.expander(f"{index + 1}. {order['name']} ({order['target_count']} docs)", expanded=False):
            st.json(order)

    col_run, col_clear = st.columns([1, 1])
    with col_clear:
        if st.button("Clear Orders"):
            st.session_state.orders = []
            st.rerun()

    with col_run:
        run = st.button("Run Search", type="primary")

    if run:
        status_box = st.empty()

        def show_progress(scanned: int, counts: dict[str, int]) -> None:
            counts_text = "\n".join(
                f"- `{name}`: {count}" for name, count in counts.items()
            )
            status_box.markdown(
                f"Scanned `{scanned}` FinePDFs rows for current language.\n\n"
                f"Rows written so far:\n\n{counts_text}\n\n"
                f"Partial CSV is being written to `{output_path}`."
            )

        with st.spinner("Searching Propella and FinePDFs..."):
            result = extract_texts(
                st.session_state.orders,
                int(propella_scan),
                int(finepdfs_scan),
                output_path,
                progress_callback=show_progress,
            )
            st.session_state.last_result = result

        st.success(f"Saved {len(result)} rows to {output_path}")

if "last_result" in st.session_state:
    result = st.session_state.last_result
    render_csv_viewer(result, f"Last Search Results ({len(result)} rows)", output_path)

st.divider()
st.subheader("CSV Viewer")
viewer_col_a, viewer_col_b = st.columns([1, 4])
with viewer_col_a:
    if st.button("Refresh CSV"):
        st.rerun()

existing_output = read_existing_output(output_path)
if existing_output is None:
    st.info("No CSV saved yet at the selected output path.")
else:
    render_csv_viewer(existing_output, f"Saved CSV ({len(existing_output)} rows)", output_path)
