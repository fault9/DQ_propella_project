"""Smoke tests (no network, no API key needed).

    python -m pytest tests/ -v
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.llm_scorer as llm_scorer
import src.text_fetcher as text_fetcher
from src.sampler import stratified_sample
from src.llm_scorer import parse_scores, combine_scores, score_documents
from src.text_fetcher import fetch_texts

DUMMY_API = {"provider": "test", "model": "test", "openai_compatible": True}


def _make_pool(n_per=40):
    rows = []
    strata = [(0, 0), (1, 2), (2, 3), (3, 3), (4, 4), (2, 1)]  # 6 strata
    i = 0
    for (e, q) in strata:
        for _ in range(n_per):
            rows.append({"id": f"doc_{i}", "edu_ord": e, "quality_ord": q})
            i += 1
    return pd.DataFrame(rows)


# 1. Sampler
def test_sampler_count_and_strata():
    pool = _make_pool(40)  # 240 docs across 6 strata
    out, stats = stratified_sample(pool, sample_size=60, seed=42, mode="uniform")
    assert len(out) == 60
    assert "id" in out.columns
    assert "stratum_weight" in out.columns          # reweighting column present
    distinct = out[["edu_ord", "quality_ord"]].drop_duplicates()
    assert len(distinct) >= 2                        # covers multiple strata
    assert out["id"].is_unique
    assert len(stats) == 6                           # per-stratum stats for all 6


def test_uniform_more_balanced_than_proportional():
    # Skewed pool: one huge stratum, several small ones.
    import pandas as pd
    rows = []
    plan = {(0, 0): 500, (4, 4): 20, (2, 2): 20, (1, 3): 20}  # one dominant stratum
    i = 0
    for (e, q), n in plan.items():
        for _ in range(n):
            rows.append({"id": f"d{i}", "edu_ord": e, "quality_ord": q}); i += 1
    pool = pd.DataFrame(rows)
    uni, _ = stratified_sample(pool, sample_size=80, seed=0, mode="uniform")
    prop, _ = stratified_sample(pool, sample_size=80, seed=0, mode="proportional")
    # The dominant stratum should take a much larger share under proportional.
    uni_dom = (uni[["edu_ord", "quality_ord"]].apply(tuple, axis=1) == (0, 0)).mean()
    prop_dom = (prop[["edu_ord", "quality_ord"]].apply(tuple, axis=1) == (0, 0)).mean()
    assert prop_dom > uni_dom                         # proportional mirrors the skew
    assert uni_dom <= 0.40                            # uniform spreads across strata


# 2. Valid two-axis JSON parsing
def test_parse_valid_json():
    assert parse_scores('{"educational_value": 0.7, "content_quality": 0.4}') == (0.7, 0.4)
    assert parse_scores('Sure!\n{"educational_value": 0.5, "content_quality": 0.5}\nThanks') == (0.5, 0.5)
    assert parse_scores('educational_value: 0.4\ncontent_quality: 0.8') == (0.4, 0.8)


# 3. Malformed / partial responses
def test_parse_malformed():
    assert parse_scores("I think it's pretty good, maybe keep it") == (None, None)  # no numbers
    assert parse_scores('{"score": 0.5}') == (None, None)                          # wrong keys
    assert parse_scores("") == (None, None)
    assert parse_scores(None) == (None, None)
    # A partial response still yields whatever axis it carried.
    assert parse_scores('{"educational_value": 0.6}') == (0.6, None)


# 4. Clamping out-of-range (per axis)
def test_parse_clamps():
    assert parse_scores('{"educational_value": 1.5, "content_quality": -0.3}') == (1.0, 0.0)
    assert parse_scores('{"educational_value": 2, "content_quality": 0.5}') == (1.0, 0.5)


# 4b. Combination: educational-value-weighted mean (default), with geometric/mean/min alternatives.
def test_combine_scores():
    # Default is 'weighted' = 0.6*edu + 0.4*qual (config.EDU_WEIGHT / QUAL_WEIGHT).
    assert abs(combine_scores(0.9, 0.9) - 0.9) < 1e-9
    assert abs(combine_scores(0.9, 0.2) - (0.6 * 0.9 + 0.4 * 0.2)) < 1e-9    # 0.62, edu dominates
    assert abs(combine_scores(0.2, 0.9) - (0.6 * 0.2 + 0.4 * 0.9)) < 1e-9    # 0.48 < 0.62: edu weighted higher
    assert combine_scores(0.0, 0.6) > 0                                      # readable non-educational != garbage (no zero-collapse)
    assert combine_scores(0.5, None) is None                                # need both axes
    assert combine_scores(None, 0.5) is None
    # Alternatives remain available explicitly:
    assert abs(combine_scores(0.9, 0.2, mode="geometric") - (0.9 * 0.2) ** 0.5) < 1e-9
    assert combine_scores(0.9, 0.2, mode="mean") == 0.55
    assert combine_scores(0.9, 0.2, mode="min") == 0.2


# 5. Resume from partial
def test_resume_skips_scored(tmp_path, monkeypatch):
    lang = "swe_Latn"
    partial = tmp_path / f"gold_standard_{lang}_partial.parquet"
    pd.DataFrame([{"doc_id": "A", "educational_value": 0.9, "content_quality": 0.9,
                   "quality_score": 0.9, "raw_response": "cached"}]).to_parquet(partial, index=False)

    called = []

    def fake_score_one(doc_id, text, prompt_template, api_config):
        called.append(doc_id)
        return doc_id, 0.4, 0.9, 0.6, '{"educational_value": 0.4, "content_quality": 0.9}'

    monkeypatch.setattr(llm_scorer, "score_one", fake_score_one)

    gold = score_documents({"A": "text a", "B": "text b"}, "PROMPT {text}", DUMMY_API,
                           output_dir=str(tmp_path), language=lang, batch_size=2)
    by_id = dict(zip(gold["doc_id"], gold["quality_score"]))
    assert called == ["B"]              # A was skipped (already in partial)
    assert by_id["A"] == 0.9            # kept from partial
    assert by_id["B"] == 0.6            # freshly scored (combined)


# 6. Cache: load instead of streaming
def test_cache_loads_without_streaming(tmp_path, monkeypatch):
    lang = "swe_Latn"
    cache = tmp_path / f"raw_texts_{lang}.parquet"
    pd.DataFrame({"id": ["A", "B"], "text": ["alpha", "beta"]}).to_parquet(cache, index=False)

    def boom(*a, **k):
        raise AssertionError("load_dataset should not be called when cache exists")

    monkeypatch.setattr(text_fetcher, "load_dataset", boom)

    # Full cache present -> no streaming needed even with stream=True (load_dataset would raise).
    out = fetch_texts(["A", "B"], lang, output_dir=str(tmp_path), stream=True)
    assert out == {"A": "alpha", "B": "beta"}


def test_fetch_resume_keeps_partial_cache(tmp_path, monkeypatch):
    # A partial cache (only "A") + stream=False -> returns "A", never streams.
    lang = "swe_Latn"
    cache = tmp_path / f"raw_texts_{lang}.parquet"
    pd.DataFrame({"id": ["A"], "text": ["alpha"]}).to_parquet(cache, index=False)

    def boom(*a, **k):
        raise AssertionError("stream=False must not call load_dataset")

    monkeypatch.setattr(text_fetcher, "load_dataset", boom)
    out = fetch_texts(["A", "B"], lang, output_dir=str(tmp_path), stream=False)
    assert out == {"A": "alpha"}            # B missing, no streaming attempted


# 7. Output projection: two files, exact column contracts, null rows preserved
def test_write_outputs_projection(tmp_path):
    from run_pipeline import _write_outputs
    gold = pd.DataFrame([
        {"doc_id": "A", "educational_value": 0.8, "content_quality": 0.6,
         "quality_score": 0.69, "raw_response": "{...}"},
        {"doc_id": "B", "educational_value": None, "content_quality": None,
         "quality_score": None, "raw_response": "ERROR"},      # failed doc -> null row
    ])
    gp, gc = tmp_path / "gold.parquet", tmp_path / "gold.csv"
    ap, ac = tmp_path / "axes.parquet", tmp_path / "axes.csv"
    _write_outputs(gold, gp, gc, ap, ac)

    # Deliverable: EXACTLY (id, quality_score) — doc_id renamed to id; null row kept.
    for f in (gp, gc):
        d = pd.read_parquet(gp) if f.suffix == ".parquet" else pd.read_csv(gc)
        assert list(d.columns) == ["id", "quality_score"]
        assert len(d) == 2 and d["quality_score"].isna().sum() == 1

    # Axes: all four columns under doc_id; null row kept.
    cols = ["doc_id", "educational_value", "content_quality", "quality_score"]
    for f in (ap, ac):
        a = pd.read_parquet(ap) if f.suffix == ".parquet" else pd.read_csv(ac)
        assert list(a.columns) == cols
        assert len(a) == 2 and a["educational_value"].isna().sum() == 1
