# A/B Test Results

## Overview

Three raters (Axel, Felix, Marcus) each completed 45 pairs across three sessions. The table below summarises the raw response counts.

| | Axel | Felix | Marcus | Total |
|---|---|---|---|---|
| Total responses | 45 | 45 | 45 | 135 |
| Skipped | 3 | 3 | 3 | 9 |
| Real pairs rated | 36 | 36 | 36 | 108 |
| Calibration (readability) | 5 | 5 | 5 | 15 |
| Calibration (substance) | 4 | 4 | 4 | 12 |

All nine skipped responses fell on real pairs (6) and readability calibration pairs (3). No substance calibration pairs were skipped. The skip rate of 6.7% is consistent with the instruction to skip only genuinely unrateable documents.

---

## 1. Timing

Time per pair in seconds (skipped pairs excluded):

| Rater | Median | P10 | P90 | Min | Max |
|---|---|---|---|---|---|
| Axel | 102 s | 41 s | 272 s | 34 s | 405 s |
| Felix | 56 s | 26 s | 115 s | 15 s | 305 s |
| Marcus | 73 s | 42 s | 105 s | 38 s | 166 s |
| **Overall** | **72 s** | **38 s** | **195 s** | — | — |

Axel spent noticeably more time per pair (median 102 s) with a long right tail, suggesting careful deliberation on difficult pairs. Felix was the fastest (median 56 s) but still well above the threshold at which shallow skimming would be a concern. Marcus was the most consistent, with a narrow P10–P90 range (42–105 s). No rater's minimum approaches a speed that would indicate random clicking.

---

## 2. Calibration accuracy

Calibration pairs have a known correct answer (the higher-quality document). Accuracy is the fraction of calibration pairs where the rater chose the correct document.

| Rater | Readability | Substance |
|---|---|---|
| Axel | 3 / 4 (75%) | 4 / 4 (100%) |
| Felix | 3 / 4 (75%) | 3 / 4 (75%) |
| Marcus | 4 / 4 (100%) | 3 / 4 (75%) |

All raters achieved at least 75% on both axes. The one skipped readability calibration pair per rater reduces the denominator to 4 rather than 5. No rater fell below chance on any axis, confirming that all three applied the rubric as intended.

---

## 3. Propella label agreement

This section measures whether human preferences aligned with propella-1's ordinal labels: `content_quality` for readability, and `educational_value` for substance. "Agree" means the rater chose the document with the higher propella label; "disagree" means they chose the lower-labelled one; "tie" means both documents had the same label on that axis.

### Overall

| Axis | Agree | Disagree | Tie | Agreement rate (excl. ties) |
|---|---|---|---|---|
| Readability vs `content_quality` | 51 | 18 | 33 | **73.9%** |
| Substance vs `educational_value` | 48 | 25 | 29 | **65.8%** |

Human preferences agreed with propella-1's readability labels on nearly three-quarters of decisive pairs. Agreement on substance was lower (65.8%), suggesting that the `educational_value` label is a harder dimension to capture, or that human raters weigh it differently from the model.

### Per rater

| Rater | Readability | Substance |
|---|---|---|
| Axel | 15 / 22 (68.2%) | 19 / 23 (82.6%) |
| Felix | 17 / 22 (77.3%) | 16 / 25 (64.0%) |
| Marcus | 19 / 25 (76.0%) | 13 / 25 (52.0%) |

Axel and Felix show the clearest split between axes: Axel agreed more with propella on substance than readability; Felix showed the reverse. Marcus's 52.0% substance agreement is close to chance and is the main driver of the lower overall substance rate.

---

## 4. Axis separability

Axis separability measures the fraction of pairs where the rater chose a different document on the readability and substance questions. A high divergence rate indicates that raters treated the two axes as genuinely independent, rather than applying a single overall quality judgment.

| Rater | Divergent pairs | Total pairs | Rate |
|---|---|---|---|
| Axel | 16 | 34 | 47.1% |
| Felix | 7 | 34 | 20.6% |
| Marcus | 7 | 34 | 20.6% |
| **Overall** | **30** | **102** | **29.4%** |

Overall, 29.4% of pairs received opposite answers on the two questions, confirming that readability and substance are meaningfully distinct dimensions in this corpus. Axel diverged considerably more often (47%) than Felix and Marcus (both 21%), which may reflect a stricter separation of the two concepts or a different reading strategy.

---

## 5. Inter-rater agreement

Nine pairs were seen by two different raters each. Agreement was measured in two ways:

- **Binary agreement**: both raters chose the same document (regardless of strength).
- **Exact agreement**: both raters chose the same document *and* the same strength label (much / slightly).

| Axis | Binary agreement | Exact agreement |
|---|---|---|
| Readability | 8 / 9 (88.9%) | 6 / 9 (66.7%) |
| Substance | 8 / 9 (88.9%) | 5 / 9 (55.6%) |

Binary agreement of 88.9% on both axes indicates strong reliability: raters reliably identified the same document as preferable. The lower exact agreement (67% and 56%) shows that while raters agreed on *direction*, they sometimes differed on *magnitude* (one rating "much better" where the other rated "slightly better"). This is expected for genuinely close pairs and does not undermine the binary signal.

---

## 6. Choice distribution

The table below counts how often each choice label was used across all non-skipped real pair responses.

| Label | Readability | Substance |
|---|---|---|
| A much better | 19 (18.6%) | 18 (17.6%) |
| A slightly better | 31 (30.4%) | 40 (39.2%) |
| B slightly better | 32 (31.4%) | 30 (29.4%) |
| B much better | 20 (19.6%) | 14 (13.7%) |

Both axes show a roughly balanced split between A and B (readability: 49% A vs 51% B; substance: 57% A vs 43% B). The slight substance imbalance towards A is an artefact of which documents happened to be labelled doc_a in a given pair and is not meaningful given randomised side assignment. The distribution of "much" vs "slightly" labels is similar across axes, with a mild tendency to use "slightly" more often — consistent with pairs being genuinely close in quality.

---

## 7. Document scores

Bradley-Terry scores were derived from the combined readability and substance outcomes (see `scoring_method.md` for details). 130 of the 263 documents in the pool received a score; the remainder did not appear in any rated pair.

| Statistic | Value |
|---|---|
| Documents scored | 130 |
| Documents with ≥1 win | 91 |
| Documents with 0 wins | 39 |
| Mean score | 0.516 |
| Median score | 0.515 |
| Standard deviation | 0.224 |
| Score range | 0.000 – 1.000 |

Score distribution by decile:

| Score range | Count |
|---|---|
| 0.0 – 0.1 | 3 |
| 0.1 – 0.2 | 4 |
| 0.2 – 0.3 | 20 |
| 0.3 – 0.4 | 24 |
| 0.4 – 0.5 | 4 |
| 0.5 – 0.6 | 25 |
| 0.6 – 0.7 | 15 |
| 0.7 – 0.8 | 23 |
| 0.8 – 0.9 | 7 |
| 0.9 – 1.0 | 5 |

The distribution is roughly bimodal, with clusters in the 0.2–0.4 and 0.5–0.8 ranges and a dip around 0.4–0.5. This reflects the Bradley-Terry model's tendency to separate documents that consistently win from those that consistently lose, with a sparse middle when few pairs directly compare mid-range documents. The 39 documents that never won any comparison all receive scores in the bottom deciles; scores for these documents carry high uncertainty given the small number of comparisons (typically 2–4 per document).

---

## 8. Summary

| Metric | Value |
|---|---|
| Total responses | 135 |
| Skip rate | 6.7% |
| Calibration accuracy (all raters, both axes) | 75–100% |
| Propella agreement — readability | 73.9% |
| Propella agreement — substance | 65.8% |
| Inter-rater binary agreement | 88.9% (both axes) |
| Axis divergence rate | 29.4% |
| Documents with a human score | 130 / 263 |

The study provides moderate evidence that propella-1's `content_quality` labels track human readability judgments (73.9% agreement). The weaker alignment on `educational_value` (65.8%, near chance for one rater) suggests that the model's substance signal is less reliable or that it captures a different notion of educational value than native-speaker raters apply. The high inter-rater binary agreement (88.9%) indicates that the results are reliable across the three raters used.
