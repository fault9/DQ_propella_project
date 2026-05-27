# Research basis for the LLM gold-standard scorer

This document records the published research and the in-house empirical evidence behind the
key design decisions in this pipeline: the choice of evaluation criteria, the 0–1 scoring
scale, the two-axis decomposition, the combination rule, the prompt design, and the choice of
judge model. Each section states the **decision**, the **rationale**, the **references**, and
**where it lives in the code**.

> Scope note. This is a *pointwise absolute* quality scorer producing a per-document
> `quality_score ∈ [0,1]` (deliverable: `(doc_id, quality_score)`). It is one input to a
> downstream ranking/curation effort; the pairwise A/B preference data and the L2 reranker are
> owned separately. Where a decision trades against the pairwise/reward-modeling literature,
> that is called out explicitly (§B, §Limitations).

All arXiv IDs below were verified against arXiv / the ACL Anthology / NeurIPS proceedings.

---

## A. Two evaluation criteria: *educational value* + *content quality*

**Decision.** Each document is judged on exactly two axes — **educational value** and
**content quality & fluency** — and nothing else (no originality, "purity", topicality, or
length terms).

**Rationale.**
1. **Alignment with the annotation scheme we validate against.** The Propella annotations
   (`openeurollm/propella-annotations`, `finepdfs`) define these two ordinal axes verbatim as
   *Educational Value = "Potential for teaching, learning, and knowledge transfer"* and
   *Content Quality = "Overall quality of content considering writing excellence, substantive
   value, and presentation quality regardless of authorship origin"* (Propella property
   descriptions, `ellamind/propella-1-4b`). Emitting matching axes makes each one directly
   joinable/validatable against the human ordinals, and matches the two axes the colleague's
   A/B pairs use. **Note** that Propella's content_quality is a *three-leg composite*; we scope
   ours more narrowly — see "Scope of content_quality" below.
2. **Educational value is an established curation signal.** FineWeb-Edu filtered web text by an
   *educational quality* score and showed large downstream gains on knowledge/reasoning
   benchmarks (MMLU, ARC) — i.e. "does this teach?" is a high-value axis for pre-training
   corpora, not an arbitrary choice (Penedo et al. 2024).
3. **Content quality / fluency is the second orthogonal axis** standard in multi-attribute
   evaluation (HelpSteer2 scores helpfulness, **correctness**, **coherence**, complexity,
   verbosity as separate attributes; Wang et al. 2024).

**Scope of `content_quality` — writing/presentation, not substantive value (deliberate divergence).**
Propella's content_quality (quoted above) is a **three-leg composite**: writing excellence +
*substantive value* + presentation. We deliberately scope our `content_quality` axis to the
**writing/presentation quality of connected prose** (fluency, coherence, structure, extraction
integrity) and route the **substantive-value** leg to `educational_value`. This is recorded as
an explicit decision so the divergence is not read as an oversight:

- *Distinct axes, by design (the theoretical prior).* Fine-grained decomposition raises judge
  reliability **because the axes are non-overlapping** — FLASK (Ye et al. 2023) decomposes into
  distinct skill sets; HelpSteer2 (Wang et al. 2024) scores helpfulness/correctness/coherence/etc.
  as separate attributes. Folding substantive value into content_quality would re-merge it with
  `educational_value` (and with Propella's *own* separate Information Density and Reasoning
  Indicators axes), re-opening the halo/blend channel §C and §E exist to close.
- *Divergence from Propella is expected and acceptable.* Propella is a **validation reference,
  not the optimization target** (§B): the human A/B pairs are the arbiter, and Propella
  divergences are "often *definitional*" (Limitations). A fluent-but-vacuous document will score
  higher on our content_quality than on Propella's — intended, documented behaviour, not a defect.
- *Theory proposes, evidence arbitrates.* The non-overlap argument above is a **motivated
  hypothesis, not a proof** that this decomposition is the right one. It is justified
  *theoretically* by the decomposition literature but **arbitrated empirically by agreement with
  the human A/B pairs, which remain the arbiter**. If an A/B-validated `quality_score` is better
  served by returning substantive value to content_quality, this scoping is revisited — the
  literature sets the prior, the A/B evidence sets the verdict. (Do not read this section as
  "the literature says we're right, therefore we're right.")
- *Current bundling and a clean future split.* As scoped, `content_quality` still bundles three
  things — **fluency**, **coherence**, and **extraction integrity** (OCR noise / mojibake /
  decay-into-nonsense). Of these, extraction integrity maps cleanly onto Propella's *separate*
  **Content Integrity** axis (`complete … severely_degraded`) and is the natural candidate for a
  future split. We are **not** splitting it for this project: on a machine-translated PDF corpus
  integrity and fluency defects co-occur heavily, so a separate axis adds validation/stratification
  cost for marginal discriminative gain.

**Code.** `prompts/quality_prompt.txt` (the two criteria + per-axis anchors);
`src/config.py: ORDINAL_MAPS, STRATA_FEATURES` (the Propella axes used for stratification).

**References.** Penedo et al. 2024 (FineWeb/FineWeb-Edu); Wang et al. 2024 (HelpSteer2);
Ye et al. 2023 (FLASK); Propella property descriptions (`ellamind/propella-1-4b`).

---

## B. Pointwise absolute scoring (0–1), not pairwise — with eyes open

**Decision.** Score each document **independently** on an absolute 0–1 scale, rather than
ranking documents pairwise.

**Rationale.** Pairwise preference / reward modeling (Bradley–Terry 1952; Stiennon et al. 2020;
Ouyang et al. 2022 InstructGPT; Bai et al. 2022) is the RLHF-standard and is generally a
*higher-resolution* signal than absolute scores — humans and models compare more reliably than
they rate on an absolute scale. We use pointwise anyway because:
- the deliverable must be a **per-document score joinable by `doc_id`** with Propella and with
  other signals, which a pure ranking does not give;
- pairwise over ~2000 docs is O(n²) comparisons — infeasible at this budget;
- the pairwise signal **is** being produced separately (the colleague's A/B pairs), and is the
  intended **arbiter** for validating these pointwise scores.

**Known limitation (documented, not hidden).** Absolute LLM scoring saturates and clusters
(see §D). This is why the pairwise A/B set, not Propella, is treated as the ground-truth
validation target.

**Code.** `src/llm_scorer.py: score_one()` (one independent call per doc).

**References.** Bradley & Terry 1952; Stiennon et al. 2020; Ouyang et al. 2022; Bai et al. 2022.

---

## C. Decompose into two axes, then combine in code (don't ask for one blended number)

**Decision.** The LLM returns **two** scores (`educational_value`, `content_quality`); the
combined `quality_score` is computed in code, not by the model.

**Rationale.** Fine-grained, decomposed scoring is more reliable than a single holistic score:
- **FLASK** (Ye et al. 2023, ICLR 2024 Spotlight) decomposes coarse scoring into skill-level
  scoring and finds "the fine-graininess of evaluation is crucial … and increases the
  reliability of the evaluation," with higher model–human correlation.
- **HelpSteer2** (Wang et al. 2024) collects and predicts *multiple attributes separately*
  rather than one scalar reward.
- A single blended score invites a **halo/blend effect** where one strong axis inflates the
  whole rating. We observed exactly this (see §E, Empirical): a single-score prompt rated
  coherent-but-non-educational records, and an educational-*looking* but garbled math
  derivation, at 0.9.

**Code.** `prompts/quality_prompt.txt` (asks for both axes);
`src/llm_scorer.py: parse_scores(), combine_scores()`.

**References.** Ye et al. 2023 (FLASK); Wang et al. 2024 (HelpSteer2).

---

## D. Continuous 0.0–1.0 scale **with explicit anchors**

**Decision.** Each axis is scored on a continuous 0.0–1.0 scale, with an explicit calibration
band for each quartile (e.g. `0.0–0.2 nothing to learn … 0.9–1.0 richly educational`).

**Rationale.**
- **Anchoring fights score clustering.** G-Eval (Liu et al. 2023, EMNLP) documents that
  LLM evaluators clump scores and addresses it with structured criteria and a
  probability-weighted scoring scheme; the classic measurement remedy for clustering and low
  rater reliability is **explicit scale anchors** ("what is a 2 vs a 4"). Prometheus (Kim et
  al. 2023) likewise relies on a detailed score rubric (with per-level descriptions) to reach
  GPT-4-level agreement.
- **In-house evidence.** Without anchors, our single-score judge used only ~5 distinct values
  and put ~46% of docs at 0.9 with an empty 0.5–0.8 band (a coarse classifier, not a graded
  score). Per-axis anchors restored a populated, full-range distribution (see Empirical).

**Code.** `prompts/quality_prompt.txt` (the four anchor bands under each axis).

**References.** Liu et al. 2023 (G-Eval); Kim et al. 2023 (Prometheus).

---

## E. Combine the axes with an **educational-value-weighted mean**

**Decision.** `quality_score = EDU_WEIGHT·educational_value + QUAL_WEIGHT·content_quality` with
`EDU_WEIGHT=0.6`, `QUAL_WEIGHT=0.4` (configurable; `geometric`, `mean`, and `min` remain
available alternatives). Educational value is the **primary driver**; writing quality
contributes but does not dominate.

**Rationale.** Three requirements pin the combine, and a weighted mean is the option that meets
all three:
1. **Educational value must lead.** It is the high-value curation signal (FineWeb-Edu/DCLM, §G),
   so a plain 0.5/0.5 mean *undersells* it by treating writing quality as equally important.
   Weighting 0.6/0.4 keeps edu the primary driver — defensible straight from the literature —
   while still letting writing quality move the score.
2. **No degenerate point mass, so ranking survives at the bottom.** A multiplicative
   ("geometric") combine collapses every `educational_value≈0` document to ≈0 regardless of how
   readable it is — and that corner is *populated* (well-written but non-educational records:
   landing pages, minutes, obituaries). In an n≈52 pilot ~1/4 of docs piled at exactly 0 under
   geometric, conflating "readable but not educational" with "garbage." A weighted mean spreads
   them by writing quality, so **readable text outranks garbage** — the ranking signal we want.
3. **No corruption gate needed.** Corruption shows up as low on *both* axes empirically (in the
   same pilot, every `content_quality≤0.2` doc also had `educational_value≤0.3`), so a weighted
   mean already lands corrupted docs low (~0.1–0.25) without a special-case gate.

**Why not geometric (the prior choice), honestly.** Geometric's selling point is the "AND-like"
property — analogous to F-score over precision/recall — that guards against *educational-looking
garbled text* reaching the top. But the n≈52 joint-distribution check found the dangerous corner,
`content_quality≤0.2 & educational_value≥0.6`, **empty** (the axes are positively correlated at
the top; you cannot teach with garbled text). So the AND-property is protecting against a case
that does not occur here, while its cost — the point mass in (2) — is real. With geometric's
advantage neutralized and its disadvantage live, the decision flips to the weighted mean. The
trade-off we accept: a weighted mean is *not* AND-like, so a doc can score moderately on one axis
alone — which, for the populated corner (readable, non-educational), is exactly the behavior we
want, not a defect.

**Still empirical, not settled by argument.** The above is the theoretical prior; the **human A/B
pairs remain the arbiter** (§B). Both raw axes are stored, so the combine — weights included — is
recomputable and re-tunable in one line without re-running the LLM. (The pilot evidence is
directional: n≈52, one judge, partial run.)

**Code.** `src/llm_scorer.py: combine_scores()`; `src/config.py: COMBINE_MODE, EDU_WEIGHT, QUAL_WEIGHT`.

---

## F. Prompt design: specific rubric, but specificity ≠ verbosity

**Decision.** A detailed rubric — construct definitions, per-axis anchors, named failure modes
(records vs teaching; corrupted-content vs harmless markup; a coherence gate) — kept as tight
as the calibration allows.

**Rationale.**
- **Detailed criteria raise LLM-judge agreement with humans.** Zheng et al. 2023 (MT-Bench)
  show strong LLM judges reach >80% agreement with humans (≈ human–human agreement) but carry
  biases that vague prompts amplify; G-Eval and Prometheus show structured criteria / reference
  rubrics materially improve correlation. The operative mechanism is **construct
  disambiguation** — an underspecified "rate quality" makes the model fall back on its own
  training prior (which is *why* we removed the "score for pre-training an LLM" framing: it
  imported the model's curation prior instead of measuring the two target axes).
- **But longer is not strictly better.** "Lost in the Middle" (Liu et al. 2024, TACL) shows
  models attend less to mid-prompt content, so over-long rubrics bury key rules. The useful
  specificity is *new* information (a construct definition, an anchor, an edge-case rule);
  repetition only adds tokens and dilution. The practical test we apply: a clause earns its
  place only if removing it changes scores on a held-out pilot.

**Code.** `prompts/quality_prompt.txt`; cost of the rubric (≈1,183 tokens, the static prefix)
is best mitigated by prompt caching rather than by cutting construct-defining text.

**References.** Zheng et al. 2023 (MT-Bench); Liu et al. 2023 (G-Eval); Kim et al. 2023
(Prometheus); Liu et al. 2024 (Lost in the Middle).

---

## G. Educational value = "teaches / reasons", not "records / transacts"

**Decision.** Reserve high educational value for content that explains, teaches, or reasons
about generalizable knowledge; score records/transactions (financial reports, fund sheets,
audit/regulatory decisions, minutes, forms, download/landing pages) **low even when fluent**.

**Rationale.** This mirrors FineWeb-Edu's notion of "educational" content as the high-value
signal for pre-training (Penedo et al. 2024), and the broader finding that
**reasoning/"textbook-quality" data is disproportionately valuable** — phi-1 / "Textbooks Are
All You Need" (Gunasekar et al. 2023) trained a small model to strong code performance on
curated textbook-quality + synthetic-exercise data. Model-based quality filtering generally
beats heuristic filtering at corpus scale (DCLM; Li et al. 2024), which is why an LLM judge is
used here at all. (It also explains why a *correct* math/reasoning derivation has real training
value — see §I for the corrupted-tail caveat.)

**Code.** `prompts/quality_prompt.txt` (educational-value paragraph + anchors).

**References.** Penedo et al. 2024 (FineWeb-Edu); Gunasekar et al. 2023 (phi-1); Li et al. 2024
(DCLM).

---

## H. Judge model + cross-model validation

**Decision.** Default judge is an open ~70B model on an EU/OpenAI-compatible endpoint
(Llama-3.3-70B via Berget); a `--anthropic` flag runs Claude as a **different-family second
judge** for cross-validation.

**Rationale.**
- FineWeb-Edu used **Llama-3-70B-Instruct** as its educational-quality annotator (Penedo et al.
  2024), so a same-class open model is a defensible, low-cost default annotator.
- Zheng et al. 2023 document **self-enhancement bias** (a judge favors its own family's
  outputs) and other judge biases; the standard mitigation is multiple judges / a different
  family. Cross-family agreement is a far stronger signal than a model agreeing with itself —
  hence the second-judge flag. In our pilot, Claude independently reproduced the
  records→low / educational→high behaviour and additionally caught a corrupted case Llama
  missed (see Empirical), which is the evidence the *architecture* (not one model) is sound.

**Code.** `src/config.py: DEFAULT_MODEL, DEFAULT_ANTHROPIC_MODEL, default_model_for()`;
`src/llm_scorer.py: _call_api()` (native Anthropic + OpenAI-compatible paths).

**References.** Penedo et al. 2024 (FineWeb-Edu annotator); Zheng et al. 2023 (judge biases).

---

## I. Coherence gate & corrupted-content rule

**Decision.** A coherence gate caps content-quality for text stitched from unrelated fragments;
content that **decays into incoherence** (e.g. a derivation that degenerates into non-resolving
formulas) is a content-quality defect — distinct from harmless markup and from honest
truncation.

**Rationale.** Removing incoherent/boilerplate/duplicated text is core data-curation practice
(FineWeb's filtering/dedup ablations; Penedo et al. 2024). The corrupted-tail clause was added
after a math doc whose first half was a correct derivation but whose tail degraded into garbled
LaTeX; the single-score and Llama two-axis judges over-rated its fluency, while Claude's
content-quality axis caught it — illustrating both the need for the rule and the residual limit
that LLM judges cannot verify mathematical correctness.

**Code.** `prompts/quality_prompt.txt` (coherence gate + "decays into incoherence" clause +
"truncation ≠ corruption" rule).

**References.** Penedo et al. 2024 (FineWeb filtering).

---

## Empirical validation (in-house, preliminary)

From a small **n = 25** cached-text pilot (`score_sample.py`), scored against Propella's two
ordinals. Treat as directional, not a formal evaluation.

- **Anchors + decomposition removed the score collapse.** Single blended score → ~46% at 0.9,
  empty 0.5–0.8. Two anchored axes + geometric mean → full-range spread (min 0.00, median ≈0.49,
  max 0.90, populated middle).
- **Per-axis Spearman vs Propella:**
  | judge | educational | content quality | combined |
  |---|---|---|---|
  | Llama-3.3-70B | +0.73 | +0.61 | +0.61 |
  | Claude Sonnet | +0.67 | **+0.86** | **+0.74** |

  *The `combined` column (and the cross-family figure below) was computed under the then-default
  geometric combine; recompute under the 0.6/0.4 weighted default (§E) before citing — the
  per-axis columns are unaffected.*
- **Cross-family agreement** (combined, Spearman): **+0.80** — the records→low / educational→high
  behaviour is robust across two model families, i.e. an architecture effect, not a single
  model's quirk.
- **Corrupted math doc** (Propella edu/quality = 2/1): single-score 0.90 → two-axis Llama 0.85
  (content-quality 0.90, missed) → two-axis Claude **0.37** (content-quality 0.25, caught).

---

## Limitations & honest caveats

- **Propella is not ground truth for "good pre-training data."** Divergences are often
  *definitional*: Propella's content-quality scores *presentation*; its educational axis may
  credit a document's underlying work while the LLM judges the *extracted text* (e.g. download
  landing pages). The pairwise human A/B set is the intended arbiter.
- **Pointwise absolute scoring has a resolution ceiling** (saturation/clustering); §B.
- **LLM judges carry biases** (position, verbosity, self-enhancement; Zheng et al. 2023) and
  **cannot verify factual/mathematical correctness** — they pattern-match surface fluency.
- **n = 25** is a pilot; correlations are noisy. Do not over-tune the prompt to it.

---

## References

1. **Bradley, R. A., & Terry, M. E. (1952).** Rank Analysis of Incomplete Block Designs.
   *Biometrika* 39(3/4). — pairwise preference model underlying reward modeling.
2. **Stiennon, N., et al. (2020).** Learning to Summarize from Human Feedback. NeurIPS.
   arXiv:2009.01325.
3. **Ouyang, L., et al. (2022).** Training Language Models to Follow Instructions with Human
   Feedback (InstructGPT). NeurIPS. arXiv:2203.02155.
4. **Bai, Y., et al. (2022).** Training a Helpful and Harmless Assistant with RLHF.
   arXiv:2204.05862.
5. **Liu, Y., et al. (2023).** G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.
   EMNLP 2023, pp. 2511–2522. arXiv:2303.16634. <https://arxiv.org/abs/2303.16634>
6. **Zheng, L., et al. (2023).** Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
   NeurIPS 2023 (Datasets & Benchmarks). arXiv:2306.05685. <https://arxiv.org/abs/2306.05685>
7. **Ye, S., et al. (2023).** FLASK: Fine-grained Language Model Evaluation based on Alignment
   Skill Sets. ICLR 2024 (Spotlight). arXiv:2307.10928. <https://arxiv.org/abs/2307.10928>
8. **Kim, S., et al. (2023).** Prometheus: Inducing Fine-grained Evaluation Capability in
   Language Models. ICLR 2024. arXiv:2310.08491. <https://arxiv.org/abs/2310.08491>
9. **Liu, N. F., et al. (2024).** Lost in the Middle: How Language Models Use Long Contexts.
   TACL, vol. 12. arXiv:2307.03172. <https://arxiv.org/abs/2307.03172>
10. **Gunasekar, S., et al. (2023).** Textbooks Are All You Need (phi-1). arXiv:2306.11644.
    <https://arxiv.org/abs/2306.11644>
11. **Penedo, G., et al. (2024).** The FineWeb Datasets: Decanting the Web for the Finest Text
    Data at Scale (incl. FineWeb-Edu). NeurIPS 2024 (D&B). arXiv:2406.17557.
    <https://arxiv.org/abs/2406.17557>
12. **Li, J., et al. (2024).** DataComp-LM (DCLM): In Search of the Next Generation of Training
    Sets for Language Models. arXiv:2406.11794. <https://arxiv.org/abs/2406.11794>
13. **Wang, Z., et al. (2024).** HelpSteer2: Open-source Dataset for Training Top-performing
    Reward Models. NeurIPS 2024 (D&B). arXiv:2406.08673. <https://arxiv.org/abs/2406.08673>

> Citation accuracy: the LLM-as-judge, evaluation, and data-curation references (5–13) were
> verified against arXiv / ACL Anthology / NeurIPS proceedings. The foundational RLHF references
> (1–4) are standard; verify exact venue/pages before formal publication.
