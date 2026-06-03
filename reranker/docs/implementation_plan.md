# Implementation Plan

The first implementation is a Python package plus CLI/config workflow. A terminal wizard is included as a convenience, but config-driven runs are the primary interface.

Important design choices:

- Active priority features are the six Propella features plus FinePDFs `language_confidence`.
- `minhash_cluster_size` and `duplicate_count` are L2-only internal signals after log normalization and inversion.
- `one_sentence_description`, SimHash, `R`, and `R_z` are not used.
- Gold labels are universal quality scores, not preference-specific labels.
- L2 relevance uses 8 bins.
- L2 synthetic training groups are bucketed by `content_type` and `length`.
- Train/validation/test splitting is by stable document hash.
- L2 is optimized/evaluated for `k = [10, 25, 50, 100]`.

The prototype heuristic is deliberately marked in code, CLI output, config, and docs. It exists only to exercise the pipeline before gold labels are available.
