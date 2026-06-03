"""Run holdout + optional external evaluation and write the Markdown report."""

from __future__ import annotations

from pathlib import Path

from learned_reranker.config import load_config
from learned_reranker.evaluation import run_full_evaluation, write_evaluation_report


def main() -> None:
    config_path = Path("configs/finepdfs_swe_latn_llm.yaml")
    labels_path = Path("traindata/LLM_train_extended.csv")
    output_report = Path("outputs/evaluation_report.md")
    config = load_config(config_path)
    human_labels = Path("traindata/human_train.csv")
    external: tuple[tuple[str, Path, Path], ...] = ()
    if human_labels.exists():
        external = (("Human train (external)", human_labels, human_labels),)
    report = run_full_evaluation(
        config,
        labels_path,
        config_path=config_path,
        external_evals=external,
    )
    write_evaluation_report(
        report,
        config_path=config_path,
        markdown_path=output_report,
        json_path=output_report.with_suffix(".json"),
    )
    print(f"Wrote {output_report}")


if __name__ == "__main__":
    main()
