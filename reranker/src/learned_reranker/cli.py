from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from learned_reranker.config import L2Mode, PipelineConfig, load_config, save_config
from learned_reranker.pipeline import preview_json, run_pipeline, write_output
from learned_reranker.schema import PRIORITY_FEATURES
from learned_reranker.evaluation import run_full_evaluation, write_evaluation_report
from learned_reranker.training import train_l1, train_l2

app = typer.Typer(help="Curate and rerank Propella-annotated FinePDFs documents.")
console = Console()


@app.command()
def run(
    config_path: Path = typer.Option(..., "--config", "-c", help="YAML config path."),
    preview: bool = typer.Option(False, "--preview", help="Print top-k JSON and write no file."),
) -> None:
    config = load_config(config_path)
    result = run_pipeline(config)
    if preview:
        console.print_json(preview_json(result, config.ranking.k))
        return
    write_output(result, config)
    if not config.l1_model_path or config.ranking.l2_mode == L2Mode.prototype:
        console.print(f"[yellow]{config.prototype.enabled_warning}[/yellow]")
    console.print(f"Wrote {result.output.height} rows to {config.ranking.output_path}")


@app.command()
def wizard(
    output_config: Path = typer.Option(
        Path("configs/wizard_finepdfs_swe_latn.yaml"),
        "--output-config",
        help="Where to save the generated config.",
    ),
) -> None:
    config = PipelineConfig()
    console.print("[bold]Learned reranker setup[/bold]")
    config.ranking.k = IntPrompt.ask("How many top documents do you want?", default=config.ranking.k)
    config.ranking.m = IntPrompt.ask("Candidate multiplier m", default=config.ranking.m)
    mode = Prompt.ask(
        "L2 mode",
        choices=[mode.value for mode in L2Mode],
        default=config.ranking.l2_mode.value,
    )
    config.ranking.l2_mode = L2Mode(mode)
    config.hard_filter.expected_lid = Prompt.ask(
        "Expected document language id",
        default=config.hard_filter.expected_lid,
    )
    credits: dict[str, float] = {}
    while True:
        remaining = 100.0
        credits.clear()
        for idx, name in enumerate(PRIORITY_FEATURES):
            if idx == len(PRIORITY_FEATURES) - 1:
                value = remaining
                console.print(f"{name}: {value:.2f} credits")
            else:
                value = FloatPrompt.ask(
                    f"{name} credits ({remaining:.2f} remaining)",
                    default=round(remaining / (len(PRIORITY_FEATURES) - idx), 2),
                )
            credits[name] = value
            remaining -= value
            if remaining < -1e-6:
                break
        if abs(sum(credits.values()) - 100.0) <= 1e-6 and all(v >= 0 for v in credits.values()):
            break
        console.print("[red]Credits must be non-negative and sum to exactly 100. Try again.[/red]")
    config.priority.credits = credits
    config.priority.mu = None
    if Confirm.ask("Set a custom personalization multiplier?", default=False):
        config.priority.mu = FloatPrompt.ask("mu", default=1.0)
    config.ranking.output_path = Path(
        Prompt.ask("Output report path", default=str(config.ranking.output_path))
    )
    save_config(config, output_config)
    console.print(f"Wrote config to {output_config}")


@app.command("train-l1")
def train_l1_command(
    config_path: Path = typer.Option(..., "--config", "-c"),
    labels: Path = typer.Option(..., "--labels"),
    output_model: Path = typer.Option(Path("models/l1_pairwise.pkl"), "--output-model"),
) -> None:
    report = train_l1(load_config(config_path), labels, output_model)
    console.print_json(data=report)


@app.command("train-l2")
def train_l2_command(
    config_path: Path = typer.Option(..., "--config", "-c"),
    labels: Path = typer.Option(..., "--labels"),
    output_model: Path = typer.Option(Path("models/l2_lambdarank.pkl"), "--output-model"),
) -> None:
    report = train_l2(load_config(config_path), labels, output_model)
    console.print_json(data=report)


@app.command()
def evaluate(
    config_path: Path = typer.Option(..., "--config", "-c"),
    labels: Path = typer.Option(..., "--labels"),
    output_report: Path = typer.Option(
        Path("outputs/evaluation_report.md"),
        "--output-report",
        help="Human-readable Markdown evaluation report.",
    ),
    output_json: Path | None = typer.Option(
        Path("outputs/evaluation_report.json"),
        "--output-json",
        help="Optional machine-readable metrics JSON.",
    ),
    seed: int = typer.Option(13, "--seed", help="RNG seed for the random baseline."),
    external_labels: Path | None = typer.Option(
        None,
        "--external-labels",
        help="Optional external labels file (e.g. human_train.csv) for a second evaluation section.",
    ),
    external_data: Path | None = typer.Option(
        None,
        "--external-data",
        help="Combined data CSV for external eval (defaults to --external-labels path).",
    ),
    external_name: str = typer.Option(
        "Human train (external)",
        "--external-name",
        help="Section title for the external evaluation block in the report.",
    ),
) -> None:
    config = load_config(config_path)
    external_evals: tuple[tuple[str, Path, Path], ...] = ()
    if external_labels is not None:
        data_path = external_data or external_labels
        external_evals = ((external_name, external_labels, data_path),)
    report = run_full_evaluation(
        config,
        labels,
        config_path=config_path,
        seed=seed,
        external_evals=external_evals,
    )
    write_evaluation_report(
        report,
        config_path=config_path,
        markdown_path=output_report,
        json_path=output_json,
    )
    console.print(f"Wrote evaluation report to {output_report}")
    if output_json is not None:
        console.print(f"Wrote evaluation metrics to {output_json}")
