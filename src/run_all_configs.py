#!/usr/bin/env python3
"""
Run all JSON config files from a configs directory through src/train_pos.py.

For each config file <name>.json, this runner writes:
    resources/results/<name>/summary.json (+ one JSON per trained model)
    resources/models/<name>/
    resources/logs/<name>.log
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'rich'. Install it with: pip install rich"
    ) from exc


console = Console()

try:
    from training import _DependencyResolver, _expand_pick_best_dependencies
except ImportError as exc:
    raise SystemExit(
        "Missing training dependencies. Ensure src/ is on PYTHONPATH."
    ) from exc


def discover_config_files(configs_dir: Path) -> list[Path]:
    return sorted(
        p for p in configs_dir.glob("*.json")
        if p.is_file()
    )


def sort_config_files_by_priority(config_mod, config_files: list[Path]) -> list[Path]:
    """Sort config files by their minimum expanded config priority (ascending)."""
    max_priority = 10**9
    ranked: list[tuple[int, str, Path]] = []

    for path in config_files:
        priority = max_priority
        try:
            expanded = config_mod.load_configs(str(path))
            if expanded:
                priority = min(int(cfg.get("priority", 100)) for cfg in expanded)
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"[yellow]Priority read failed for {path.name}; placing last ({exc}).[/yellow]"
            )

        ranked.append((priority, path.name, path))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [path for _, _, path in ranked]


def ensure_train_script(project_root: Path, train_script: Path) -> Path:
    candidate = train_script if train_script.is_absolute() else (project_root / train_script)
    if not candidate.exists():
        raise FileNotFoundError(f"train_pos.py not found at: {candidate}")
    return candidate.resolve()


def resolve_python(project_root: Path, explicit_python: Path | None) -> Path:
    if explicit_python is not None:
        py = explicit_python.expanduser()
        if not py.exists():
            raise FileNotFoundError(f"Explicit python not found: {py}")
        return py

    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python

    return Path(sys.executable)


def _status_style(status: str) -> str:
    if status == "DONE":
        return "green"
    if status == "SKIPPED":
        return "yellow"
    return "red"


def _load_config_module(project_root: Path):
    """Load src/training/config.py directly to count expanded configs safely."""
    cfg_path = project_root / "src" / "training" / "config.py"
    spec = importlib.util.spec_from_file_location("training_config_for_runner", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _estimate_planned_runs(
    config_mod,
    config_path: Path,
    results_root: Path,
) -> tuple[int | None, int | None]:
    """Return (expanded_configs, planned_runs) or (None, None) if estimate fails."""
    try:
        expanded = config_mod.load_configs(str(config_path))
        resolver = _DependencyResolver(results_root=results_root)
        expanded = _expand_pick_best_dependencies(expanded, resolver)
        expanded_count = len(expanded)
        planned_runs = sum(int(cfg.get("runs-count", 1)) for cfg in expanded)
        return expanded_count, planned_runs
    except Exception:  # noqa: BLE001
        return None, None


def _count_existing_result_models(results_dir: Path) -> int:
    """Count existing per-model result JSON files (excluding summary)."""
    if not results_dir.exists():
        return 0
    return sum(1 for p in results_dir.glob("*.json") if p.is_file() and p.name != "summary.json")


def _print_run_banner(
    idx: int,
    total: int,
    config_name: str,
    expanded_configs: int | None,
    planned_runs: int | None,
    results_dir: Path,
    summary_json: Path,
    model_dir: Path,
    log_path: Path,
    cmd: list[str],
) -> str:
    expanded_text = str(expanded_configs) if expanded_configs is not None else "unknown"
    planned_text = str(planned_runs) if planned_runs is not None else "unknown"
    header_lines = [
        f"Config file: {idx}/{total}",
        f"Config: {config_name}",
        f"Expanded configs in file: {expanded_text}",
        "Run count source: config runs-count (fallback: 1)",
        f"Planned trainer runs for this file: {planned_text}",
        f"Results dir: {results_dir}",
        f"Summary: {summary_json}",
        f"Models: {model_dir}",
        f"Log: {log_path}",
        f"Command: {' '.join(cmd)}",
    ]
    text = "\n".join(header_lines)
    console.print(Panel(text, title="Config Execution", border_style="cyan"))
    return text


def _stream_output_to_console_and_log(process: subprocess.Popen, logf) -> None:
    """Tee child process output to both terminal and log without rich formatting."""
    assert process.stdout is not None
    while True:
        chunk = process.stdout.read(8192)
        if not chunk:
            break
        sys.stdout.write(chunk)
        sys.stdout.flush()
        logf.write(chunk)
        logf.flush()

def main() -> int:
    parser = argparse.ArgumentParser(description="Run train_pos.py for all config JSON files.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/home/dan-gavriluta/Coding/ml-project"),
        help="Project root directory.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("/home/dan-gavriluta/Coding/ml-project/resources/configs"),
        help="Directory containing config JSON files.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("src/train_pos.py"),
        help="Path to train_pos.py, absolute or relative to project root.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Python interpreter to use. Defaults to <project-root>/.venv/bin/python if it exists.",
    )
    parser.add_argument(
        "--use-gpu",
        choices=["amd", "nvidia", "none"],
        default="amd",
        help="GPU mode passed to train_pos.py.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed forwarded to train_pos.py for run seed generation.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Root folder for per-config results. Defaults to <project-root>/resources/results",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Root folder for per-config model folders. Defaults to <project-root>/resources/models",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=None,
        help="Root folder for per-config logs. Defaults to <project-root>/resources/logs",
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether train_pos.py should save model files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining configs even if one fails.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip already-trained models and skip a config file only when all planned models "
            "are already present."
        ),
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args appended to every train_pos.py call.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    configs_dir = args.configs_dir.resolve()
    train_script = ensure_train_script(project_root, args.train_script)
    python_bin = resolve_python(project_root, args.python)

    results_root = (args.results_root or (project_root / "resources" / "results")).resolve()
    models_root = (args.models_root or (project_root / "resources" / "models")).resolve()
    logs_root = (args.logs_root or (project_root / "resources" / "logs")).resolve()

    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")

    config_mod = _load_config_module(project_root)

    config_files = discover_config_files(configs_dir)
    if not config_files:
        console.print(f"[red]No JSON config files found in {configs_dir}[/red]")
        return 1

    config_files = sort_config_files_by_priority(config_mod, config_files)

    overview = Table(show_header=False, box=None, pad_edge=False)
    overview.add_row("Project root", str(project_root))
    overview.add_row("Train script", str(train_script))
    overview.add_row("Configs dir", str(configs_dir))
    overview.add_row("Results root", str(results_root))
    overview.add_row("Models root", str(models_root))
    overview.add_row("Logs root", str(logs_root))
    overview.add_row("Python", str(python_bin))
    overview.add_row("GPU mode", args.use_gpu)
    overview.add_row("Base seed", str(args.seed))
    overview.add_row("Save models", str(args.save_models))
    overview.add_row("Config files", str(len(config_files)))
    overview.add_row("File order", "priority asc (then filename)")
    console.print(Panel(overview, title="Run All Configs", border_style="green"))

    failures: list[tuple[str, int]] = []
    summary_rows: list[tuple[str, str, int, str]] = []
    skipped_count = 0

    for idx, config_path in enumerate(config_files, start=1):
        stem = config_path.stem
        out_dir = results_root / stem
        summary_json = out_dir / "summary.json"
        model_dir = models_root / stem
        log_path = logs_root / f"{stem}.log"

        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        logs_root.mkdir(parents=True, exist_ok=True)

        expanded_configs, planned_runs = _estimate_planned_runs(
            config_mod,
            config_path,
            results_root,
        )

        existing_models = _count_existing_result_models(out_dir)
        if args.skip_existing and planned_runs is not None and existing_models >= planned_runs:
            skipped_count += 1
            summary_rows.append((config_path.name, "SKIPPED", 0, str(log_path)))
            console.print(
                f"[yellow]SKIPPED[/yellow] {config_path.name} "
                f"(existing models {existing_models}/{planned_runs})"
            )
            continue

        if args.skip_existing and existing_models > 0 and planned_runs is not None:
            console.print(
                f"[cyan]RESUME[/cyan] {config_path.name} "
                f"(existing models {existing_models}/{planned_runs})"
            )

        cmd = [
            str(python_bin),
            str(train_script),
            "--use-gpu", args.use_gpu,
            "--seed", str(args.seed),
            "--config", str(config_path),
            "--output", str(out_dir),
            "--models-dir", str(model_dir),
            "--save-models", str(args.save_models),
            "--skip-existing-models", str(args.skip_existing),
        ]

        if args.extra_args:
            cmd.extend(args.extra_args)

        banner_text = _print_run_banner(
            idx,
            len(config_files),
            config_path.name,
            expanded_configs,
            planned_runs,
            out_dir,
            summary_json,
            model_dir,
            log_path,
            cmd,
        )

        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(banner_text + "\n\n")
            logf.flush()

            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            _stream_output_to_console_and_log(process, logf)
            return_code = process.wait()

        status = "DONE" if return_code == 0 else "FAILED"
        summary_rows.append((config_path.name, status, int(return_code), str(log_path)))

        if return_code != 0:
            failures.append((config_path.name, return_code))
            console.print(
                f"[red]FAILED[/red] {config_path.name} (exit code {return_code})"
            )
            if not args.continue_on_error:
                break
        else:
            console.print(f"[green]DONE[/green] {config_path.name}")

    summary = Table(title="Execution Summary")
    summary.add_column("Config", style="cyan")
    summary.add_column("Status")
    summary.add_column("Exit", justify="right")
    summary.add_column("Log", overflow="fold")
    for config_name, status, exit_code, log in summary_rows:
        summary.add_row(
            config_name,
            f"[{_status_style(status)}]{status}[/{_status_style(status)}]",
            str(exit_code),
            log,
        )
    console.print(summary)

    done_count = sum(1 for _, status, _, _ in summary_rows if status == "DONE")
    fail_count = sum(1 for _, status, _, _ in summary_rows if status == "FAILED")
    console.print(
        Panel(
            f"Done: {done_count}  |  Skipped: {skipped_count}  |  Failed: {fail_count}",
            title="Totals",
            border_style="magenta",
        )
    )

    if failures:
        console.print("[red]Completed with failures:[/red]")
        for name, code in failures:
            console.print(f"  - {name}: exit code {code}")
        return 1

    console.print("[bold green]All config files completed successfully.[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())