#!/usr/bin/env python3
"""
Run all JSON config files from a configs directory through src/train_pos.py.

For each config file <name>.json, this runner writes:
    resources/results/<name>/training_results.json
    resources/models/<name>/
    resources/logs/<name>.log
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def discover_config_files(configs_dir: Path) -> list[Path]:
    return sorted(
        p for p in configs_dir.glob("*.json")
        if p.is_file()
    )


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
        action="store_true",
        help="Pass --save-models to train_pos.py",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining configs even if one fails.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a config if its output JSON already exists.",
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

    config_files = discover_config_files(configs_dir)
    if not config_files:
        print(f"No JSON config files found in {configs_dir}")
        return 1

    print(f"Project root : {project_root}")
    print(f"Train script : {train_script}")
    print(f"Configs dir  : {configs_dir}")
    print(f"Results root : {results_root}")
    print(f"Models root  : {models_root}")
    print(f"Logs root    : {logs_root}")
    print(f"Python       : {python_bin}")
    print(f"GPU mode     : {args.use_gpu}")
    print(f"Found {len(config_files)} config file(s).")

    failures: list[tuple[str, int]] = []

    for idx, config_path in enumerate(config_files, start=1):
        stem = config_path.stem
        out_dir = results_root / stem
        output_json = out_dir / "training_results.json"
        model_dir = models_root / stem
        log_path = logs_root / f"{stem}.log"

        if args.skip_existing and output_json.exists():
            print(f"\n[{idx}/{len(config_files)}] Skipping {stem} (already exists: {output_json})")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        logs_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(python_bin),
            str(train_script),
            "--use-gpu", args.use_gpu,
            "--config", str(config_path),
            "--output", str(output_json),
            "--models-dir", str(model_dir),
        ]

        if args.save_models:
            cmd.append("--save-models")

        if args.extra_args:
            cmd.extend(args.extra_args)

        header = [
            "=" * 100,
            f"[{idx}/{len(config_files)}] Running config: {config_path.name}",
            f"Output JSON: {output_json}",
            f"Models dir : {model_dir}",
            f"Log file   : {log_path}",
            "Command    : " + " ".join(cmd),
            "=" * 100,
            "",
        ]

        print("\n".join(header))

        with log_path.open("w", encoding="utf-8") as logf:
            logf.write("\n".join(header))
            logf.flush()

            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                logf.write(line)

            return_code = process.wait()

        if return_code != 0:
            failures.append((config_path.name, return_code))
            print(f"\nFAILED: {config_path.name} (exit code {return_code})")
            if not args.continue_on_error:
                break
        else:
            print(f"\nDONE: {config_path.name}")

    print("\n" + "#" * 100)
    if failures:
        print("Completed with failures:")
        for name, code in failures:
            print(f"  - {name}: exit code {code}")
        return 1

    print("All config files completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())