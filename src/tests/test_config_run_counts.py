#!/usr/bin/env python3
"""Print expanded run counts for one or more training config files."""

import argparse
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_MODULE_PATH = PROJECT_ROOT / "src" / "training" / "config.py"
DEFAULT_CONFIGS_GLOB = "resources/configs/**/*.json"


def _load_config_module():
    """Load src/training/config.py directly to avoid package import side-effects."""
    spec = importlib.util.spec_from_file_location("training_config", CONFIG_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {CONFIG_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_config_path(raw_path: str) -> Path:
    """Resolve config path relative to project root when needed."""
    resolved = Path(raw_path)
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    return resolved


def _find_default_config_paths() -> list[str]:
    """Find all config JSON files used when --config is not provided."""
    return [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in sorted(PROJECT_ROOT.glob(DEFAULT_CONFIGS_GLOB))
        if path.is_file()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print number of expanded runs for config file(s).",
    )
    parser.add_argument(
        "--config",
        action="append",
        help=(
            "Path to config JSON file. Can be repeated. "
            "If omitted, all resources/configs/**/*.json files are tested."
        ),
    )
    args = parser.parse_args()

    config_paths = args.config or _find_default_config_paths()
    if not config_paths:
        print(f"ERROR: no config files found matching {DEFAULT_CONFIGS_GLOB}")
        return 1

    config_mod = _load_config_module()

    had_errors = False
    for raw_path in config_paths:
        resolved = _resolve_config_path(raw_path)

        if not resolved.exists():
            print(f"ERROR {raw_path}: file not found ({resolved})")
            had_errors = True
            continue

        try:
            configs = config_mod.load_configs(str(resolved))
            actual_runs = sum(int(cfg.get("runs-count", 1)) for cfg in configs)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {raw_path}: could not load config ({exc})")
            had_errors = True
            continue

        print(f"{raw_path}: {actual_runs}")

    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
