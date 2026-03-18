#!/usr/bin/env python3
"""
POS Tagging Model Training Script

A modular training pipeline for transformer and LSTM-based POS tagging models.
Supports automatic OOM fallback, per-tag statistics, and configurable experiments.

Usage:
    python train_pos.py --config path/to/config.json --use-gpu nvidia
    python train_pos.py --config path/to/config.json --run-count 3 --seed 42
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

# Add src to path so imports work from scripts
sys.path.insert(0, str(Path(__file__).parent))

from data import DatasetCache
from training import (
    configure_runtime,
    load_configs,
    make_json_safe,
    print_config_summary,
    set_seed,
    slugify,
    train_one_config,
)


def _parse_bool(value: str | bool) -> bool:
    """Parse common boolean CLI text forms."""
    if isinstance(value, bool):
        return value

    raw = str(value).strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{value}'")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train POS tagging models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--use-gpu",
        choices=["amd", "nvidia", "none"],
        default="none",
        help="GPU backend to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed used to generate random run seeds",
    )
    parser.add_argument(
        "--run-count",
        type=int,
        default=1,
        help="Number of random-seed runs per config",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config or config list (required)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resources/results",
        help=(
            "Results destination. Can be a results root directory or a JSON path; "
            "results are saved as one JSON per trained model plus summary.json in "
            "a folder named after the config file"
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="resources/models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--save-models",
        type=_parse_bool,
        default=True,
        help="Whether to save trained model files (true/false)",
    )
    parser.add_argument(
        "--skip-existing-models",
        type=_parse_bool,
        default=False,
        help="Skip model runs that already exist in output results directory (true/false)",
    )

    return parser.parse_args()


def run_training(args: argparse.Namespace) -> list[dict]:
    """Run the training pipeline."""
    default_run_count = int(args.run_count)
    if default_run_count < 1:
        raise ValueError("--run-count must be >= 1")

    # Initialize
    set_seed(int(args.seed))
    configure_runtime(args.use_gpu)

    # Load configs
    configs = load_configs(args.config)
    resolver = _DependencyResolver(results_root=_resolve_results_root(args.output, args.config))
    configs = [_resolve_pick_best_dependencies(cfg, resolver) for cfg in configs]
    configs = [_resolve_derived_params(cfg) for cfg in configs]
    print_config_summary(configs)
    print(f"Default run count (--run-count): {default_run_count}")

    # Train models; each run is a standalone result row.
    if args.skip_existing_models:
        results = _load_existing_results(args.output, args.config)
        if results:
            print(f"Loaded {len(results)} existing model result(s) from output dir")
    else:
        results = []
    existing_names = {str(r.get("name")) for r in results if isinstance(r, dict) and r.get("name")}

    dataset_cache = DatasetCache()
    models_dir = Path(args.models_dir)
    total_configs = len(configs)

    for config_index, cfg in enumerate(configs, start=1):
        cfg_run_count = _resolve_cfg_run_count(cfg, default_run_count)
        run_seed_base = int(args.seed) + config_index
        run_seeds = _generate_run_seeds(base_seed=run_seed_base, run_count=cfg_run_count)

        base_name = cfg.get("name", f"{cfg.get('model_type', 'model')}_run")
        print(
            f"\n[Trainer] Config {config_index}/{total_configs}: {base_name} "
            f"(priority={cfg.get('priority', 100)}, run_count={cfg_run_count})"
        )
        if cfg.get("pick_best_from") is not None:
            print(f"[Trainer] pick_best_from={cfg.get('pick_best_from')}")
        print(f"[Trainer] Generated run seeds: {run_seeds}")

        for run_index, run_seed in enumerate(run_seeds, start=1):
            print(
                f"[Trainer] Currently running config {config_index}/{total_configs} "
                f"run {run_index}/{cfg_run_count} (seed={run_seed})"
            )
            run_name = f"{base_name}_run{run_index}_seed{run_seed}"
            if args.skip_existing_models and run_name in existing_names:
                print(f"[Trainer] SKIP existing model run: {run_name}")
                continue

            cfg_run = dict(cfg)
            cfg_run["name"] = run_name
            result = train_one_config(
                cfg_run,
                args.use_gpu,
                models_dir,
                dataset_cache,
                run_seed=run_seed,
                run_index=run_index,
                run_count=cfg_run_count,
                save_models=args.save_models,
            )
            results.append(result)
            existing_names.add(run_name)
            resolver.record_result(result)
            # Persist after each finished model so partial runs still leave JSON outputs.
            save_results(results, args.output, args.config)

    return results


NON_INHERITED_KEYS = {
    "name",
    "group",
    "hypothesis",
    "stage_id",
    "priority",
    "pick_best_from",
    "runs-count",
    "run_count",
    "seed",
    "vocab_size",
    "num_tags",
}


_MULTIPLE_EXPR = re.compile(
    r"^\s*(?P<factor>[0-9]+(?:\.[0-9]+)?)\s*x\s*(?P<ref>[A-Za-z_][A-Za-z0-9_]*)\s*$"
)


def _normalize_pick_best_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
        return out
    return []


def _cfg_identifiers(cfg: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("stage_id", "hypothesis", "name"):
        value = cfg.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized and normalized not in ids:
                ids.append(normalized)
    return ids


def _result_score(result: dict[str, Any]) -> tuple[float, float]:
    val_acc = float(result.get("best_val_accuracy", float("-inf")))
    val_loss = float(result.get("best_val_loss", float("inf")))
    return (val_acc, -val_loss)


class _DependencyResolver:
    """Resolve pick_best_from references using best prior run results."""

    def __init__(self, results_root: Path) -> None:
        self.results_root = results_root
        self._best_by_ref: dict[str, tuple[tuple[float, float, float], dict[str, Any]]] = {}
        self._loaded_disk = False

    def _ensure_loaded(self) -> None:
        if self._loaded_disk:
            return
        self._loaded_disk = True
        if not self.results_root.exists():
            return

        for path in self.results_root.rglob("*.json"):
            if path.name == "summary.json":
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    result = json.load(f)
            except Exception:
                continue
            self.record_result(result)

    def record_result(self, result: dict[str, Any]) -> None:
        cfg = result.get("config")
        if not isinstance(cfg, dict):
            return

        score = _result_score(result)
        for ref in _cfg_identifiers(cfg):
            current = self._best_by_ref.get(ref)
            if current is None or score > current[0]:
                self._best_by_ref[ref] = (score, cfg)

    def resolve(self, ref: str) -> dict[str, Any] | None:
        self._ensure_loaded()
        best = self._best_by_ref.get(ref)
        return None if best is None else dict(best[1])


def _resolve_pick_best_dependencies(
    cfg: dict[str, Any],
    resolver: _DependencyResolver,
) -> dict[str, Any]:
    refs = _normalize_pick_best_refs(cfg.get("pick_best_from"))
    if not refs:
        return cfg

    merged = dict(cfg)
    for ref in refs:
        source = resolver.resolve(ref)
        if source is None:
            raise ValueError(
                f"Could not resolve pick_best_from='{ref}'. "
                f"No prior results found under {resolver.results_root}."
            )

        for key, value in source.items():
            if key in NON_INHERITED_KEYS:
                continue
            if key not in merged:
                merged[key] = value

    return merged


def _resolve_derived_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve expressions like ff_dim='2xembed_dim' after inheritance."""
    resolved = dict(cfg)
    expr_keys = [k for k, v in resolved.items() if isinstance(v, str) and _MULTIPLE_EXPR.fullmatch(v)]
    if not expr_keys:
        return resolved

    pending = set(expr_keys)
    max_passes = len(expr_keys) + 1

    for _ in range(max_passes):
        progressed = False
        for key in list(pending):
            raw = resolved.get(key)
            if not isinstance(raw, str):
                pending.remove(key)
                continue

            match = _MULTIPLE_EXPR.fullmatch(raw)
            if match is None:
                pending.remove(key)
                continue

            ref_key = match.group("ref")
            ref_value = resolved.get(ref_key)
            if isinstance(ref_value, str):
                continue
            if isinstance(ref_value, bool) or not isinstance(ref_value, (int, float)):
                raise ValueError(
                    f"Config '{cfg.get('name', '<unnamed>')}' has invalid dependent param '{key}'. "
                    f"Reference '{ref_key}' must resolve to numeric after pick_best_from merge."
                )

            factor = float(match.group("factor"))
            value = factor * float(ref_value)
            if key in {"sentences", "maxlen", "batch_size", "epochs", "patience", "embed_dim", "num_heads", "ff_dim", "num_layers", "lstm_units", "lstm_layers", "lr_warmup_steps"}:
                if not float(value).is_integer():
                    raise ValueError(
                        f"Config '{cfg.get('name', '<unnamed>')}' computed non-integer '{key}'="
                        f"{value} from expression '{raw}'."
                    )
                resolved[key] = int(value)
            else:
                resolved[key] = float(value)

            pending.remove(key)
            progressed = True

        if not pending:
            return resolved
        if not progressed:
            break

    unresolved = ", ".join(sorted(pending))
    raise ValueError(
        f"Config '{cfg.get('name', '<unnamed>')}' has unresolved dependent parameter(s) "
        f"after pick_best_from merge: {unresolved}."
    )


def _generate_run_seeds(base_seed: int, run_count: int) -> list[int]:
    """Generate deterministic random seeds from a base seed."""
    rng = random.Random(base_seed)
    seen: set[int] = set()
    seeds: list[int] = []

    while len(seeds) < run_count:
        val = rng.randint(1, 2_147_483_647)
        if val not in seen:
            seeds.append(val)
            seen.add(val)

    return seeds


def _resolve_cfg_run_count(cfg: dict, default_run_count: int) -> int:
    """Resolve per-config run count from metadata, falling back to CLI default."""
    if "runs-count" in cfg:
        raw = cfg["runs-count"]
    elif "run_count" in cfg:
        raw = cfg["run_count"]
    else:
        return default_run_count

    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(
            f"Config '{cfg.get('name', '<unnamed>')}' has invalid runs-count={raw}. "
            "Expected integer >= 1."
        )
    return int(raw)


def _resolve_results_dir(output_path: str, config_path: str) -> Path:
    """Resolve the directory where per-model result JSON files are written."""
    target = Path(output_path)
    config_stem = Path(config_path).stem

    if target.suffix.lower() == ".json":
        if target.stem == "summary":
            return target.parent
        if target.parent.name == config_stem:
            return target.parent
        return target.parent / config_stem

    if target.name == config_stem:
        return target
    return target / config_stem


def _resolve_results_root(output_path: str, config_path: str) -> Path:
    """Resolve root folder that contains per-config result directories."""
    target = Path(output_path)
    config_stem = Path(config_path).stem

    if target.suffix.lower() == ".json":
        if target.stem == "summary":
            if target.parent.name == config_stem:
                return target.parent.parent
            return target.parent
        if target.parent.name == config_stem:
            return target.parent.parent
        return target.parent

    if target.name == config_stem:
        return target.parent
    return target


def _load_existing_results(output_path: str, config_path: str) -> list[dict]:
    """Load existing per-model result JSONs (excluding summary) for resume mode."""
    results_dir = _resolve_results_dir(output_path, config_path)
    if not results_dir.exists():
        return []

    loaded: list[dict] = []
    for file_path in sorted(results_dir.glob("*.json")):
        if file_path.name == "summary.json":
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, dict) and isinstance(payload.get("name"), str):
            loaded.append(payload)

    return loaded


def save_results(results: list[dict], output_path: str, config_path: str) -> None:
    """Save one JSON per trained model and a summary.json file."""
    results_dir = _resolve_results_dir(output_path, config_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    per_model_files: list[str] = []
    summary_models: list[dict] = []

    for idx, result in enumerate(results, start=1):
        model_name = str(result.get("name", f"model_{idx}"))
        file_name = f"{idx:03d}_{slugify(model_name)}.json"
        file_path = results_dir / file_name

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(result), f, indent=2)

        per_model_files.append(file_name)

        test_metrics = result.get("test_metrics", {})
        summary_models.append(
            {
                "name": model_name,
                "file": file_name,
                "model_type": result.get("model_type"),
                "run_index": result.get("run_index"),
                "run_seed": result.get("run_seed"),
                "best_val_accuracy": result.get("best_val_accuracy"),
                "accuracy": test_metrics.get("accuracy"),
                "f1": test_metrics.get("f1"),
            }
        )

    summary = {
        "config_name": Path(config_path).stem,
        "config_path": str(Path(config_path)),
        "total_models": len(results),
        "result_files": per_model_files,
        "models": summary_models,
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary), f, indent=2)

    print(f"\nSaved {len(results)} per-model result files to: {results_dir}")
    print(f"Saved summary to: {summary_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
