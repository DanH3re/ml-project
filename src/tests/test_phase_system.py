#!/usr/bin/env python3
"""Validate staged phase config behavior (sweeps, inheritance, and run counts)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_MODULE_PATH = PROJECT_ROOT / "src" / "training" / "config.py"
TRAIN_POS_MODULE_PATH = PROJECT_ROOT / "src" / "train_pos.py"

PHASE_CONFIGS = {
    "phase1": ("resources/configs/phase1_architecture_13k.json", 36),
    "phase2": ("resources/configs/phase2_optimizer_13k.json", 21),
    "phase3": ("resources/configs/phase3_regularization_13k.json", 7),
    "phase4": ("resources/configs/phase4_robustness_13k.json", 10),
}


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config_module():
    return _load_module(CONFIG_MODULE_PATH, "training_config_test")


def _load_train_pos_module():
    return _load_module(TRAIN_POS_MODULE_PATH, "train_pos_test")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_phase_run_counts(config_mod) -> None:
    """Real phase configs should expand to known total trainer runs."""
    for name, (rel_path, expected_total_runs) in PHASE_CONFIGS.items():
        cfg_path = (PROJECT_ROOT / rel_path).resolve()
        expanded = config_mod.load_configs(str(cfg_path))
        total_runs = sum(int(cfg.get("runs-count", 1)) for cfg in expanded)
        _assert(
            total_runs == expected_total_runs,
            f"{name} expected {expected_total_runs} total runs, got {total_runs}",
        )


def test_missing_lr_rules(config_mod) -> None:
    """Missing lr is allowed only when pick_best_from is present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        ok_cfg_path = tmp / "ok.json"
        ok_cfg_path.write_text(
            json.dumps(
                [
                    {
                        "name": "p2_dep",
                        "stage_id": "phase2_dep",
                        "model_type": "transformer",
                        "dataset": "brown",
                        "sentences": 13000,
                        "maxlen": 30,
                        "ff_dim": "2xembed_dim",
                        "pick_best_from": "phase1_transformer_architecture_13k",
                        "priority": 20,
                        "runs-count": 1,
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        loaded = config_mod.load_configs(str(ok_cfg_path))
        _assert(len(loaded) == 1, "Dependent config should load")
        _assert(loaded[0]["ff_dim"] == "2xembed_dim", "ff_dim should remain deferred at config-load stage")

        bad_cfg_path = tmp / "bad.json"
        bad_cfg_path.write_text(
            json.dumps(
                [
                    {
                        "name": "bad_no_lr",
                        "stage_id": "bad_no_lr",
                        "model_type": "lstm",
                        "dataset": "brown",
                        "sentences": 13000,
                        "maxlen": 30,
                        "priority": 1,
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        try:
            config_mod.load_configs(str(bad_cfg_path))
            raise AssertionError("Expected missing-lr config to fail")
        except ValueError as exc:
            _assert("missing 'lr'" in str(exc), f"Unexpected error for missing lr: {exc}")


def test_unresolved_derived_without_dependency_fails(config_mod) -> None:
    """Derived expressions must resolve immediately unless dependency inheritance is used."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "bad_derived.json"
        cfg_path.write_text(
            json.dumps(
                [
                    {
                        "name": "bad_derived",
                        "stage_id": "bad_derived",
                        "model_type": "transformer",
                        "dataset": "brown",
                        "sentences": 13000,
                        "maxlen": 30,
                        "ff_dim": "2xembed_dim",
                        "lr": 0.00003,
                        "priority": 1,
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        try:
            config_mod.load_configs(str(cfg_path))
            raise AssertionError("Expected unresolved derived param to fail")
        except ValueError as exc:
            _assert("Reference 'embed_dim'" in str(exc), f"Unexpected derived-param error: {exc}")


def test_runtime_dependency_resolution(train_pos_mod) -> None:
    """pick_best_from should inherit best prior config and resolve deferred ff_dim."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        phase1_dir = root / "phase1_architecture_13k"
        phase1_dir.mkdir(parents=True, exist_ok=True)

        result_payload = {
            "name": "P1_transformer_arch_run",
            "best_val_accuracy": 0.91,
            "test_metrics": {"f1": 0.88, "accuracy": 0.90},
            "config": {
                "name": "P1_transformer_arch",
                "stage_id": "phase1_transformer_architecture_13k",
                "hypothesis": "P1_architecture_transformer_13k",
                "model_type": "transformer",
                "dataset": "brown",
                "sentences": 13000,
                "maxlen": 30,
                "num_layers": 2,
                "embed_dim": 128,
                "num_heads": 4,
                "ff_dim": 256,
                "dropout": 0.1,
                "lr": 0.00003,
                "batch_size": 32,
                "epochs": 60,
                "patience": 3,
                "runs-count": 1,
            },
        }

        (phase1_dir / "001_result.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

        resolver = train_pos_mod._DependencyResolver(results_root=root)
        dependent_cfg = {
            "name": "P2_transformer_opt",
            "stage_id": "phase2_transformer_optimizer_13k",
            "model_type": "transformer",
            "dataset": "brown",
            "sentences": 13000,
            "maxlen": 30,
            "pick_best_from": "phase1_transformer_architecture_13k",
            "ff_dim": "2xembed_dim",
            "dropout": 0.1,
            "lr": 0.00003,
            "batch_size": 32,
            "priority": 20,
            "runs-count": 1,
        }

        merged = train_pos_mod._resolve_pick_best_dependencies(dependent_cfg, resolver)
        resolved = train_pos_mod._resolve_derived_params(merged)

        _assert(resolved["embed_dim"] == 128, "embed_dim should be inherited from referenced best stage")
        _assert(resolved["num_heads"] == 4, "num_heads should be inherited from referenced best stage")
        _assert(resolved["ff_dim"] == 256, "ff_dim should resolve from inherited embed_dim")


def test_runtime_dependency_missing_results_fails(train_pos_mod) -> None:
    """Missing pick_best_from artifacts should fail with a clear error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        resolver = train_pos_mod._DependencyResolver(results_root=Path(tmpdir))
        dependent_cfg = {
            "name": "phase2_missing_dep",
            "stage_id": "phase2_missing_dep",
            "model_type": "transformer",
            "dataset": "brown",
            "sentences": 13000,
            "maxlen": 30,
            "pick_best_from": "nonexistent_stage",
            "dropout": 0.1,
            "lr": 0.00003,
            "batch_size": 32,
            "priority": 20,
        }

        try:
            train_pos_mod._resolve_pick_best_dependencies(dependent_cfg, resolver)
            raise AssertionError("Expected missing dependency to fail")
        except ValueError as exc:
            _assert("Could not resolve pick_best_from" in str(exc), f"Unexpected missing-dependency error: {exc}")


def _run_test(label: str, fn) -> bool:
    try:
        fn()
        print(f"PASS {label}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {label}: {exc}")
        return False


def main() -> int:
    config_mod = _load_config_module()
    train_pos_mod = _load_train_pos_module()

    tests = [
        ("phase run counts", lambda: test_phase_run_counts(config_mod)),
        ("missing lr rules", lambda: test_missing_lr_rules(config_mod)),
        ("unresolved derived without dependency", lambda: test_unresolved_derived_without_dependency_fails(config_mod)),
        ("runtime dependency resolution", lambda: test_runtime_dependency_resolution(train_pos_mod)),
        ("runtime missing dependency error", lambda: test_runtime_dependency_missing_results_fails(train_pos_mod)),
    ]

    passed = sum(_run_test(name, fn) for name, fn in tests)
    total = len(tests)
    print(f"\nPassed: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
