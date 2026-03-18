"""Configuration loading and defaults for training."""
from decimal import Decimal
from itertools import product
import json
import re
from typing import Any


def load_configs(config_path: str | None) -> list[dict[str, Any]]:
    """Load configurations from a JSON file."""
    if config_path is None:
        raise ValueError("--config is required. All training parameters must come from config.")

    configs = _load_from_file(config_path)
    _validate_dependency_priority(configs)
    configs = _expand_param_ranges(configs)
    _apply_defaults(configs)
    return _sort_by_priority(configs)


def _load_from_file(config_path: str) -> list[dict[str, Any]]:
    """Load and parse configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data

    raise ValueError("Config JSON must be either an object or a list of objects.")


def _apply_defaults(configs: list[dict[str, Any]]) -> None:
    """Validate required fields and enforce config contract."""
    for cfg in configs:
        if "split_seed" in cfg:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' must not define 'split_seed'. "
                "Train/test split is globally fixed in dataset.py constants."
            )
        if "test_size" in cfg:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' must not define 'test_size'. "
                "Train/test split is globally fixed in dataset.py constants."
            )
        if "seed" in cfg:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' must not define 'seed'. "
                "Seeds are generated at training time via --run-count and --seed."
            )

        if "sentences" not in cfg:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' is missing 'sentences'. "
                "All run parameters must be set in config."
            )
        if "maxlen" not in cfg:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' is missing 'maxlen'. "
                "All run parameters must be set in config."
            )

        if isinstance(cfg["sentences"], bool) or not isinstance(cfg["sentences"], int):
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' has invalid 'sentences' value. "
                "Only positive integers are supported."
            )
        if cfg["sentences"] <= 0:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' has invalid 'sentences' value. "
                "'sentences' must be > 0."
            )

        if "lr" in cfg:
            if isinstance(cfg.get("lr"), bool) or not isinstance(cfg.get("lr"), (int, float)):
                raise ValueError(
                    f"Config '{cfg.get('name', '<unnamed>')}' has invalid 'lr' value. "
                    "Only numeric values are supported after range expansion."
                )
            if float(cfg["lr"]) <= 0:
                raise ValueError(
                    f"Config '{cfg.get('name', '<unnamed>')}' has invalid 'lr' value. "
                    "'lr' must be > 0."
                )
        elif cfg.get("pick_best_from") is None:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' is missing 'lr'. "
                "Provide lr directly or define pick_best_from to inherit it from a prior phase."
            )

        priority = cfg.get("priority", DEFAULT_PRIORITY)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' has invalid 'priority'. "
                "'priority' must be an integer."
            )
        if priority < 0:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' has invalid 'priority'. "
                "'priority' must be >= 0."
            )
        cfg["priority"] = priority

        pick_best_from = cfg.get("pick_best_from")
        if pick_best_from is not None:
            if isinstance(pick_best_from, str):
                if not pick_best_from.strip():
                    raise ValueError(
                        f"Config '{cfg.get('name', '<unnamed>')}' has empty 'pick_best_from'."
                    )
            elif isinstance(pick_best_from, list):
                if not pick_best_from:
                    raise ValueError(
                        f"Config '{cfg.get('name', '<unnamed>')}' has empty 'pick_best_from' list."
                    )
                if not all(isinstance(v, str) and v.strip() for v in pick_best_from):
                    raise ValueError(
                        f"Config '{cfg.get('name', '<unnamed>')}' has invalid 'pick_best_from'. "
                        "Expected string or list of non-empty strings."
                    )
            else:
                raise ValueError(
                    f"Config '{cfg.get('name', '<unnamed>')}' has invalid 'pick_best_from'. "
                    "Expected string or list of strings."
                )

        runs_count = _normalize_runs_count(cfg.get("runs-count", cfg.get("run_count", DEFAULT_RUNS_COUNT)))
        cfg["runs-count"] = runs_count
        if "run_count" in cfg:
            del cfg["run_count"]


def _expand_param_ranges(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand configs that define one or more sweep parameter lists.

    Supported sweep syntax for any numeric hyperparameter:
        <param>: [<number>, <number>, ...]

    Expansion uses cartesian product across all swept parameters.
    Scalar parameters remain constant across generated runs.
    """
    expanded: list[dict[str, Any]] = []

    for cfg in configs:
        if cfg.get("sentences") == "max":
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' uses sentences='max', "
                "which is no longer supported. Use an explicit integer sentence count."
            )

        if "sentence_include_max" in cfg:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' uses 'sentence_include_max', "
                "which is no longer supported. Use an explicit integer or [list]."
            )

        for legacy_key in ("sentence_start", "sentence_end", "sentence_step"):
            if legacy_key in cfg:
                raise ValueError(
                    f"Config '{cfg.get('name', '<unnamed>')}' uses legacy key '{legacy_key}'. "
                    "Use sentences=[...] instead."
                )

        for key, value in cfg.items():
            if _is_range_object(value):
                raise ValueError(
                    f"Config '{cfg.get('name', '<unnamed>')}' uses deprecated range object for "
                    f"'{key}'. start/step/end sweeps are no longer supported. Use '{key}': [...]."
                )

        if "sentences" not in cfg:
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' is missing 'sentences'. "
                "Provide sentences as an integer or [list]."
            )
        derived_keys = [k for k, v in cfg.items() if _is_multiple_expr(v)]
        sweep_keys = [
            k for k, v in cfg.items()
            if k not in NON_SWEEP_KEYS and _is_numeric_list(v)
        ]
        if not sweep_keys:
            concrete = dict(cfg)
            _resolve_derived_params(concrete, derived_keys, cfg)
            assigned = {k: concrete[k] for k in derived_keys if isinstance(concrete.get(k), (int, float))}
            concrete["name"] = _name_for_params(cfg.get("name", "run"), assigned)
            expanded.append(concrete)
            continue

        sweep_values: dict[str, list[int | float]] = {}
        for key in sweep_keys:
            sweep_values[key] = _normalize_numeric_list(cfg[key], key, cfg)

        # Cartesian expansion over all swept parameters.
        for combo in product(*(sweep_values[key] for key in sweep_keys)):
            concrete = dict(cfg)
            assigned: dict[str, int | float] = {}
            for key, value in zip(sweep_keys, combo):
                concrete[key] = value
                assigned[key] = value

            _resolve_derived_params(concrete, derived_keys, cfg)
            for key in derived_keys:
                assigned[key] = concrete[key]

            concrete["name"] = _name_for_params(cfg.get("name", "run"), assigned)
            expanded.append(concrete)

    return expanded


def _is_range_object(value: Any) -> bool:
    """Return True when value is a strict deprecated {start, step, end} object."""
    return isinstance(value, dict) and set(value.keys()) == {"start", "step", "end"}


def _is_numeric_list(value: Any) -> bool:
    """Return True when value is a non-empty list/tuple of numeric scalars."""
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) == 0:
        return False
    return all((not isinstance(v, bool)) and isinstance(v, (int, float)) for v in value)


_MULTIPLE_EXPR = re.compile(
    r"^\s*(?P<factor>[0-9]+(?:\.[0-9]+)?)\s*x\s*(?P<ref>[A-Za-z_][A-Za-z0-9_]*)\s*$"
)


def _is_multiple_expr(value: Any) -> bool:
    """Return True for dependent expressions like '2xembed_dim'."""
    return isinstance(value, str) and _MULTIPLE_EXPR.fullmatch(value) is not None


def _resolve_derived_params(
    concrete: dict[str, Any],
    derived_keys: list[str],
    cfg: dict[str, Any],
) -> None:
    """Resolve dependent parameters declared as '<factor>x<other_param>'."""
    if not derived_keys:
        return

    unresolved = set(derived_keys)
    max_passes = len(derived_keys) + 1

    for _ in range(max_passes):
        progressed = False
        for key in list(unresolved):
            raw = concrete.get(key)
            if not isinstance(raw, str):
                unresolved.remove(key)
                continue

            match = _MULTIPLE_EXPR.fullmatch(raw)
            if match is None:
                unresolved.remove(key)
                continue

            ref_key = match.group("ref")
            ref_value = concrete.get(ref_key)
            if isinstance(ref_value, str):
                # The reference may itself be derived and unresolved in this pass.
                continue
            if isinstance(ref_value, bool) or not isinstance(ref_value, (int, float)):
                if cfg.get("pick_best_from") is not None:
                    # Resolve after inherited params are merged at training time.
                    continue
                raise ValueError(
                    f"Config '{cfg.get('name', '<unnamed>')}' has invalid dependent param '{key}'. "
                    f"Reference '{ref_key}' must resolve to numeric before '{key}' can be computed."
                )

            factor_dec = Decimal(match.group("factor"))
            base_dec = Decimal(str(ref_value))
            value_dec = factor_dec * base_dec

            if key in INT_SWEEP_KEYS:
                if value_dec != value_dec.to_integral_value():
                    raise ValueError(
                        f"Config '{cfg.get('name', '<unnamed>')}' computed non-integer '{key}'="
                        f"{value_dec} from expression '{raw}'."
                    )
                concrete[key] = int(value_dec)
            else:
                concrete[key] = float(value_dec)

            unresolved.remove(key)
            progressed = True

        if not unresolved:
            return
        if not progressed:
            break

    if cfg.get("pick_best_from") is not None:
        # Dependent values can be resolved later after inherited parameters are applied.
        return

    unresolved_list = ", ".join(sorted(unresolved))
    raise ValueError(
        f"Config '{cfg.get('name', '<unnamed>')}' has unresolved dependent parameter(s): "
        f"{unresolved_list}."
    )


INT_SWEEP_KEYS = {
    "sentences",
    "maxlen",
    "batch_size",
    "epochs",
    "patience",
    "embed_dim",
    "num_heads",
    "ff_dim",
    "num_layers",
    "lstm_units",
    "lstm_layers",
    "lr_warmup_steps",
}


DEFAULT_PRIORITY = 100


NON_SWEEP_KEYS = {
    "priority",
    "pick_best_from",
    "runs-count",
    "run_count",
    "stage_id",
}


DEFAULT_RUNS_COUNT = 1


def _normalize_runs_count(value: Any) -> int:
    """Validate optional per-config runs-count value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("'runs-count' must be an integer >= 1.")
    if value < 1:
        raise ValueError("'runs-count' must be >= 1.")
    return value


def _normalize_pick_best_refs(value: Any) -> list[str]:
    """Return pick_best_from as normalized list of dependency identifiers."""
    if value is None:
        return []
    if isinstance(value, str):
        val = value.strip()
        return [val] if val else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                val = item.strip()
                if val:
                    out.append(val)
        return out
    return []


def _cfg_identifier(cfg: dict[str, Any], index: int) -> str:
    """Build deterministic identifier used for dependency graph checks."""
    stage_id = cfg.get("stage_id")
    if isinstance(stage_id, str) and stage_id.strip():
        return stage_id.strip()

    hypothesis = cfg.get("hypothesis")
    if isinstance(hypothesis, str) and hypothesis.strip():
        return hypothesis.strip()

    name = cfg.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()

    return f"cfg_{index}"


def _validate_dependency_priority(configs: list[dict[str, Any]]) -> None:
    """Validate that local dependencies always have lower priority than dependents."""
    id_to_priority: dict[str, int] = {}

    for idx, cfg in enumerate(configs):
        ident = _cfg_identifier(cfg, idx)
        priority_raw = cfg.get("priority", DEFAULT_PRIORITY)
        if isinstance(priority_raw, bool) or not isinstance(priority_raw, int):
            raise ValueError(
                f"Config '{cfg.get('name', '<unnamed>')}' has invalid 'priority'. "
                "'priority' must be an integer."
            )
        if ident in id_to_priority:
            raise ValueError(
                f"Duplicate config identifier '{ident}'. Add unique 'stage_id' values."
            )
        id_to_priority[ident] = int(priority_raw)

    for idx, cfg in enumerate(configs):
        ident = _cfg_identifier(cfg, idx)
        priority = int(cfg.get("priority", DEFAULT_PRIORITY))
        refs = _normalize_pick_best_refs(cfg.get("pick_best_from"))
        for ref in refs:
            if ref not in id_to_priority:
                # Reference may point to an external phase file; local ordering cannot be checked.
                continue
            ref_priority = id_to_priority[ref]
            if priority <= ref_priority:
                raise ValueError(
                    f"Config '{ident}' depends on '{ref}' via pick_best_from, but has priority "
                    f"{priority} <= {ref_priority}. Dependents must use a strictly larger priority."
                )


def _sort_by_priority(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort configs by ascending priority while preserving deterministic order."""
    return sorted(configs, key=lambda cfg: (int(cfg.get("priority", DEFAULT_PRIORITY)), str(cfg.get("name", ""))))


def _normalize_numeric_list(values: Any, key_name: str, cfg: dict[str, Any]) -> list[int | float]:
    """Validate and normalize explicit numeric lists used for sweeps."""
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError(
            f"Config '{cfg.get('name', '<unnamed>')}' has invalid '{key_name}'. "
            "Expected a non-empty numeric list."
        )

    if any(isinstance(v, bool) for v in values):
        raise ValueError(f"{key_name} list values must be numeric, not boolean.")
    if not all(isinstance(v, (int, float)) for v in values):
        raise ValueError(f"{key_name} list values must be numeric.")

    deduped: list[int | float] = []
    seen: set[str] = set()
    min_allowed = Decimal("0") if key_name == "dropout" else Decimal("1e-30")
    bound_text = ">= 0" if key_name == "dropout" else "> 0"

    for raw in values:
        dec = Decimal(str(raw))
        if dec < min_allowed:
            raise ValueError(f"{key_name} list values must be {bound_text}.")

        normalized_key = str(dec.normalize())
        if normalized_key in seen:
            continue

        seen.add(normalized_key)
        if key_name in INT_SWEEP_KEYS:
            if dec != dec.to_integral_value():
                raise ValueError(
                    f"{key_name} list contains non-integer value {raw}. "
                    "Use integer values for this parameter."
                )
            deduped.append(int(dec))
        else:
            deduped.append(float(dec))

    if not deduped:
        raise ValueError(f"{key_name} list expansion produced no values.")

    return deduped


def _name_for_params(base_name: str, assigned: dict[str, int | float]) -> str:
    """Create a run name with placeholder replacement and stable suffix fallback."""
    out = base_name
    replaced_any = False
    for key, value in assigned.items():
        placeholder = f"{{{key}}}"
        if placeholder in out:
            out = out.replace(placeholder, _param_label(value))
            replaced_any = True

    if replaced_any:
        return out

    suffix_parts = [f"{k}_{_param_label(v)}" for k, v in assigned.items()]
    return f"{base_name}_{'_'.join(suffix_parts)}"


def _param_label(value: int | float) -> str:
    """Create a compact filesystem-safe label for numeric values."""
    if isinstance(value, int):
        return str(value)

    dec = Decimal(str(value)).normalize()
    txt = format(dec, "f").rstrip("0").rstrip(".")
    if not txt:
        txt = "0"
    return txt.replace(".", "p")


def print_config_summary(configs: list[dict[str, Any]]) -> None:
    """Print a summary of loaded configurations."""
    print(f"\nLoaded {len(configs)} config(s).")

    unique_sentences = sorted({int(cfg["sentences"]) for cfg in configs})
    unique_lrs = sorted({float(cfg["lr"]) for cfg in configs if "lr" in cfg})
    unique_maxlens = sorted({int(cfg["maxlen"]) for cfg in configs})

    print(f"Sentence-count values in sweep: {unique_sentences}")
    if unique_lrs:
        print(f"Learning-rate values in sweep: {unique_lrs}")
    else:
        print("Learning-rate values in sweep: inherited via pick_best_from")
    print(f"Max-length values in sweep: {unique_maxlens}")
