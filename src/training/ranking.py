import json
from pathlib import Path
import re
from typing import Any
from dataclasses import dataclass
from training import make_json_safe, slugify


_RUN_SUFFIX_RE = re.compile(r"_run\d+_seed\d+$")
NON_INHERITED_KEYS = {
    "name",
    "hypothesis",
    "stage_id",
    "priority",
    "pick_best_from",
    "pick_best_count",
    "pick_best_parent_name",
    "pick_best_parent_rank",
    "pick_best_parent_ref",
    "pick_best_parent_score",
    "pick_best_label",
    "runs-count",
    "run_count",
    "seed",
    "vocab_size",
    "num_tags",
}

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


@dataclass
class _RankedConfig:
    score: tuple[float, float]
    config: dict[str, Any]


def _strip_run_suffix(name: str) -> str:
    return _RUN_SUFFIX_RE.sub("", name)


def _canonicalize_result_config(cfg: dict[str, Any]) -> dict[str, Any]:
    canonical = dict(cfg)
    name = canonical.get("name")
    if isinstance(name, str) and name:
        canonical["name"] = _strip_run_suffix(name)
    canonical.pop("seed", None)
    return canonical


def _config_signature(cfg: dict[str, Any]) -> str:
    stable = {key: value for key, value in cfg.items() if key not in NON_INHERITED_KEYS}
    return json.dumps(make_json_safe(stable), sort_keys=True, separators=(",", ":"))


def _mean_score(scores: list[tuple[float, float]]) -> tuple[float, float]:
    if not scores:
        return (float("-inf"), float("-inf"))
    dims = len(scores[0])
    return tuple(sum(score[idx] for score in scores) / len(scores) for idx in range(dims))


def _cfg_sort_name(cfg: dict[str, Any]) -> str:
    for key in ("name", "stage_id", "hypothesis"):
        value = cfg.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
    return ""


def _resolve_cfg_pick_best_count(cfg: dict[str, Any]) -> int:
    raw = cfg.get("pick_best_count", 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(
            f"Config '{cfg.get('name', '<unnamed>')}' has invalid pick_best_count={raw}. "
            "Expected integer >= 1."
        )
    return int(raw)


def _pick_best_label(ref: str, rank: int) -> str:
    return f"from_{slugify(ref)}_top{rank}"


def _augment_name_with_pick_best(base_name: str, ref: str, rank: int) -> str:
    return f"{base_name}__{_pick_best_label(ref, rank)}"


def _result_score(result: dict[str, Any]) -> tuple[float, float]:
    val_acc = float(result.get("best_val_accuracy", float("-inf")))
    val_loss = float(result.get("best_val_loss", float("inf")))
    return (val_acc, -val_loss)


class _DependencyResolver:
    """Resolve pick_best_from references using top prior configs per reference."""

    def __init__(self, results_root: Path) -> None:
        self.results_root = results_root
        self._candidates_by_ref: dict[str, dict[str, dict[str, Any]]] = {}
        self._loaded_disk = False

    def _ensure_loaded(self) -> None:
        if self._loaded_disk:
            return
        self._loaded_disk = True
        if not self.results_root.exists():
            return

        for path in sorted(self.results_root.rglob("*.json")):
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

        canonical_cfg = _canonicalize_result_config(cfg)
        score = _result_score(result)
        signature = _config_signature(canonical_cfg)

        for ref in _cfg_identifiers(canonical_cfg):
            by_signature = self._candidates_by_ref.setdefault(ref, {})
            entry = by_signature.get(signature)
            if entry is None:
                entry = {"config": canonical_cfg, "scores": []}
                by_signature[signature] = entry
            entry["scores"].append(score)

    def resolve_top(self, ref: str, limit: int = 1) -> list[_RankedConfig]:
        self._ensure_loaded()
        by_signature = self._candidates_by_ref.get(ref, {})
        ranked: list[_RankedConfig] = [
            _RankedConfig(score=_mean_score(entry["scores"]), config=dict(entry["config"]))
            for entry in by_signature.values()
        ]
        ranked.sort(key=lambda item: _cfg_sort_name(item.config))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def resolve(self, ref: str) -> dict[str, Any] | None:
        ranked = self.resolve_top(ref, limit=1)
        return None if not ranked else dict(ranked[0].config)


def _expand_pick_best_dependencies(
    configs: list[dict[str, Any]],
    resolver: _DependencyResolver,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for cfg in configs:
        expanded.extend(_resolve_pick_best_dependencies(cfg, resolver))
    return expanded


def _resolve_pick_best_dependencies(
    cfg: dict[str, Any],
    resolver: _DependencyResolver,
) -> list[dict[str, Any]]:
    refs = _normalize_pick_best_refs(cfg.get("pick_best_from"))
    if not refs:
        return [cfg]

    pick_count = _resolve_cfg_pick_best_count(cfg)
    merged_variants: list[dict[str, Any]] = [dict(cfg)]

    for ref in refs:
        sources = resolver.resolve_top(ref, limit=pick_count)
        if not sources:
            raise ValueError(
                f"Could not resolve pick_best_from='{ref}'. "
                f"No prior results found under {resolver.results_root}."
            )

        next_variants: list[dict[str, Any]] = []
        for base_cfg in merged_variants:
            for rank, ranked in enumerate(sources, start=1):
                merged = dict(base_cfg)
                source = ranked.config
                for key, value in source.items():
                    if key in NON_INHERITED_KEYS:
                        continue
                    if key not in merged:
                        merged[key] = value

                merged["pick_best_parent_ref"] = ref
                merged["pick_best_parent_rank"] = rank
                merged["pick_best_parent_name"] = source.get("name")
                merged["pick_best_parent_score"] = list(ranked.score)
                merged["pick_best_label"] = _pick_best_label(ref, rank)
                merged["name"] = _augment_name_with_pick_best(
                    str(base_cfg.get("name", f"{base_cfg.get('model_type', 'model')}_run")),
                    ref,
                    rank,
                )
                next_variants.append(merged)
        merged_variants = next_variants

    return merged_variants