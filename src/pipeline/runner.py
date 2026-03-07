from __future__ import annotations
from typing import Sequence

from __future__ import annotations
from typing import Sequence

from src.pipeline.context import Context
from src.pipeline.registry import registry
import src.pipeline.steps  # noqa: F401 — triggers step registrations at startup


def run(
    steps_filter: Sequence[str] | None = None,
    skip_filter: Sequence[str] | None = None,
) -> Context:
    steps = registry.steps()
    known = set(steps.keys())

    if not steps:
        raise RuntimeError("No steps registered. Add at least one step_*.py module.")

    selected = list(steps.keys())

    if steps_filter:
        unknown = [s for s in steps_filter if s not in known]
        if unknown:
            raise ValueError(
                f"Unknown step(s): {unknown}. Known steps: {sorted(known)}"
            )
        selected = [s for s in selected if s in steps_filter]

    if skip_filter:
        unknown = [s for s in skip_filter if s not in known]
        if unknown:
            raise ValueError(
                f"Unknown step(s) in --skip: {unknown}. Known steps: {sorted(known)}"
            )
        selected = [s for s in selected if s not in skip_filter]

    if not selected:
        raise RuntimeError(
            f"No steps to run after filtering. Known steps: {sorted(known)}"
        )

    ctx = Context()

    for name in selected:
        print(f"[pipeline] running {name}...")
        steps[name](ctx)
        print(f"[pipeline] {name} done.")

    return ctx
