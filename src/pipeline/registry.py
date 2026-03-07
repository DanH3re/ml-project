from __future__ import annotations
from collections import OrderedDict
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.context import Context

StepFn = Callable[["Context"], None]


class Registry:
    def __init__(self) -> None:
        self._steps: list[tuple[int, str, StepFn]] = []

    def step(self, *, order: int) -> Callable[[StepFn], StepFn]:
        """Decorator that registers a step function with the given execution order."""
        def decorator(fn: StepFn) -> StepFn:
            self._steps.append((order, fn.__name__, fn))
            return fn
        return decorator

    def steps(self) -> OrderedDict[str, StepFn]:
        """Return registered steps sorted by order."""
        return OrderedDict(
            (name, fn) for _, name, fn in sorted(self._steps, key=lambda x: x[0])
        )


registry = Registry()
