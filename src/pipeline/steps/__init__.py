import importlib
import pkgutil
from pathlib import Path

# Import all step_* modules so their @registry.step decorators run at startup.
for _info in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if _info.name.startswith("step_"):
        importlib.import_module(f"src.pipeline.steps.{_info.name}")
