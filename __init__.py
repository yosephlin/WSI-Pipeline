"""
Pipeline utilities for preprocessing, ROI sampling, and feature extraction.

This package also re-exports legacy symbols from the root-level pipeline.py
to avoid breaking existing imports (e.g., `from pipeline import ColabPipeline`).
"""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import List, Optional

_legacy_path = Path(__file__).resolve().parents[1] / "pipeline.py"
_legacy_module: Optional[ModuleType] = None


def _load_legacy() -> Optional[ModuleType]:
    global _legacy_module
    if _legacy_module is not None:
        return _legacy_module
    if not _legacy_path.exists():
        return None
    spec = spec_from_file_location("pipeline_legacy", _legacy_path)
    if not spec or not spec.loader:
        return None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _legacy_module = module
    for name, value in vars(module).items():
        if not name.startswith("_"):
            globals().setdefault(name, value)
    return module


def __getattr__(name: str):
    module = _load_legacy()
    if module is not None and hasattr(module, name):
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    module = _load_legacy()
    if module is None:
        return sorted(list(globals().keys()))
    public = [name for name in vars(module).keys() if not name.startswith("_")]
    return sorted(set(list(globals().keys()) + public))
