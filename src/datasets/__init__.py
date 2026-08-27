"""Dataset registry. Adding a new dataset = one adapter module + one line
here -- src/api.py and the FOVEA engine never branch on dataset name.
"""
from __future__ import annotations

from src.datasets.base import DatasetAdapter
from src.datasets.rellis_adapter import RellisAdapter
from src.datasets.semanticposs import SemanticPOSSAdapter

_ADAPTER_CLASSES: dict[str, type] = {
    "SemanticPOSS": SemanticPOSSAdapter,
    "RELLIS-3D": RellisAdapter,
}


def available_datasets() -> list[str]:
    return list(_ADAPTER_CLASSES.keys())


def get_adapter(name: str, cfg: dict) -> DatasetAdapter:
    if name not in _ADAPTER_CLASSES:
        raise ValueError(f"Unknown dataset '{name}'. Available: {available_datasets()}")
    return _ADAPTER_CLASSES[name](cfg)


__all__ = ["DatasetAdapter", "available_datasets", "get_adapter"]
