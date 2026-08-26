"""Central configuration loading. All tunable constants live in config.yaml —
nothing in this codebase should hardcode a weight, threshold, or path."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Expected a YAML file at this path "
            f"(e.g. copy config.yaml from the repo root)."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return _resolve_data_root(cfg, path.parent)


def _resolve_data_root(cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        root = (base_dir / root).resolve()
    cfg["data"]["root"] = str(root)
    return cfg
