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
    cfg = _resolve_data_root(cfg, path.parent)
    cfg = _resolve_dataset_roots(cfg, path.parent)
    cfg = _resolve_perception_checkpoint_dirs(cfg, path.parent)
    return cfg


def _resolve_data_root(cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        root = (base_dir / root).resolve()
    cfg["data"]["root"] = str(root)
    return cfg


# Config keys, per dataset entry under cfg["datasets"], that hold a filesystem
# path and need resolving relative to the repo root (mirrors _resolve_data_root
# above, generalized across however many path-like keys a dataset entry has).
_DATASET_PATH_KEYS = ("root", "pointcloud_root", "label_root")


def _resolve_dataset_roots(cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolves relative paths under cfg["datasets"][<name>][...] to absolute
    paths, same convention as _resolve_data_root. Absent/older config.yaml
    files without a `datasets:` section are left untouched (no regression for
    anyone still on a pre-multi-dataset config)."""
    if "datasets" not in cfg:
        return cfg
    cfg = copy.deepcopy(cfg)
    for name, entry in cfg["datasets"].items():
        if name == "default" or not isinstance(entry, dict):
            continue
        for key in _DATASET_PATH_KEYS:
            if key not in entry:
                continue
            p = Path(entry[key])
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            entry[key] = str(p)
    return cfg


def _resolve_perception_checkpoint_dirs(cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolves cfg["perception"]["models"][<name>]["checkpoint_dir"] to an
    absolute path, same convention as the dataset roots above. Absent in
    configs predating the perception layer -- left untouched."""
    if "perception" not in cfg or "models" not in cfg["perception"]:
        return cfg
    cfg = copy.deepcopy(cfg)
    for entry in cfg["perception"]["models"].values():
        if not isinstance(entry, dict) or "checkpoint_dir" not in entry:
            continue
        p = Path(entry["checkpoint_dir"])
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        entry["checkpoint_dir"] = str(p)
    return cfg
