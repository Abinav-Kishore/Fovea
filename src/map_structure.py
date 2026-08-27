"""Variable-resolution 2.5D map structure and the shared point->cell aggregation
used by all three representations (uniform, distance-adaptive, FOVEA).

A Cell is a real spatial region with its own size/bounds -- not a colored pixel
in a fixed-size array. Cells are keyed by (cell_size, ix, iy); since ix, iy are
floor((coord - origin) / cell_size), this key is a pure function of physical
location, so two different locations can never collide and one location always
resolves to the same cell regardless of which band/level produced it.

Elevation aggregation: MEDIAN z of points in the cell. This is a documented
choice (robust to outliers vs. max-Z) and is NOT a bare-earth/ground elevation
model -- a cell containing a building or tree reports that structure's median
height, not terrain height. This is a semantic-elevation representation, and
like any 2.5D representation it cannot capture multiple stacked surfaces
(e.g. a bridge over a road) -- only one elevation value exists per XY cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.geometry import cell_center
from src.semantic_labels import LABEL_DICT, UNKNOWN_CLASS_NAME


@dataclass
class Cell:
    cell_size_m: float
    resolution_level: int          # 0=coarse .. higher=finer; meaning is method-specific
    band_id: int                   # -1 if no distance-band policy applies (uniform baseline)
    ix: int
    iy: int
    center_x: float
    center_y: float
    point_count: int
    elevation_median: float
    elevation_std: float
    elevation_min: float  # min/max of the cell's own points -- additive stats, no effect on any FOVEA score/decision
    elevation_max: float
    majority_class_id: int
    majority_class_name: str
    majority_class_fraction: float
    semantic_priority: float
    geometry_score: float
    dynamic_score: float
    information_score: float
    range_m: float                 # range of cell center from origin
    weights_used: dict = field(default_factory=dict)


def _majority_class(labels: np.ndarray) -> tuple[int, float]:
    ids, counts = np.unique(labels, return_counts=True)
    top = np.argmax(counts)
    return int(ids[top]), float(counts[top] / labels.sum())


def compute_information_score(
    majority_class_id: int,
    elevation_std: float,
    priority_table: dict,
    dynamic_classes: set,
    geometry_max_std: float,
    weights: dict,
    label_dict: dict | None = None,
) -> tuple[float, float, float, float]:
    """Returns (semantic_score, geometry_score, dynamic_score, information_score).

    information_score is a weighted sum, renormalized by the sum of weights
    actually enabled (uncertainty defaults to weight 0 -- SemanticPOSS has no
    confidence field, so we do not invent one).

    `label_dict` is an OPTIONAL class-id->name table (defaults to SemanticPOSS's
    LABEL_DICT for exact backward compatibility). This lets a different dataset
    (e.g. RELLIS-3D, which has its own ontology -- see src/datasets/) resolve
    class names correctly, without this module hardcoding one dataset's ids.
    No allocation/scoring math changes based on which table is passed in.
    """
    name = (label_dict or LABEL_DICT).get(int(majority_class_id), UNKNOWN_CLASS_NAME)
    sem_score = float(priority_table.get(name, 0.1))
    geo_score = float(np.clip(elevation_std / max(geometry_max_std, 1e-9), 0.0, 1.0))
    dyn_score = 1.0 if name in dynamic_classes else 0.0

    w_sem = weights.get("semantic", 0.0)
    w_geo = weights.get("geometry", 0.0)
    w_dyn = weights.get("dynamic", 0.0)
    w_unc = weights.get("uncertainty", 0.0)
    unc_score = 0.5 if w_unc > 0 else 0.0  # neutral value only if explicitly enabled

    total_w = w_sem + w_geo + w_dyn + w_unc
    if total_w <= 0:
        info_score = 0.0
    else:
        info_score = (w_sem * sem_score + w_geo * geo_score + w_dyn * dyn_score + w_unc * unc_score) / total_w
    return sem_score, geo_score, dyn_score, float(info_score)


def aggregate_points_to_cells(
    points: np.ndarray,
    semantic_labels: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    cell_size: np.ndarray,
    resolution_level: np.ndarray,
    band_id: np.ndarray,
    origin: tuple[float, float],
    priority_table: dict,
    dynamic_classes: set,
    geometry_max_std: float,
    weights: dict,
    label_dict: dict | None = None,
) -> tuple[list[Cell], dict]:
    """Vectorized group-by over points using an integer group key, then a
    per-group aggregation pass. Returns (cells, accounting) where accounting
    tracks input/assigned/rejected point counts (no silent point loss).

    `label_dict` is OPTIONAL (defaults to SemanticPOSS's LABEL_DICT) -- see
    compute_information_score's docstring for why.
    """
    _label_dict = label_dict if label_dict is not None else LABEL_DICT
    n_input = points.shape[0]
    valid = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(points[:, 2]) & (band_id >= -1)
    valid &= ~np.isnan(cell_size)

    n_valid = int(valid.sum())
    n_rejected = n_input - n_valid

    if n_valid == 0:
        return [], {
            "input_points": n_input,
            "valid_points": n_valid,
            "rejected_points": n_rejected,
            "assigned_points": 0,
            "assignment_rate": 0.0,
        }

    pts = points[valid]
    labels = semantic_labels[valid].astype(np.int64)
    ixv = ix[valid]
    iyv = iy[valid]
    csv = cell_size[valid]
    resv = resolution_level[valid]
    bandv = band_id[valid]
    ox, oy = origin

    # integer group key: quantized cell_size (avoid float equality issues) + ix + iy.
    # All aggregation below uses pandas' compiled (Cython) groupby -- not a
    # Python-level loop over groups -- so this stays fast even with tens of
    # thousands of cells.
    cs_key = np.round(csv * 1e6).astype(np.int64)
    df = pd.DataFrame({
        "cs_key": cs_key, "ix": ixv, "iy": iyv,
        "cell_size": csv, "res": resv, "band": bandv,
        "z": pts[:, 2], "label": labels,
    })
    group_cols = ["cs_key", "ix", "iy"]
    g = df.groupby(group_cols, sort=False)

    stats = g.agg(
        cell_size=("cell_size", "first"),
        resolution_level=("res", "first"),
        band_id=("band", "first"),
        point_count=("z", "size"),
        elevation_median=("z", "median"),
        elevation_std=("z", "std"),
        elevation_min=("z", "min"),
        elevation_max=("z", "max"),
    ).reset_index()
    stats["elevation_std"] = stats["elevation_std"].fillna(0.0)  # single-point cells -> 0 spread

    # Majority (highest point-count) semantic class per cell -- deterministic
    # aggregation rule documented in the module docstring.
    label_counts = df.groupby(group_cols + ["label"], sort=False).size().rename("cnt").reset_index()
    top_idx = label_counts.groupby(group_cols, sort=False)["cnt"].idxmax()
    majority = label_counts.loc[top_idx].rename(columns={"label": "majority_class_id", "cnt": "majority_count"})
    stats = stats.merge(majority, on=group_cols, how="left")
    stats["majority_class_fraction"] = stats["majority_count"] / stats["point_count"]

    class_names = stats["majority_class_id"].map(_label_dict).fillna(UNKNOWN_CLASS_NAME)
    stats["majority_class_name"] = class_names
    stats["semantic_priority"] = class_names.map(priority_table).fillna(0.1).astype(float)
    stats["geometry_score"] = (stats["elevation_std"] / max(geometry_max_std, 1e-9)).clip(0.0, 1.0)
    stats["dynamic_score"] = class_names.isin(dynamic_classes).astype(float)

    w_sem = weights.get("semantic", 0.0)
    w_geo = weights.get("geometry", 0.0)
    w_dyn = weights.get("dynamic", 0.0)
    w_unc = weights.get("uncertainty", 0.0)
    unc_score = 0.5 if w_unc > 0 else 0.0
    total_w = w_sem + w_geo + w_dyn + w_unc
    if total_w > 0:
        stats["information_score"] = (
            w_sem * stats["semantic_priority"] + w_geo * stats["geometry_score"]
            + w_dyn * stats["dynamic_score"] + w_unc * unc_score
        ) / total_w
    else:
        stats["information_score"] = 0.0

    stats["center_x"] = cell_center(stats["ix"].to_numpy(), ox, stats["cell_size"].to_numpy())
    stats["center_y"] = cell_center(stats["iy"].to_numpy(), oy, stats["cell_size"].to_numpy())
    stats["range_m"] = np.hypot(stats["center_x"].to_numpy(), stats["center_y"].to_numpy())

    cells: list[Cell] = [
        Cell(
            cell_size_m=float(r.cell_size),
            resolution_level=int(r.resolution_level),
            band_id=int(r.band_id),
            ix=int(r.ix),
            iy=int(r.iy),
            center_x=float(r.center_x),
            center_y=float(r.center_y),
            point_count=int(r.point_count),
            elevation_median=float(r.elevation_median),
            elevation_std=float(r.elevation_std),
            elevation_min=float(r.elevation_min),
            elevation_max=float(r.elevation_max),
            majority_class_id=int(r.majority_class_id),
            majority_class_name=str(r.majority_class_name),
            majority_class_fraction=float(r.majority_class_fraction),
            semantic_priority=float(r.semantic_priority),
            geometry_score=float(r.geometry_score),
            dynamic_score=float(r.dynamic_score),
            information_score=float(r.information_score),
            range_m=float(r.range_m),
            weights_used=dict(weights),
        )
        for r in stats.itertuples(index=False)
    ]

    accounting = {
        "input_points": n_input,
        "valid_points": n_valid,
        "rejected_points": n_rejected,
        "assigned_points": n_valid,  # every valid point lands in exactly one cell
        "assignment_rate": n_valid / n_input if n_input else 0.0,
        "output_cell_count": len(cells),
    }
    return cells, accounting
