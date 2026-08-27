"""Baseline 1: conventional uniform-resolution 2.5D map."""
from __future__ import annotations

import time

import numpy as np

from src.geometry import cell_index
from src.map_structure import Cell, aggregate_points_to_cells


def build_uniform_map(points, semantic_labels, cfg, label_dict: dict | None = None) -> tuple[list[Cell], dict]:
    """`label_dict` is OPTIONAL (defaults to SemanticPOSS's LABEL_DICT) so a
    different dataset (e.g. RELLIS-3D) can resolve class names correctly --
    see src/map_structure.py's aggregate_points_to_cells docstring."""
    t0 = time.perf_counter()
    origin = (cfg["origin"]["x"], cfg["origin"]["y"])
    cell_size_m = float(cfg["uniform"]["cell_size_m"])

    x, y = points[:, 0], points[:, 1]
    ix = cell_index(x, origin[0], cell_size_m)
    iy = cell_index(y, origin[1], cell_size_m)
    cell_size = np.full(x.shape, cell_size_m, dtype=np.float64)
    resolution_level = np.zeros(x.shape, dtype=np.int64)
    band_id = np.full(x.shape, -1, dtype=np.int64)

    fovea_cfg = cfg["fovea"]
    n_total_input = points.shape[0]
    cells, agg_accounting = aggregate_points_to_cells(
        points, semantic_labels, ix, iy, cell_size, resolution_level, band_id,
        origin, fovea_cfg["semantic_priority"], set(fovea_cfg["dynamic_classes"]),
        fovea_cfg["geometry"]["max_expected_std_m"], fovea_cfg["weights"],
        label_dict=label_dict,
    )
    accounting = {
        "input_points": n_total_input,
        "out_of_envelope_points": 0,
        "invalid_points": n_total_input - agg_accounting["valid_points"],
        "assigned_points": agg_accounting["assigned_points"],
        "assignment_rate": agg_accounting["assigned_points"] / n_total_input if n_total_input else 0.0,
        "output_cell_count": agg_accounting["output_cell_count"],
        "latency_s": time.perf_counter() - t0,
        "method": "uniform",
        "cell_size_m": cell_size_m,
    }
    return cells, accounting
