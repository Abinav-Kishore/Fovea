"""Baseline 2: distance-adaptive 2.5D map.

Cell size depends only on radial range from the sensor -- near = fine,
far = coarse -- per the SIH example. This is purely distance-driven: it does
NOT look at semantic content, so two regions at the same distance always get
the same resolution here (that's exactly the behavior FOVEA is built to
change, see src/fovea_allocator.py).
"""
from __future__ import annotations

import time

import numpy as np

from src.geometry import assign_bands, cell_index, radial_range
from src.map_structure import Cell, aggregate_points_to_cells


def effective_bands(bands: list[dict], actual_max_range_m: float | None) -> list[dict]:
    """If evaluation.max_range_m is null, we don't fabricate coverage: bands
    beyond the actual observed range still exist in config (for reproducibility)
    but will simply receive zero points, which is reported truthfully rather
    than hidden."""
    return bands


def build_distance_adaptive_map(points, semantic_labels, cfg) -> tuple[list[Cell], dict]:
    t0 = time.perf_counter()
    origin = (cfg["origin"]["x"], cfg["origin"]["y"])
    bands = cfg["distance_adaptive"]["bands"]

    ranges = radial_range(points)
    band_id, cell_size = assign_bands(ranges, bands)

    in_envelope = band_id >= 0
    x, y = points[:, 0], points[:, 1]
    ix = np.zeros(x.shape, dtype=np.int64)
    iy = np.zeros(x.shape, dtype=np.int64)
    ix[in_envelope] = cell_index(x[in_envelope], origin[0], cell_size[in_envelope])
    iy[in_envelope] = cell_index(y[in_envelope], origin[1], cell_size[in_envelope])
    resolution_level = np.zeros(x.shape, dtype=np.int64)  # single level per band in this baseline

    # points outside the configured envelope (beyond the last band's max_m) are
    # excluded from the map but counted explicitly, never silently dropped.
    n_out_of_envelope = int((~in_envelope).sum())

    fovea_cfg = cfg["fovea"]
    n_total_input = points.shape[0]
    cells, agg_accounting = aggregate_points_to_cells(
        points[in_envelope], semantic_labels[in_envelope],
        ix[in_envelope], iy[in_envelope], cell_size[in_envelope],
        resolution_level[in_envelope], band_id[in_envelope],
        origin, fovea_cfg["semantic_priority"], set(fovea_cfg["dynamic_classes"]),
        fovea_cfg["geometry"]["max_expected_std_m"], fovea_cfg["weights"],
    )
    n_invalid = int(in_envelope.sum()) - agg_accounting["valid_points"]
    accounting = {
        "input_points": n_total_input,
        "out_of_envelope_points": n_out_of_envelope,
        "invalid_points": n_invalid,
        "assigned_points": agg_accounting["assigned_points"],
        "assignment_rate": agg_accounting["assigned_points"] / n_total_input if n_total_input else 0.0,
        "output_cell_count": agg_accounting["output_cell_count"],
        "latency_s": time.perf_counter() - t0,
        "method": "distance_adaptive",
        "bands": bands,
    }
    return cells, accounting
