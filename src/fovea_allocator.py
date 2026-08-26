"""FOVEA v0.1 allocation engine.

PROTOTYPE HEURISTIC -- not a validated or final mathematical formulation.
Implements a transparent, configurable information-aware refinement on top of
the distance-adaptive envelope (src/distance_baseline.py):

  - Each distance band's configured cell_size_m (config.yaml ->
    distance_adaptive.bands) is the COARSEST cell allowed in that band --
    identical to the distance-adaptive baseline. This is the "distance
    establishes an allowable resolution envelope" half of the design.
  - Within a band, a quadtree-style top-down split decision (standard
    adaptive-mesh technique, not novel) may refine a region to base/2 or
    base/refinement_factor if the INFORMATION SCORE (semantic + geometry +
    dynamic-relevance; see map_structure.compute_information_score) of any
    point cluster inside it clears a configured threshold. This is the
    "scene information can influence where detail is valuable" half.
  - A region can never end up coarser than its band's base size, and never
    finer than base/refinement_factor -- the envelope is a hard constraint,
    information score only decides finer-or-not *within* it.

Because the split decision is local (per coarse/medium group), two regions at
the same distance band can end up at different resolution levels if their
information scores differ -- this is the key behavior distinguishing FOVEA
from the pure distance-adaptive baseline, and is checked explicitly in
tests/test_core.py.
"""
from __future__ import annotations

import time

import numpy as np

from src.geometry import assign_bands, cell_index, radial_range
from src.map_structure import aggregate_points_to_cells

_OFFSET = 1 << 20          # covers +-1,048,576 grid cells, far beyond any SemanticPOSS frame extent
_SHIFT = 1 << 21


def _encode(band: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Bijective integer key for (band, ix, iy) used for fast group-by / lookup
    via sort+searchsorted instead of Python-level dict lookups per point."""
    band = band.astype(np.int64)
    ix = ix.astype(np.int64) + _OFFSET
    iy = iy.astype(np.int64) + _OFFSET
    return (band * _SHIFT + ix) * _SHIFT + iy


def build_fovea_map(points: np.ndarray, semantic_labels: np.ndarray, cfg: dict) -> tuple[list, dict]:
    t0 = time.perf_counter()
    n_total_input = points.shape[0]
    origin = (cfg["origin"]["x"], cfg["origin"]["y"])
    bands = cfg["distance_adaptive"]["bands"]
    fovea_cfg = cfg["fovea"]

    factor = int(fovea_cfg["refinement_factor"])
    half = factor // 2
    split_med_th = float(fovea_cfg["split_to_medium_threshold"])
    split_fine_th = float(fovea_cfg["split_to_fine_threshold"])
    priority_table = fovea_cfg["semantic_priority"]
    dynamic_classes = set(fovea_cfg["dynamic_classes"])
    geometry_max_std = fovea_cfg["geometry"]["max_expected_std_m"]
    weights = fovea_cfg["weights"]

    ranges = radial_range(points)
    band_id_all, base_cs_all = assign_bands(ranges, bands)
    in_env = band_id_all >= 0
    n_out_of_envelope = int((~in_env).sum())

    pts, labels = points[in_env], semantic_labels[in_env]
    band_id, base_cs = band_id_all[in_env], base_cs_all[in_env]
    x, y = pts[:, 0], pts[:, 1]

    if pts.shape[0] == 0:
        return [], {
            "input_points": n_total_input, "assigned_points": 0,
            "out_of_envelope_points": n_out_of_envelope, "invalid_points": n_total_input,
            "assignment_rate": 0.0, "output_cell_count": 0,
            "latency_s": time.perf_counter() - t0, "method": "fovea",
        }

    finest_cs = base_cs / factor
    ix_f = cell_index(x, origin[0], finest_cs)
    iy_f = cell_index(y, origin[1], finest_cs)

    # Step 1: aggregate at the finest permitted resolution to get an information
    # score for every micro-cell that actually contains points.
    finest_cells, finest_accounting = aggregate_points_to_cells(
        pts, labels, ix_f, iy_f, finest_cs, np.full(x.shape, 2, dtype=np.int64), band_id,
        origin, priority_table, dynamic_classes, geometry_max_std, weights,
    )
    n_invalid = pts.shape[0] - finest_accounting["valid_points"]

    if not finest_cells:
        return [], {
            "input_points": n_total_input, "assigned_points": 0,
            "out_of_envelope_points": n_out_of_envelope, "invalid_points": n_invalid,
            "assignment_rate": 0.0, "output_cell_count": 0,
            "latency_s": time.perf_counter() - t0, "method": "fovea",
        }

    fc_band = np.array([c.band_id for c in finest_cells], dtype=np.int64)
    fc_ix = np.array([c.ix for c in finest_cells], dtype=np.int64)
    fc_iy = np.array([c.iy for c in finest_cells], dtype=np.int64)
    fc_score = np.array([c.information_score for c in finest_cells], dtype=np.float64)
    fc_code = _encode(fc_band, fc_ix, fc_iy)

    # Step 2: coarse-group (band, ix//factor, iy//factor) decision -- split to medium?
    fc_cix, fc_ciy = fc_ix // factor, fc_iy // factor
    coarse_code = _encode(fc_band, fc_cix, fc_ciy)
    uniq_coarse, inv_coarse = np.unique(coarse_code, return_inverse=True)
    coarse_max = np.zeros(uniq_coarse.shape[0])
    np.maximum.at(coarse_max, inv_coarse, fc_score)
    split_to_medium = coarse_max >= split_med_th

    # Step 3: medium-group (band, ix//half, iy//half) decision -- split further to fine?
    fc_mix, fc_miy = fc_ix // half, fc_iy // half
    medium_code = _encode(fc_band, fc_mix, fc_miy)
    uniq_medium, inv_medium = np.unique(medium_code, return_inverse=True)
    medium_max = np.zeros(uniq_medium.shape[0])
    np.maximum.at(medium_max, inv_medium, fc_score)
    split_to_fine = medium_max >= split_fine_th

    coarse_split_per_fc = split_to_medium[inv_coarse]
    medium_split_per_fc = split_to_fine[inv_medium]
    is_medium = coarse_split_per_fc & (~medium_split_per_fc)
    is_fine = coarse_split_per_fc & medium_split_per_fc

    fc_base_cs = np.array([bands[int(b)]["cell_size_m"] for b in fc_band], dtype=np.float64)

    final_res = np.zeros(fc_ix.shape[0], dtype=np.int64)
    final_cs = fc_base_cs.copy()
    final_ix, final_iy = fc_cix.copy(), fc_ciy.copy()

    final_res[is_medium] = 1
    final_cs[is_medium] = fc_base_cs[is_medium] / half
    final_ix[is_medium], final_iy[is_medium] = fc_mix[is_medium], fc_miy[is_medium]

    final_res[is_fine] = 2
    final_cs[is_fine] = fc_base_cs[is_fine] / factor
    final_ix[is_fine], final_iy[is_fine] = fc_ix[is_fine], fc_iy[is_fine]

    # Step 4: broadcast each finest micro-cell's decision back to its raw points.
    point_code = _encode(band_id, ix_f, iy_f)
    order = np.argsort(fc_code)
    sorted_codes = fc_code[order]
    pos = np.clip(np.searchsorted(sorted_codes, point_code), 0, len(sorted_codes) - 1)
    matched = sorted_codes[pos] == point_code
    fc_idx_for_point = order[pos]

    point_final_cs = final_cs[fc_idx_for_point]
    point_final_res = final_res[fc_idx_for_point]
    point_final_ix = final_ix[fc_idx_for_point]
    point_final_iy = final_iy[fc_idx_for_point]

    cells, agg_accounting = aggregate_points_to_cells(
        pts[matched], labels[matched],
        point_final_ix[matched], point_final_iy[matched], point_final_cs[matched],
        point_final_res[matched], band_id[matched],
        origin, priority_table, dynamic_classes, geometry_max_std, weights,
    )

    n_unmatched = int((~matched).sum())
    accounting = {
        "input_points": n_total_input,
        "out_of_envelope_points": n_out_of_envelope,
        "invalid_points": n_invalid + n_unmatched,
        "assigned_points": agg_accounting["assigned_points"],
        "assignment_rate": agg_accounting["assigned_points"] / n_total_input if n_total_input else 0.0,
        "output_cell_count": agg_accounting["output_cell_count"],
        "latency_s": time.perf_counter() - t0,
        "method": "fovea",
        "bands": bands,
        "refinement_factor": factor,
        "split_to_medium_threshold": split_med_th,
        "split_to_fine_threshold": split_fine_th,
        "cells_by_resolution_level": {
            int(lvl): int((np.array([c.resolution_level for c in cells]) == lvl).sum())
            for lvl in sorted(set(c.resolution_level for c in cells))
        } if cells else {},
    }
    return cells, accounting


def explain_allocation(cell) -> str:
    """Human-readable explanation of a cell's resolution decision, generated
    from the actual stored factors (never a hardcoded/canned string)."""
    level_name = {0: "coarse (band base size)", 1: "medium (band base / 2)", 2: "fine (band base / refinement_factor)"}
    reasons = []
    if cell.semantic_priority >= 0.6:
        reasons.append(f"high semantic priority for '{cell.majority_class_name}' ({cell.semantic_priority:.2f})")
    elif cell.semantic_priority <= 0.15:
        reasons.append(f"low semantic priority for '{cell.majority_class_name}' ({cell.semantic_priority:.2f})")
    if cell.geometry_score >= 0.5:
        reasons.append(f"high local geometric complexity ({cell.geometry_score:.2f})")
    if cell.dynamic_score >= 1.0:
        reasons.append("classified as a dynamic-relevant object category")

    reason_text = "; ".join(reasons) if reasons else "no single factor dominated"
    return (
        f"Resolution level = {level_name.get(cell.resolution_level, cell.resolution_level)} "
        f"(cell size {cell.cell_size_m:.4f} m) at range {cell.range_m:.1f} m. "
        f"Information score = {cell.information_score:.2f} ({reason_text}). "
        f"This is within the distance envelope for this range -- FOVEA refined/kept this cell "
        f"based on information score without exceeding the band's coarsest/finest limits."
    )
