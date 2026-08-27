"""Explicit 2.5D perception representation.

This module adds NO new spatial allocation or scoring logic -- it wraps the
EXISTING, unmodified FOVEA engine's output (a list of `src.map_structure.Cell`
+ its `accounting` dict, from build_uniform_map / build_distance_adaptive_map
/ build_fovea_map, run in either ground-truth or prediction mode) into an
explicit representation that makes elevation, terrain/traversability, and
obstacle status first-class fields, and validates the point-conservation
guarantees the engine already provides rather than re-deriving them.

    Cell (existing, from map_structure.py)
        |  center_x, center_y, cell_size_m, elevation_*, majority_class_name,
        |  semantic_priority, geometry_score, dynamic_score, information_score
        v
    PerceptionCell (this module)
        |  + explicit x/y bounds, terrain_class, obstacle_class, label_source
        v
    Map2_5DStats (this module) -- map-level terrain/elevation/coverage summary
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.map_structure import Cell
from src.spatial.obstacle import ObstacleClass, classify_obstacle
from src.spatial.terrain import TerrainClass, classify_terrain


@dataclass
class PerceptionCell:
    """One cell of the explicit 2.5D representation. Every field is either
    copied directly from an existing Cell, trivially derived from one
    (bounds from center +/- half cell size), or produced by the transparent
    terrain/obstacle rule tables above -- nothing here is a new spatial or
    scoring computation."""

    cell_id: int  # position in this map's cell list -- NOT persistent across frames/rebuilds

    # Spatial bounds -- derived, not stored redundantly by the engine.
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    center_x: float
    center_y: float
    cell_size_m: float
    resolution_level: int  # 0=coarse, 1=medium, 2=fine (FOVEA); always 0 for the two baselines
    band_id: int
    range_m: float

    # Elevation -- from the cell's own LiDAR points (never fabricated). Every
    # Cell FOVEA's aggregator produces has point_count >= 1 by construction
    # (see map_structure.py's docstring -- cells are only ever created for
    # occupied groups), so elevation_* below is always a real measurement,
    # never a placeholder for "empty". `has_points` is still an explicit,
    # checked field rather than an assumption -- see build_2_5d_map's
    # docstring for what "no points" actually means in this sparse
    # representation (there is no dense background grid to leave empty).
    has_points: bool
    point_count: int
    elevation_median: Optional[float]
    elevation_min: Optional[float]
    elevation_max: Optional[float]
    elevation_std: Optional[float]

    # Semantic (from Phase 1's label_source -- ground_truth or prediction)
    semantic_class: str
    semantic_priority: float
    geometry_score: float
    dynamic_score: float  # existing FOVEA field -- semantic-category proxy, NOT motion tracking
    information_score: float

    # Terrain / obstacle (this module)
    terrain_class: TerrainClass
    obstacle_class: ObstacleClass

    # Provenance -- which labels produced this cell, never silently mixed.
    label_source: str  # "ground_truth" | "prediction"
    source: str  # dataset name (ground_truth) or model ontology (prediction)


def _cell_to_perception_cell(cell_id: int, cell: Cell, label_source: str, source: str) -> PerceptionCell:
    half = cell.cell_size_m / 2.0
    has_points = cell.point_count > 0
    terrain = classify_terrain(cell.majority_class_name, source)
    obstacle = classify_obstacle(terrain, cell.dynamic_score)
    return PerceptionCell(
        cell_id=cell_id,
        x_min=cell.center_x - half,
        x_max=cell.center_x + half,
        y_min=cell.center_y - half,
        y_max=cell.center_y + half,
        center_x=cell.center_x,
        center_y=cell.center_y,
        cell_size_m=cell.cell_size_m,
        resolution_level=cell.resolution_level,
        band_id=cell.band_id,
        range_m=cell.range_m,
        has_points=has_points,
        point_count=cell.point_count,
        elevation_median=cell.elevation_median if has_points else None,
        elevation_min=cell.elevation_min if has_points else None,
        elevation_max=cell.elevation_max if has_points else None,
        elevation_std=cell.elevation_std if has_points else None,
        semantic_class=cell.majority_class_name,
        semantic_priority=cell.semantic_priority,
        geometry_score=cell.geometry_score,
        dynamic_score=cell.dynamic_score,
        information_score=cell.information_score,
        terrain_class=terrain,
        obstacle_class=obstacle,
        label_source=label_source,
        source=source,
    )


@dataclass
class Map2_5DStats:
    """Map-level statistics -- every number is a real aggregate over the
    actual PerceptionCells/accounting passed in, nothing fabricated."""

    total_cells: int
    occupied_cells: int  # == total_cells in this sparse representation -- see build_2_5d_map's docstring
    unknown_terrain_cells: int
    drivable_cells: int
    non_drivable_cells: int
    obstacle_cells: int
    static_obstacle_cells: int
    potential_dynamic_obstacle_cells: int

    mean_elevation_m: Optional[float]
    elevation_min_m: Optional[float]
    elevation_max_m: Optional[float]
    elevation_relief_std_m: Optional[float]  # std, ACROSS CELLS, of each cell's own elevation_median -- map-level relief, not within-cell spread

    input_points: int
    represented_points: int  # accounting["assigned_points"] -- points that landed in a returned cell
    out_of_envelope_points: int
    invalid_points: int
    representation_coverage: float  # represented_points / input_points

    label_source: str
    source: str


def summarize_2_5d_map(cells: list[PerceptionCell], accounting: dict, label_source: str, source: str) -> Map2_5DStats:
    total = len(cells)
    unknown = sum(1 for c in cells if c.terrain_class == TerrainClass.UNKNOWN)
    drivable = sum(1 for c in cells if c.terrain_class == TerrainClass.DRIVABLE)
    non_drivable = sum(1 for c in cells if c.terrain_class == TerrainClass.NON_DRIVABLE)
    obstacle = sum(1 for c in cells if c.terrain_class == TerrainClass.OBSTACLE)
    static_obstacle = sum(1 for c in cells if c.obstacle_class == ObstacleClass.STATIC_OBSTACLE)
    dynamic_obstacle = sum(1 for c in cells if c.obstacle_class == ObstacleClass.POTENTIAL_DYNAMIC_OBSTACLE)

    elevations = [c.elevation_median for c in cells if c.elevation_median is not None]
    if elevations:
        mean_elev = sum(elevations) / len(elevations)
        elev_min = min(c.elevation_min for c in cells if c.elevation_min is not None)
        elev_max = max(c.elevation_max for c in cells if c.elevation_max is not None)
        mean_sq = sum((e - mean_elev) ** 2 for e in elevations) / len(elevations)
        relief_std = mean_sq ** 0.5
    else:
        mean_elev = elev_min = elev_max = relief_std = None

    input_points = int(accounting.get("input_points", 0))
    represented = int(accounting.get("assigned_points", 0))
    out_of_envelope = int(accounting.get("out_of_envelope_points", 0))
    invalid = int(accounting.get("invalid_points", 0))
    coverage = represented / input_points if input_points else 0.0

    return Map2_5DStats(
        total_cells=total,
        occupied_cells=total,
        unknown_terrain_cells=unknown,
        drivable_cells=drivable,
        non_drivable_cells=non_drivable,
        obstacle_cells=obstacle,
        static_obstacle_cells=static_obstacle,
        potential_dynamic_obstacle_cells=dynamic_obstacle,
        mean_elevation_m=mean_elev,
        elevation_min_m=elev_min,
        elevation_max_m=elev_max,
        elevation_relief_std_m=relief_std,
        input_points=input_points,
        represented_points=represented,
        out_of_envelope_points=out_of_envelope,
        invalid_points=invalid,
        representation_coverage=coverage,
        label_source=label_source,
        source=source,
    )


def build_2_5d_map(
    cells: list[Cell], accounting: dict, label_source: str, source: str
) -> tuple[list[PerceptionCell], Map2_5DStats]:
    """`cells`/`accounting` come from an EXISTING build_uniform_map /
    build_distance_adaptive_map / build_fovea_map call, in either
    ground-truth or prediction mode -- this function performs no spatial
    allocation of its own.

    `label_source` is `"ground_truth"` or `"prediction"`. `source` is a
    dataset name (`"SemanticPOSS"`/`"RELLIS-3D"`, for ground_truth) or a
    model ontology (`"semantickitti-20"`, for prediction) -- the same
    source-key convention `src/perception/ontology.py` and
    `src/spatial/terrain.py` use.

    On "empty cells": FOVEA's cell list is SPARSE -- `aggregate_points_to_cells`
    only ever creates a Cell for a group that has at least one point (see
    map_structure.py's docstring), so there is no dense background grid with
    literal zero-point cells to represent here. "Unknown" in this
    representation therefore means SEMANTIC uncertainty (a cell whose
    majority class has no terrain rule, e.g. an unlabeled/void point, or a
    class this source's table doesn't cover) -- not geometric absence.
    Points that never landed in ANY cell (outside the distance envelope, or
    rejected as invalid/non-finite) are tracked and reported via
    `Map2_5DStats.out_of_envelope_points`/`invalid_points` -- exactly the
    counts the existing engine's `accounting` dict already provides, not a
    new computation. See `validate_point_conservation` below.
    """
    perception_cells = [_cell_to_perception_cell(i, c, label_source, source) for i, c in enumerate(cells)]
    stats = summarize_2_5d_map(perception_cells, accounting, label_source, source)
    return perception_cells, stats


# ---------------------------------------------------------------------------
# Objective, ground-truth-independent representation-correctness checks
# (Step 11 -- these validate EXISTING guarantees, they don't invent new ones).
# ---------------------------------------------------------------------------

def validate_point_conservation(accounting: dict) -> dict:
    """Confirms the existing engine's own accounting is internally
    consistent: every input point is exactly one of assigned / out-of-
    envelope / invalid. This is real validation of a real guarantee
    (map_structure.py's aggregator never drops a point silently), not a
    fabricated pass/fail."""
    input_points = int(accounting.get("input_points", 0))
    assigned = int(accounting.get("assigned_points", 0))
    out_of_envelope = int(accounting.get("out_of_envelope_points", 0))
    invalid = int(accounting.get("invalid_points", 0))
    accounted = assigned + out_of_envelope + invalid
    return {
        "input_points": input_points,
        "assigned_points": assigned,
        "out_of_envelope_points": out_of_envelope,
        "invalid_points": invalid,
        "accounted_points": accounted,
        "conserved": accounted == input_points,
    }


def validate_no_duplicate_cells(cells: list[Cell]) -> dict:
    """Confirms the (cell_size, ix, iy) key -- a pure function of physical
    location, per map_structure.py's own docstring -- never repeats."""
    keys = [(round(c.cell_size_m, 6), c.ix, c.iy) for c in cells]
    unique = set(keys)
    return {"cell_count": len(keys), "unique_cell_count": len(unique), "no_duplicates": len(keys) == len(unique)}


def validate_cell_geometry(cells: list[PerceptionCell]) -> dict:
    """Bounds, elevation, and resolution sanity checks -- cheap, objective,
    and independent of any ground truth."""
    bad_bounds = [c.cell_id for c in cells if not (c.x_min < c.x_max and c.y_min < c.y_max)]
    bad_elevation = [
        c.cell_id
        for c in cells
        if c.has_points and not (c.elevation_min <= c.elevation_median <= c.elevation_max)
    ]
    bad_resolution = [c.cell_id for c in cells if c.resolution_level not in (0, 1, 2) or c.cell_size_m <= 0]
    return {
        "cell_count": len(cells),
        "bad_bounds_cell_ids": bad_bounds,
        "bad_elevation_cell_ids": bad_elevation,
        "bad_resolution_cell_ids": bad_resolution,
        "valid": not bad_bounds and not bad_elevation and not bad_resolution,
    }
