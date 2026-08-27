"""Explicit 2.5D perception representation layer -- wraps the existing,
unmodified FOVEA engine output (Cell + accounting) with elevation bounds,
terrain/traversability, and obstacle classification. See map_2_5d.py.
"""
from src.spatial.map_2_5d import (
    Map2_5DStats,
    PerceptionCell,
    build_2_5d_map,
    validate_cell_geometry,
    validate_no_duplicate_cells,
    validate_point_conservation,
)
from src.spatial.obstacle import ObstacleClass, classify_obstacle
from src.spatial.terrain import TerrainClass, available_terrain_sources, classify_terrain

__all__ = [
    "PerceptionCell",
    "Map2_5DStats",
    "build_2_5d_map",
    "validate_point_conservation",
    "validate_no_duplicate_cells",
    "validate_cell_geometry",
    "TerrainClass",
    "classify_terrain",
    "available_terrain_sources",
    "ObstacleClass",
    "classify_obstacle",
]
