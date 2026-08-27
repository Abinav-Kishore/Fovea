"""Obstacle classification.

Builds on terrain.py's OBSTACLE category, adding the static/potentially-
dynamic distinction the SIH representation needs -- using FOVEA's EXISTING
`dynamic_score` field (src/map_structure.py, computed by the unmodified
FOVEA engine) rather than any new computation.

IMPORTANT, stated plainly: `dynamic_score` is a STATIC semantic-category
proxy (documented since Phase 0 -- "is this class's name in the configured
dynamic_classes list") with NO temporal/motion component. A cell flagged
POTENTIAL_DYNAMIC_OBSTACLE here means "this object's semantic category is
one that COULD move (car, person, ...)", never "this object WAS OBSERVED
moving". Real temporal motion tracking does not exist yet and is not
claimed here -- see README's "2.5D Perception Representation" section.
"""
from __future__ import annotations

from enum import Enum

from src.spatial.terrain import TerrainClass


class ObstacleClass(str, Enum):
    NON_OBSTACLE = "non_obstacle"
    STATIC_OBSTACLE = "static_obstacle"
    POTENTIAL_DYNAMIC_OBSTACLE = "potential_dynamic_obstacle"  # semantic-category proxy ONLY -- see module docstring
    UNKNOWN = "unknown"


def classify_obstacle(terrain: TerrainClass, dynamic_score: float) -> ObstacleClass:
    """`dynamic_score` is read directly from an existing Cell -- never
    recomputed here."""
    if terrain == TerrainClass.UNKNOWN:
        return ObstacleClass.UNKNOWN
    if terrain != TerrainClass.OBSTACLE:
        return ObstacleClass.NON_OBSTACLE
    return ObstacleClass.POTENTIAL_DYNAMIC_OBSTACLE if dynamic_score >= 1.0 else ObstacleClass.STATIC_OBSTACLE
