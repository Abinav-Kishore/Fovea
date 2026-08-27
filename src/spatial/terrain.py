"""Terrain / traversability classification.

Maps a cell's semantic class NAME (in whichever ontology produced it --
SemanticPOSS or RELLIS-3D ground truth, or the SemanticKITTI-trained
perception model's predictions -- src/perception/) to one of four
traversability categories:

    DRIVABLE      -- a surface a ground vehicle can drive on
    NON_DRIVABLE  -- passable-looking terrain that isn't meant to be
                     driven on (sidewalks, loose vegetation, mud/water)
    OBSTACLE      -- a solid object that must be avoided
    UNKNOWN       -- the class doesn't give enough information to decide,
                     or belongs to a source with no registered table

This is a transparent, hand-authored PROTOTYPE HEURISTIC (same status as
the semantic-priority tables in config.yaml) -- not a learned or validated
traversability classifier. It deliberately uses each source's own
UN-collapsed class names rather than the coarser canonical ontology from
src/perception/ontology.py: canonical class merging loses distinctions
that matter for terrain (e.g. canonical "vegetation" merges RELLIS's
"grass" -- often traversable off-road -- with "tree"/"bush", which are
not; canonical "ground" merges SemanticKITTI's "sidewalk", which isn't
meant for vehicle traffic, with "road", which is).

Any class name not explicitly listed below maps to UNKNOWN, never guessed
-- e.g. SemanticKITTI's "other-ground" genuinely could be either, so it is
left UNKNOWN rather than assigned a coin-flip default.
"""
from __future__ import annotations

from enum import Enum


class TerrainClass(str, Enum):
    DRIVABLE = "drivable"
    NON_DRIVABLE = "non_drivable"
    OBSTACLE = "obstacle"
    UNKNOWN = "unknown"


# --- SemanticPOSS (ground truth) --------------------------------------------
# Names exactly as in src/semantic_labels.py -> LABEL_DICT.
_SEMANTICPOSS_TERRAIN: dict[str, TerrainClass] = {
    "unlabeled": TerrainClass.UNKNOWN,
    "1 person": TerrainClass.OBSTACLE,
    "2+ person": TerrainClass.OBSTACLE,
    "rider": TerrainClass.OBSTACLE,
    "car": TerrainClass.OBSTACLE,
    "trunk": TerrainClass.OBSTACLE,
    "plants": TerrainClass.NON_DRIVABLE,
    "traffic sign 1": TerrainClass.OBSTACLE,
    "traffic sign 2": TerrainClass.OBSTACLE,
    "traffic sign 3": TerrainClass.OBSTACLE,
    "pole": TerrainClass.OBSTACLE,
    "trashcan": TerrainClass.OBSTACLE,
    "building": TerrainClass.OBSTACLE,
    "cone/stone": TerrainClass.OBSTACLE,
    "fence": TerrainClass.OBSTACLE,
    "bike": TerrainClass.OBSTACLE,
    "ground": TerrainClass.DRIVABLE,
}

# --- RELLIS-3D (ground truth) -----------------------------------------------
# Names exactly as in src/datasets/rellis_labels.py -> LABEL_DICT. Off-road
# context: "grass"/"dirt" ARE reasonably traversable for this domain, unlike
# an on-road dataset where anything but paved surface would be NON_DRIVABLE.
_RELLIS_TERRAIN: dict[str, TerrainClass] = {
    "void": TerrainClass.UNKNOWN,
    "dirt": TerrainClass.DRIVABLE,
    "grass": TerrainClass.DRIVABLE,
    "tree": TerrainClass.OBSTACLE,
    "pole": TerrainClass.OBSTACLE,
    "water": TerrainClass.NON_DRIVABLE,
    "sky": TerrainClass.UNKNOWN,
    "vehicle": TerrainClass.OBSTACLE,
    "object": TerrainClass.OBSTACLE,
    "asphalt": TerrainClass.DRIVABLE,
    "building": TerrainClass.OBSTACLE,
    "log": TerrainClass.OBSTACLE,
    "person": TerrainClass.OBSTACLE,
    "fence": TerrainClass.OBSTACLE,
    "bush": TerrainClass.NON_DRIVABLE,
    "concrete": TerrainClass.DRIVABLE,
    "barrier": TerrainClass.OBSTACLE,
    "puddle": TerrainClass.NON_DRIVABLE,
    "mud": TerrainClass.NON_DRIVABLE,
    "rubble": TerrainClass.OBSTACLE,
}

# --- SemanticKITTI-20 (Phase 1 perception model's prediction ontology) -----
# Names exactly as in models/squeezeseg/data_cfg.yaml -> learning_map_inv.
_SEMANTICKITTI_TERRAIN: dict[str, TerrainClass] = {
    "unlabeled": TerrainClass.UNKNOWN,
    "car": TerrainClass.OBSTACLE,
    "bicycle": TerrainClass.OBSTACLE,
    "motorcycle": TerrainClass.OBSTACLE,
    "truck": TerrainClass.OBSTACLE,
    "other-vehicle": TerrainClass.OBSTACLE,
    "person": TerrainClass.OBSTACLE,
    "bicyclist": TerrainClass.OBSTACLE,
    "motorcyclist": TerrainClass.OBSTACLE,
    "road": TerrainClass.DRIVABLE,
    "parking": TerrainClass.DRIVABLE,
    "sidewalk": TerrainClass.NON_DRIVABLE,
    "other-ground": TerrainClass.UNKNOWN,  # genuinely ambiguous catch-all in SemanticKITTI itself
    "building": TerrainClass.OBSTACLE,
    "fence": TerrainClass.OBSTACLE,
    "vegetation": TerrainClass.NON_DRIVABLE,
    "trunk": TerrainClass.OBSTACLE,
    "terrain": TerrainClass.NON_DRIVABLE,
    "pole": TerrainClass.OBSTACLE,
    "traffic-sign": TerrainClass.OBSTACLE,
}

_TERRAIN_TABLES: dict[str, dict[str, TerrainClass]] = {
    "SemanticPOSS": _SEMANTICPOSS_TERRAIN,
    "RELLIS-3D": _RELLIS_TERRAIN,
    "semantickitti-20": _SEMANTICKITTI_TERRAIN,
}


def classify_terrain(class_name: str, source: str) -> TerrainClass:
    """`source` is a dataset name ("SemanticPOSS"/"RELLIS-3D") for
    ground-truth labels, or a model ontology ("semantickitti-20") for
    predictions -- the same source-key convention src/perception/ontology.py
    uses. An unregistered source or class name maps to UNKNOWN, never a
    guessed default."""
    table = _TERRAIN_TABLES.get(source)
    if table is None:
        return TerrainClass.UNKNOWN
    return table.get(class_name, TerrainClass.UNKNOWN)


def available_terrain_sources() -> list[str]:
    return list(_TERRAIN_TABLES.keys())
