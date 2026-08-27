"""Canonical semantic ontology + explicit mapping layer.

SqueezeSeg (SemanticKITTI, 20-class reduced ontology), SemanticPOSS, and
RELLIS-3D each define their own, INCOMPATIBLE class-id spaces -- none are
interchangeable, and this module never assumes otherwise. It defines one
small canonical class set that all three map INTO (by class NAME, since
every source already resolves raw ids to names -- src.semantic_labels,
src.datasets.rellis_labels, and this model's own raw_label_names), so
predictions (in SqueezeSeg's space) and ground truth (in SemanticPOSS's or
RELLIS's space) can be compared on equal footing in evaluation.py.

    model class (SemanticKITTI)  -\
    SemanticPOSS class            |-> canonical FOVEA class <- RELLIS-3D class
                                  -/

Any class name not explicitly listed below maps to CANONICAL_UNKNOWN and is
excluded from mIoU (same "ignore" convention SemanticKITTI itself uses for
its own "unlabeled"/"outlier" classes) -- never silently folded into an
arbitrary bucket.

IMPORTANT, stated explicitly: this mapping is a coarse, hand-authored
correspondence for cross-ontology evaluation -- a prototype heuristic, not
a validated semantic equivalence. Some canonical classes (HAZARD) exist
only in RELLIS's off-road ontology and have NO possible match in
SemanticKITTI's driving-scene vocabulary -- SqueezeSeg can structurally
never predict them. This is documented, not hidden (see
evaluation.py's `model_coverage` metric and the README's RELLIS section).
"""
from __future__ import annotations

CANONICAL_UNKNOWN = 0
CANONICAL_CLASSES: dict[int, str] = {
    0: "unknown",
    1: "ground",
    2: "vegetation",
    3: "building",
    4: "fence",
    5: "pole",
    6: "sign",
    7: "person",
    8: "rider",
    9: "car",
    10: "bike",
    11: "other_vehicle",
    12: "object",
    13: "hazard",  # off-road only (mud/puddle/water/log/rubble/barrier) -- no SemanticKITTI equivalent
}
CANONICAL_IGNORE_IN_MIOU = {CANONICAL_UNKNOWN}  # matches SemanticKITTI's own "unlabeled" ignore convention

# --- SqueezeSeg / SemanticKITTI (20-class reduced training ontology) -------
# Names exactly as in models/squeezeseg/data_cfg.yaml -> learning_map_inv names.
SEMANTICKITTI_TO_CANONICAL: dict[str, str] = {
    "unlabeled": "unknown",
    "car": "car",
    "bicycle": "bike",
    "motorcycle": "bike",
    "truck": "other_vehicle",
    "other-vehicle": "other_vehicle",
    "person": "person",
    "bicyclist": "rider",
    "motorcyclist": "rider",
    "road": "ground",
    "parking": "ground",
    "sidewalk": "ground",
    "other-ground": "ground",
    "building": "building",
    "fence": "fence",
    "vegetation": "vegetation",
    "trunk": "vegetation",
    "terrain": "ground",
    "pole": "pole",
    "traffic-sign": "sign",
}

# --- SemanticPOSS -----------------------------------------------------------
# Names exactly as in src/semantic_labels.py -> LABEL_DICT.
SEMANTICPOSS_TO_CANONICAL: dict[str, str] = {
    "unlabeled": "unknown",
    "1 person": "person",
    "2+ person": "person",
    "rider": "rider",
    "car": "car",
    "trunk": "vegetation",
    "plants": "vegetation",
    "traffic sign 1": "sign",
    "traffic sign 2": "sign",
    "traffic sign 3": "sign",
    "pole": "pole",
    "trashcan": "object",
    "building": "building",
    "cone/stone": "object",
    "fence": "fence",
    "bike": "bike",
    "ground": "ground",
}

# --- RELLIS-3D ---------------------------------------------------------------
# Names exactly as in src/datasets/rellis_labels.py -> LABEL_DICT.
# Classes mapped to "hazard" have NO SemanticKITTI equivalent -- SqueezeSeg's
# 20-class output can never land on canonical class 13, so these ground-truth
# points can only ever be misses when scoring SqueezeSeg's predictions
# against RELLIS. See evaluation.py's model_coverage metric.
RELLIS_TO_CANONICAL: dict[str, str] = {
    "void": "unknown",
    "dirt": "ground",
    "grass": "vegetation",
    "tree": "vegetation",
    "pole": "pole",
    "water": "hazard",
    "sky": "unknown",
    "vehicle": "car",
    "object": "object",
    "asphalt": "ground",
    "building": "building",
    "log": "hazard",
    "person": "person",
    "fence": "fence",
    "bush": "vegetation",
    "concrete": "ground",
    "barrier": "hazard",
    "puddle": "hazard",
    "mud": "hazard",
    "rubble": "hazard",
}

_SOURCE_TABLES: dict[str, dict[str, str]] = {
    "semantickitti-20": SEMANTICKITTI_TO_CANONICAL,
    "SemanticPOSS": SEMANTICPOSS_TO_CANONICAL,
    "RELLIS-3D": RELLIS_TO_CANONICAL,
}

_CANONICAL_NAME_TO_ID = {name: cid for cid, name in CANONICAL_CLASSES.items()}


def canonical_id_for_name(class_name: str, source: str) -> int:
    """Maps one class name from `source`'s ontology ("semantickitti-20",
    "SemanticPOSS", or "RELLIS-3D") to a canonical class id. Unmapped names
    -> CANONICAL_UNKNOWN, explicitly, never a guessed nearest class."""
    table = _SOURCE_TABLES.get(source)
    if table is None:
        raise ValueError(f"No canonical mapping registered for source '{source}'. Known: {list(_SOURCE_TABLES)}")
    canonical_name = table.get(class_name, "unknown")
    return _CANONICAL_NAME_TO_ID[canonical_name]


def map_names_to_canonical(names, source: str):
    """Vectorized-friendly helper: `names` is any iterable of class-name
    strings (e.g. a numpy array of dtype=object/str) -> list[int] canonical ids."""
    table = _SOURCE_TABLES.get(source)
    if table is None:
        raise ValueError(f"No canonical mapping registered for source '{source}'. Known: {list(_SOURCE_TABLES)}")
    return [_CANONICAL_NAME_TO_ID[table.get(n, "unknown")] for n in names]


def model_predictable_canonical_classes(model_ontology: str) -> set[int]:
    """The set of canonical class ids `model_ontology` can structurally ever
    predict -- used to report evaluation coverage honestly (e.g. RELLIS's
    "hazard" classes are never in this set for a SemanticKITTI-trained model)."""
    table = _SOURCE_TABLES.get(model_ontology)
    if table is None:
        raise ValueError(f"Unknown model ontology '{model_ontology}'.")
    return {_CANONICAL_NAME_TO_ID[name] for name in table.values()}
