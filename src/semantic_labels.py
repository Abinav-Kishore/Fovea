"""SemanticPOSS label mapping.

IMPORTANT: this LABEL_DICT is copied verbatim from the dataset's own
`SemanticPOSS_dataset/dataset/read_data.py` (the official reader shipped with
the dataset). We do NOT assume SemanticKITTI's class IDs — SemanticPOSS uses
its own sparse set of IDs (0,4-17,21,22) with gaps, which is exactly why this
mapping is loaded explicitly rather than guessed.
"""
from __future__ import annotations

# Source: SemanticPOSS_dataset/dataset/read_data.py -> LABEL_DICT
LABEL_DICT: dict[int, str] = {
    0: "unlabeled",
    4: "1 person",
    5: "2+ person",
    6: "rider",
    7: "car",
    8: "trunk",
    9: "plants",
    10: "traffic sign 1",
    11: "traffic sign 2",
    12: "traffic sign 3",
    13: "pole",
    14: "trashcan",
    15: "building",
    16: "cone/stone",
    17: "fence",
    21: "bike",
    22: "ground",
}

UNKNOWN_CLASS_NAME = "unknown"


def class_name(class_id: int) -> str:
    """Map a raw SemanticPOSS class id to its documented name.

    Any id not present in the dataset's own LABEL_DICT is reported as
    "unknown" rather than silently assumed -- this can happen if a future
    perception model predicts an id outside the dataset's label set.
    """
    return LABEL_DICT.get(int(class_id), UNKNOWN_CLASS_NAME)


def priority_lookup(class_id: int, priority_table: dict[str, float], default: float = 0.1) -> float:
    """Semantic priority for a class id, using the configured name->priority table.

    `default` applies to any class name present in the dataset but absent from
    the user's config (e.g. a newly-observed id), and to "unknown" ids.
    """
    name = class_name(class_id)
    return float(priority_table.get(name, default))
