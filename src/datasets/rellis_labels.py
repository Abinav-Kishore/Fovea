"""RELLIS-3D label mapping.

IMPORTANT: this is RELLIS-3D's OWN ontology, copied verbatim from the
project's official benchmark config -- it is NOT SemanticPOSS's LABEL_DICT
(src/semantic_labels.py) and must never be reused for RELLIS data. Kept as a
separate module/table on purpose, per the same "load explicitly, never
guess" convention as semantic_labels.py.

Source (fetched and verified against this repo's actual RELLIS-3D .label
files, see below):
  https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/main/benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml
  -> the `labels:` section (raw per-point class id -> name; NOT the reduced
     `learning_map` training taxonomy, which collapses several of these to
     "void" -- FOVEA scores the full raw ontology).

Verified against real data in this repo: for
Rellis-3D 2/00000/vel_cloud_node_semantickitti_label_id/*.label, `label &
0xFFFF` (see rellis.py) produced ids {0,3,4,8,9,17,18,19,23,27,31,33} across
5 sampled frames (one per sequence) -- every one of those ids is present in
LABEL_DICT below, confirming this is the correct raw id space for these
files (as opposed to the reduced learning_map space).
"""
from __future__ import annotations

LABEL_DICT: dict[int, str] = {
    0: "void",
    1: "dirt",
    3: "grass",
    4: "tree",
    5: "pole",
    6: "water",
    7: "sky",
    8: "vehicle",
    9: "object",
    10: "asphalt",
    12: "building",
    15: "log",
    17: "person",
    18: "fence",
    19: "bush",
    23: "concrete",
    27: "barrier",
    31: "puddle",
    33: "mud",
    34: "rubble",
}

UNKNOWN_CLASS_NAME = "unknown"


def class_name(class_id: int) -> str:
    """Map a raw RELLIS-3D class id to its official name.

    Any id not present in LABEL_DICT is reported as "unknown" rather than
    silently assumed -- mirrors semantic_labels.class_name's contract.
    """
    return LABEL_DICT.get(int(class_id), UNKNOWN_CLASS_NAME)


def priority_lookup(class_id: int, priority_table: dict[str, float], default: float = 0.1) -> float:
    """Semantic priority for a class id, using the configured RELLIS
    name->priority table (config.yaml -> datasets.RELLIS-3D.semantic_priority).
    `default` applies to any class name present in the ontology but absent
    from the configured table, and to "unknown" ids."""
    name = class_name(class_id)
    return float(priority_table.get(name, default))
