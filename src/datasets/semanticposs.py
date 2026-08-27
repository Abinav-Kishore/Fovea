"""SemanticPOSS dataset adapter -- a thin wrapper around the existing,
UNCHANGED src/data_loader.py + src/semantic_labels.py. Every method here
delegates directly to the original functions; this file adds no new
behavior, it only conforms SemanticPOSS to the DatasetAdapter shape so the
API layer can treat both datasets uniformly.
"""
from __future__ import annotations

from pathlib import Path

from src.data_loader import Frame, frame_diagnostics, list_frame_ids, load_frame
from src.semantic_labels import LABEL_DICT, class_name


class SemanticPOSSAdapter:
    name = "SemanticPOSS"

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._root = cfg["data"]["root"]

    def list_sequences(self) -> list[str]:
        seq_dir = Path(self._root) / "sequences"
        if not seq_dir.exists():
            return []
        return sorted(p.name for p in seq_dir.glob("*") if p.is_dir())

    def list_frames(self, sequence: str) -> list[str]:
        return list_frame_ids(self._root, sequence)

    def load_frame(self, sequence: str, frame_id: str) -> Frame:
        return load_frame(self._root, sequence, frame_id)

    def frame_diagnostics(self, frame: Frame) -> dict:
        return frame_diagnostics(frame)

    def label_dict(self) -> dict[int, str]:
        return LABEL_DICT

    def class_name(self, class_id: int) -> str:
        return class_name(class_id)

    def semantic_priority_table(self) -> dict[str, float]:
        # Unchanged location -- config.yaml's existing fovea.semantic_priority,
        # exactly as v0.1 defined it. Never touched by the RELLIS integration.
        return self._cfg["fovea"]["semantic_priority"]

    def dynamic_classes(self) -> set[str]:
        return set(self._cfg["fovea"]["dynamic_classes"])
