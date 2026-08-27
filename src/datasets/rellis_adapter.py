"""RELLIS-3D dataset adapter -- conforms src/datasets/rellis.py's loader
functions and src/datasets/rellis_labels.py's ontology to DatasetAdapter,
the same interface SemanticPOSSAdapter implements.
"""
from __future__ import annotations

from src.data_loader import Frame
from src.datasets import rellis
from src.datasets.rellis_labels import LABEL_DICT, class_name


class RellisAdapter:
    name = "RELLIS-3D"

    def __init__(self, cfg: dict):
        self._rcfg = cfg["datasets"]["RELLIS-3D"]

    def list_sequences(self) -> list[str]:
        return rellis.list_sequences(self._rcfg)

    def list_frames(self, sequence: str) -> list[str]:
        return rellis.list_frame_ids(self._rcfg, sequence)

    def load_frame(self, sequence: str, frame_id: str) -> Frame:
        return rellis.load_frame(self._rcfg, sequence, frame_id)

    def frame_diagnostics(self, frame: Frame) -> dict:
        return rellis.frame_diagnostics(frame)

    def label_dict(self) -> dict[int, str]:
        return LABEL_DICT

    def class_name(self, class_id: int) -> str:
        return class_name(class_id)

    def semantic_priority_table(self) -> dict[str, float]:
        return self._rcfg["semantic_priority"]

    def dynamic_classes(self) -> set[str]:
        return set(self._rcfg["dynamic_classes"])
