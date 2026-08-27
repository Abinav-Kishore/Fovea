"""Dataset abstraction: one small interface both SemanticPOSS and RELLIS-3D
implement, so the FOVEA engine and the API layer never need to know which
dataset is active beyond calling these methods.

Adding a third dataset later means writing one more adapter module + wiring
it into DATASET_ADAPTERS in __init__.py -- nothing in src/api.py, the
FOVEA engine, or the frontend needs a new branch for it.
"""
from __future__ import annotations

from typing import Protocol

from src.data_loader import Frame


class DatasetAdapter(Protocol):
    """Everything the API layer needs from a dataset, independent of its
    on-disk layout or ontology."""

    name: str

    def list_sequences(self) -> list[str]:
        """Sequence ids available for this dataset."""
        ...

    def list_frames(self, sequence: str) -> list[str]:
        """Frame ids available for `sequence`, restricted to frames that have
        BOTH a point cloud and a matching label file -- never a partial one."""
        ...

    def load_frame(self, sequence: str, frame_id: str) -> Frame:
        """Load one frame into the common Frame representation the FOVEA
        engine expects (src.data_loader.Frame: points, semantic_labels,
        instance_labels, ...) -- identical regardless of dataset."""
        ...

    def frame_diagnostics(self, frame: Frame) -> dict:
        """Same shape as src.data_loader.frame_diagnostics's output, with
        class names resolved via this dataset's own ontology."""
        ...

    def label_dict(self) -> dict[int, str]:
        """Raw class-id -> name table for this dataset's ontology."""
        ...

    def class_name(self, class_id: int) -> str:
        ...

    def semantic_priority_table(self) -> dict[str, float]:
        """Dataset-specific FOVEA semantic-priority heuristic table (see
        config.yaml). Never shared between datasets."""
        ...

    def dynamic_classes(self) -> set[str]:
        """Dataset-specific dynamic-relevance semantic-category proxy set
        (NOT real motion/temporal tracking -- see README)."""
        ...
