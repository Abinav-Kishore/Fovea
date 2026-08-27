"""Perception layer abstraction.

FOVEA must not know which neural network produced its semantic labels --
it only ever consumes the same (points, semantic_labels) shape it already
consumes from ground truth. This module defines that boundary explicitly:

  Frame (raw points + OPTIONAL ground_truth semantic_labels, from a
         DatasetAdapter -- src/data_loader.py, src/datasets/)
      |
      v
  SemanticSegmentationModel.predict(points) -> RawPrediction
      |  (model's OWN class-id space -- never assumed to match any dataset's)
      v
  PerceptionResult (predicted labels + canonical-mapped labels + timing)
      |
      v
  FOVEA engine (uniform_baseline / distance_baseline / fovea_allocator) --
      called with predicted_labels, exactly like it's called with ground
      truth today. FOVEA's own code does not change.

Ground truth (frame.semantic_labels, when has_labels=True) is NEVER passed
into a model's predict() and never returned as a "prediction" -- it is only
ever used by src/perception/evaluation.py, strictly for scoring predictions
against it after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class RawPrediction:
    """A model's raw output, in the MODEL's OWN class-id space -- not
    canonical, not any dataset's ground-truth ontology. Mapping to a
    canonical space happens later (src/perception/ontology.py), explicitly,
    never silently assumed."""

    predicted_label_ids: np.ndarray  # (N,) int, aligned 1:1 with the input points
    confidence: Optional[np.ndarray]  # (N,) float in [0,1], or None -- never fabricated
    model_ontology: str  # e.g. "semantickitti-20" -- identifies which id->name table applies


class SemanticSegmentationModel(Protocol):
    """Interface every perception model implements. FOVEA and the API layer
    depend only on this -- never on a specific architecture."""

    name: str
    model_ontology: str
    provides_confidence: bool

    def preprocess(self, points: np.ndarray) -> object:
        """Model-specific preprocessing (e.g. range-image projection).
        Returns an opaque intermediate object passed to forward()."""
        ...

    def forward(self, preprocessed: object) -> object:
        """Runs the actual network. Returns an opaque model-output object
        passed to postprocess(). This is the stage that should be timed as
        pure inference (no I/O, no projection math)."""
        ...

    def postprocess(self, points: np.ndarray, preprocessed: object, model_output: object) -> RawPrediction:
        """Converts model output back to a RawPrediction aligned 1:1 with
        the ORIGINAL input points -- len(predicted_label_ids) == len(points)
        is a hard invariant, enforced here or the caller raises loudly."""
        ...

    def predict(self, points: np.ndarray) -> RawPrediction:
        """Convenience: preprocess -> forward -> postprocess in one call,
        for callers that don't need the timing breakdown."""
        ...


@dataclass
class PerceptionResult:
    """Everything the API/CLI/FOVEA integration needs from one inference
    run on one frame. `predicted_labels` are the model's raw class ids
    (its own ontology); `predicted_canonical` are those same predictions
    mapped through src/perception/ontology.py into FOVEA's shared canonical
    class space, used for cross-dataset-consistent FOVEA scoring/evaluation."""

    model_name: str
    model_ontology: str
    predicted_labels: np.ndarray  # (N,) int, model's own ontology, aligned to frame.points
    predicted_canonical: np.ndarray  # (N,) int, canonical ontology (src/perception/ontology.py)
    confidence: Optional[np.ndarray]  # (N,) float or None -- never fabricated
    num_points: int
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    total_perception_ms: float
    points_per_second: float
