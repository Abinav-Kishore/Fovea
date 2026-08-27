"""FOVEA perception layer: real neural-network semantic segmentation,
producing predicted labels FOVEA consumes exactly like it consumes ground
truth. Ground truth is used only by evaluation.py, never as a prediction.
"""
from src.perception.base import PerceptionResult, RawPrediction, SemanticSegmentationModel
from src.perception.evaluation import EvaluationResult, evaluate_predictions
from src.perception.inference import run_perception
from src.perception.ontology import (
    CANONICAL_CLASSES,
    canonical_id_for_name,
    map_names_to_canonical,
    model_predictable_canonical_classes,
)

__all__ = [
    "PerceptionResult",
    "RawPrediction",
    "SemanticSegmentationModel",
    "EvaluationResult",
    "evaluate_predictions",
    "run_perception",
    "CANONICAL_CLASSES",
    "canonical_id_for_name",
    "map_names_to_canonical",
    "model_predictable_canonical_classes",
]
