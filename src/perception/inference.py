"""Orchestrates one perception run: preprocess -> forward -> postprocess,
timed as three SEPARATE stages (never mixed with FOVEA allocation time,
which happens later and separately -- see src/api.py's inference-mode
endpoints), plus the canonical-ontology mapping of the raw prediction.
"""
from __future__ import annotations

import time

import numpy as np

from src.perception.base import PerceptionResult, SemanticSegmentationModel
from src.perception.ontology import map_names_to_canonical


def run_perception(model: SemanticSegmentationModel, points: np.ndarray) -> PerceptionResult:
    n = points.shape[0]

    t0 = time.perf_counter()
    preprocessed = model.preprocess(points)
    t1 = time.perf_counter()
    model_output = model.forward(preprocessed)
    t2 = time.perf_counter()
    raw = model.postprocess(points, preprocessed, model_output)
    t3 = time.perf_counter()

    if len(raw.predicted_label_ids) != n:
        raise RuntimeError(
            f"{model.name}: predicted {len(raw.predicted_label_ids)} labels for {n} input points -- "
            f"the len(predicted_labels) == len(points) invariant was violated. Refusing to return this result."
        )

    # Map raw model class ids -> names -> canonical. Requires the model to
    # expose a raw_label_names id->name table (SqueezeSegModel does, from
    # its bundled data_cfg.yaml) -- fail loudly if a model doesn't, rather
    # than silently skipping canonical mapping.
    raw_label_names = getattr(model, "raw_label_names", None)
    if raw_label_names is None:
        raise AttributeError(
            f"{model.name} does not expose raw_label_names (id->name); cannot map predictions "
            f"to the canonical ontology (src/perception/ontology.py)."
        )
    names = [raw_label_names.get(int(i), "unlabeled") for i in raw.predicted_label_ids]
    predicted_canonical = np.asarray(map_names_to_canonical(names, raw.model_ontology), dtype=np.int64)

    preprocessing_ms = (t1 - t0) * 1000.0
    inference_ms = (t2 - t1) * 1000.0
    postprocessing_ms = (t3 - t2) * 1000.0
    total_ms = (t3 - t0) * 1000.0
    points_per_second = n / (total_ms / 1000.0) if total_ms > 0 else 0.0

    return PerceptionResult(
        model_name=model.name,
        model_ontology=raw.model_ontology,
        predicted_labels=raw.predicted_label_ids,
        predicted_canonical=predicted_canonical,
        confidence=raw.confidence,
        num_points=n,
        preprocessing_ms=preprocessing_ms,
        inference_ms=inference_ms,
        postprocessing_ms=postprocessing_ms,
        total_perception_ms=total_ms,
        points_per_second=points_per_second,
    )
