"""Real evaluation: predictions vs. ground truth, both in the CANONICAL
class space (src/perception/ontology.py). Every number here is computed
from actual arrays passed in -- nothing is a placeholder or a guess.

Ground truth is used HERE ONLY. It is never passed to a model's predict()
and never appears in a PerceptionResult's predicted_* fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.perception.ontology import (
    CANONICAL_CLASSES,
    CANONICAL_IGNORE_IN_MIOU,
    model_predictable_canonical_classes,
)

DISTANCE_BANDS_M: list[tuple[float, float]] = [(0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 100.0)]


@dataclass
class DistanceBandMetrics:
    band: str
    point_count: int
    accuracy: float | None
    mean_iou: float | None


@dataclass
class EvaluationResult:
    num_points_total: int
    num_points_ignored: int  # ground truth was canonical "unknown" -- excluded from all metrics below
    num_points_evaluated: int
    overall_accuracy: float
    mean_iou: float
    per_class_iou: dict[str, float | None]
    confusion_matrix: list[list[int]]
    confusion_matrix_labels: list[str]
    distance_bands: dict[str, DistanceBandMetrics]
    model_coverage: float  # fraction of evaluated GT points whose canonical class the model can structurally predict
    excluded_canonical_classes: list[str] = field(default_factory=list)


def evaluate_predictions(
    predicted_canonical: np.ndarray,
    ground_truth_canonical: np.ndarray,
    range_m: np.ndarray,
    model_ontology: str,
) -> EvaluationResult:
    """`predicted_canonical`, `ground_truth_canonical`, `range_m` must all be
    the same length (one entry per point). Computes overall accuracy, mean
    IoU, per-class IoU, a confusion matrix, and the same metrics broken down
    by DISTANCE_BANDS_M -- all from these actual arrays, nothing fabricated.
    """
    n_total = len(ground_truth_canonical)
    if len(predicted_canonical) != n_total or len(range_m) != n_total:
        raise ValueError(
            f"Length mismatch: predictions={len(predicted_canonical)}, "
            f"ground_truth={n_total}, range={len(range_m)} -- refusing to evaluate misaligned arrays."
        )

    ignore_mask = np.isin(ground_truth_canonical, list(CANONICAL_IGNORE_IN_MIOU))
    valid = ~ignore_mask
    n_ignored = int(ignore_mask.sum())
    n_eval = int(valid.sum())

    pred_v = predicted_canonical[valid]
    gt_v = ground_truth_canonical[valid]
    range_v = range_m[valid]

    predictable = model_predictable_canonical_classes(model_ontology)
    coverage = float(np.isin(gt_v, list(predictable)).mean()) if n_eval > 0 else 0.0
    present_gt_classes = set(np.unique(gt_v).tolist()) - CANONICAL_IGNORE_IN_MIOU
    excluded = sorted(CANONICAL_CLASSES[c] for c in (present_gt_classes - predictable))

    overall_accuracy = float((pred_v == gt_v).mean()) if n_eval > 0 else 0.0

    class_ids = sorted(CANONICAL_CLASSES.keys())
    confusion = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    id_to_idx = {c: i for i, c in enumerate(class_ids)}
    if n_eval > 0:
        for g, p in zip(gt_v.tolist(), pred_v.tolist()):
            gi = id_to_idx.get(g)
            pi = id_to_idx.get(p)
            if gi is not None and pi is not None:
                confusion[gi, pi] += 1

    per_class_iou: dict[str, float | None] = {}
    ious_for_mean = []
    for cid in class_ids:
        name = CANONICAL_CLASSES[cid]
        if cid in CANONICAL_IGNORE_IN_MIOU:
            per_class_iou[name] = None
            continue
        tp = int(np.sum((pred_v == cid) & (gt_v == cid)))
        fp = int(np.sum((pred_v == cid) & (gt_v != cid)))
        fn = int(np.sum((pred_v != cid) & (gt_v == cid)))
        denom = tp + fp + fn
        if denom == 0:
            per_class_iou[name] = None  # class absent from both -- undefined, not fabricated as 0
        else:
            iou = tp / denom
            per_class_iou[name] = float(iou)
            ious_for_mean.append(iou)
    mean_iou = float(np.mean(ious_for_mean)) if ious_for_mean else 0.0

    distance_bands: dict[str, DistanceBandMetrics] = {}
    for lo, hi in DISTANCE_BANDS_M:
        band_key = f"{lo:g}-{hi:g}"
        band_mask = (range_v >= lo) & (range_v < hi)
        band_n = int(band_mask.sum())
        if band_n == 0:
            distance_bands[band_key] = DistanceBandMetrics(band=band_key, point_count=0, accuracy=None, mean_iou=None)
            continue
        bp, bg = pred_v[band_mask], gt_v[band_mask]
        band_acc = float((bp == bg).mean())
        band_ious = []
        for cid in class_ids:
            if cid in CANONICAL_IGNORE_IN_MIOU:
                continue
            tp = int(np.sum((bp == cid) & (bg == cid)))
            fp = int(np.sum((bp == cid) & (bg != cid)))
            fn = int(np.sum((bp != cid) & (bg == cid)))
            denom = tp + fp + fn
            if denom > 0:
                band_ious.append(tp / denom)
        band_miou = float(np.mean(band_ious)) if band_ious else None
        distance_bands[band_key] = DistanceBandMetrics(
            band=band_key, point_count=band_n, accuracy=band_acc, mean_iou=band_miou
        )

    return EvaluationResult(
        num_points_total=n_total,
        num_points_ignored=n_ignored,
        num_points_evaluated=n_eval,
        overall_accuracy=overall_accuracy,
        mean_iou=mean_iou,
        per_class_iou=per_class_iou,
        confusion_matrix=confusion.tolist(),
        confusion_matrix_labels=[CANONICAL_CLASSES[c] for c in class_ids],
        distance_bands=distance_bands,
        model_coverage=coverage,
        excluded_canonical_classes=excluded,
    )
