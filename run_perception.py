#!/usr/bin/env python3
"""Real perception CLI: load a real LiDAR frame, run real inference (never
ground truth passthrough), report the actual measured results, and
optionally run FOVEA on the predictions.

Usage:
    python run_perception.py --dataset SemanticPOSS --sequence 00 --frame 000000
    python run_perception.py --dataset RELLIS-3D --sequence 00000 --frame 000100 --fovea
    python run_perception.py --dataset SemanticPOSS --sequence 00 --frame 000000 --no-eval
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.config import load_config
from src.datasets import get_adapter
from src.metrics import summarize_method
from src.perception.evaluation import evaluate_predictions
from src.perception.inference import run_perception
from src.perception.model import get_squeezeseg_model
from src.perception.ontology import CANONICAL_CLASSES, map_names_to_canonical


def _print_distribution(title: str, labels: np.ndarray, name_lookup) -> None:
    n = len(labels)
    ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    print(f"\n{title}")
    for i in order:
        name = name_lookup(int(ids[i]))
        count = int(counts[i])
        print(f"    {name:20s} {count:8,d}  ({100 * count / n:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="FOVEA real-perception CLI (src/perception/)")
    parser.add_argument("--dataset", default="SemanticPOSS", choices=["SemanticPOSS", "RELLIS-3D"])
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--frame", default="000000")
    parser.add_argument("--model", default="squeezeseg")
    parser.add_argument("--fovea", action="store_true", help="Also run FOVEA on the predicted labels and report cell counts.")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation even if ground truth is available.")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    adapter = get_adapter(args.dataset, cfg)

    print(f"=== Loading real frame: dataset={args.dataset} sequence={args.sequence} frame={args.frame} ===")
    frame = adapter.load_frame(args.sequence, args.frame)
    print(f"points: {frame.points.shape[0]:,}")
    print(f"has_labels (ground truth available for evaluation only): {frame.has_labels}")

    print(f"\n=== Loading model: {args.model} ===")
    if args.model != "squeezeseg":
        print(f"No loader implementation for '{args.model}'.", file=sys.stderr)
        sys.exit(1)
    model = get_squeezeseg_model(cfg)
    print(f"model: {model.name}")
    print(f"ontology: {model.model_ontology}")
    print(f"device: {model.device}")

    print("\n=== Running real inference (points -> model -> predicted labels) ===")
    print("NOTE: ground truth is NEVER passed to the model -- only frame.points.")
    result = run_perception(model, frame.points)
    print(f"preprocessing:  {result.preprocessing_ms:8.1f} ms")
    print(f"inference:      {result.inference_ms:8.1f} ms")
    print(f"postprocessing: {result.postprocessing_ms:8.1f} ms")
    print(f"total:          {result.total_perception_ms:8.1f} ms")
    print(f"points/sec:     {result.points_per_second:,.0f}")
    if result.confidence is not None:
        print(f"mean confidence: {result.confidence.mean():.3f} (model provides real softmax confidence)")
    else:
        print("confidence: N/A (model does not provide it -- never fabricated)")

    _print_distribution(
        f"Predicted class distribution ({model.model_ontology}, model's own raw ontology):",
        result.predicted_labels,
        lambda i: model.raw_label_names.get(i, "unknown"),
    )
    _print_distribution(
        "Predicted class distribution (canonical, cross-dataset ontology):",
        result.predicted_canonical,
        lambda i: CANONICAL_CLASSES.get(i, "unknown"),
    )

    if frame.has_labels and not args.no_eval:
        print("\n=== Evaluation: predictions vs. REAL ground truth (evaluation only, never a prediction) ===")
        gt_names = [adapter.class_name(int(label_id)) for label_id in frame.semantic_labels]
        gt_canonical = np.asarray(map_names_to_canonical(gt_names, args.dataset), dtype=np.int64)
        range_m = np.linalg.norm(frame.points[:, :3], axis=1)
        ev = evaluate_predictions(result.predicted_canonical, gt_canonical, range_m, result.model_ontology)

        print(f"points total / ignored(unknown-gt) / evaluated: {ev.num_points_total:,} / {ev.num_points_ignored:,} / {ev.num_points_evaluated:,}")
        print(f"model_coverage (fraction of GT whose canonical class the model can ever predict): {ev.model_coverage:.3f}")
        if ev.excluded_canonical_classes:
            print(f"canonical classes present in GT but UNREACHABLE by this model's ontology: {ev.excluded_canonical_classes}")
        print(f"overall_accuracy: {ev.overall_accuracy:.4f}")
        print(f"mean_iou:         {ev.mean_iou:.4f}")
        print("\nper-class IoU (canonical):")
        for name, iou in ev.per_class_iou.items():
            print(f"    {name:15s} {'N/A (absent from both / ignored)' if iou is None else f'{iou:.4f}'}")
        print("\ndistance-band metrics:")
        for band, m in ev.distance_bands.items():
            if m.point_count == 0:
                print(f"    {band:>8s} m: 0 points -- not measured")
            else:
                print(f"    {band:>8s} m: n={m.point_count:6,d}  accuracy={m.accuracy:.4f}  mean_iou={m.mean_iou:.4f}")
    elif not frame.has_labels:
        print("\n=== Evaluation skipped: this frame has no ground truth. ===")

    if args.fovea:
        print("\n=== Running FOVEA on PREDICTED labels (inference mode) ===")
        from src.distance_baseline import build_distance_adaptive_map
        from src.fovea_allocator import build_fovea_map
        from src.uniform_baseline import build_uniform_map

        models_cfg = cfg["perception"]["models"][args.model]
        effective_cfg = copy.deepcopy(cfg)
        effective_cfg["fovea"]["semantic_priority"] = models_cfg["semantic_priority"]
        effective_cfg["fovea"]["dynamic_classes"] = list(models_cfg["dynamic_classes"])

        for name, fn in [
            ("uniform", build_uniform_map),
            ("distance_adaptive", build_distance_adaptive_map),
            ("fovea", build_fovea_map),
        ]:
            cells, accounting = fn(frame.points, result.predicted_labels, effective_cfg, label_dict=model.raw_label_names)
            summary = summarize_method(cells, accounting, effective_cfg["distance_adaptive"]["bands"])
            print(f"  {name:20s} cells={summary['output_cell_count']:6,d}  latency={summary['map_generation_latency_s']:.3f}s")


if __name__ == "__main__":
    main()
