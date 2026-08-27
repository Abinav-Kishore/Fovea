"""Non-visual benchmark: runs Uniform, Distance-Adaptive, and FOVEA on the same
frame(s) and writes measured metrics to JSON. Every number is measured from an
actual run; nothing here is fabricated.

Usage:
    python -m src.benchmark --config config.yaml
    python -m src.benchmark --config config.yaml --sequence 00 --num-frames 10
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data_loader import list_frame_ids, load_frame
from src.distance_baseline import build_distance_adaptive_map
from src.fovea_allocator import build_fovea_map
from src.metrics import benchmark_latency, environment_info, summarize_method
from src.uniform_baseline import build_uniform_map


def run_benchmark(cfg: dict, sequence: str, num_frames: int, repeats: int, adapter=None) -> dict:
    """`adapter` is an OPTIONAL src.datasets.DatasetAdapter. When omitted
    (default), behavior is byte-identical to v0.1: frames come from
    cfg["data"]["root"] via data_loader's SemanticPOSS-specific functions.
    When provided (e.g. a RELLIS-3D adapter), frame discovery/loading and
    class-name resolution go through the adapter instead -- no scoring/
    allocation logic changes either way."""
    if adapter is not None:
        frame_ids = adapter.list_frames(sequence)[:num_frames]
        label_dict = adapter.label_dict()
    else:
        frame_ids = list_frame_ids(cfg["data"]["root"], sequence)[:num_frames]
        label_dict = None
    if not frame_ids:
        root = adapter.name if adapter is not None else cfg["data"]["root"]
        raise RuntimeError(f"No frames found for sequence {sequence} under {root}")

    per_frame_results = []
    method_fns = {
        "uniform": build_uniform_map,
        "distance_adaptive": build_distance_adaptive_map,
        "fovea": build_fovea_map,
    }
    bands = cfg["distance_adaptive"]["bands"]

    sequence_start = datetime.datetime.now()
    for fid in frame_ids:
        frame = adapter.load_frame(sequence, fid) if adapter is not None else load_frame(cfg["data"]["root"], sequence, fid)
        frame_result = {"frame_id": fid, "num_points": int(frame.points.shape[0]), "methods": {}}
        for name, fn in method_fns.items():
            cells, acc = fn(frame.points, frame.semantic_labels, cfg, label_dict=label_dict)
            summary = summarize_method(cells, acc, bands)
            timing = benchmark_latency(fn, frame.points, frame.semantic_labels, cfg, repeats, label_dict=label_dict)
            summary["timing_repeats"] = timing
            frame_result["methods"][name] = summary
        per_frame_results.append(frame_result)

    sequence_elapsed = (datetime.datetime.now() - sequence_start).total_seconds()
    n_frames = len(frame_ids)
    throughput_fps = n_frames / sequence_elapsed if sequence_elapsed > 0 else None

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": cfg,
        "dataset": adapter.name if adapter is not None else "SemanticPOSS",
        "sequence": sequence,
        "num_frames_processed": n_frames,
        "environment": environment_info(),
        "sequence_processing": {
            "total_wall_time_s": sequence_elapsed,
            "note": "Wall time for loading + running all 3 methods per frame, sequentially, single process.",
            "approx_throughput_frames_per_s": throughput_fps,
        },
        "per_frame_results": per_frame_results,
    }


def main():
    parser = argparse.ArgumentParser(description="FOVEA v0.1 benchmark (Uniform vs Distance-Adaptive vs FOVEA)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sequence", default=None, help="Override config.yaml data.sequence")
    parser.add_argument("--num-frames", type=int, default=None, help="Override config.yaml benchmark.num_frames")
    parser.add_argument("--repeats", type=int, default=None, help="Override config.yaml benchmark.repeats_for_timing")
    parser.add_argument("--out", default="results/benchmark_result.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sequence = args.sequence or cfg["data"]["sequence"]
    num_frames = args.num_frames or cfg["benchmark"]["num_frames"]
    repeats = args.repeats or cfg["benchmark"]["repeats_for_timing"]

    print(f"Running benchmark: sequence={sequence} num_frames={num_frames} repeats={repeats}")
    result = run_benchmark(cfg, sequence, num_frames, repeats)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    print("\n=== Summary (last frame) ===")
    last = result["per_frame_results"][-1]
    for method, summary in last["methods"].items():
        print(f"{method:20s} cells={summary['output_cell_count']:6d}  "
              f"latency={summary['map_generation_latency_s']:.4f}s  "
              f"assign_rate={summary['assignment_rate']*100:.1f}%  "
              f"storage_kb={summary['estimated_map_storage_kb']:.1f}")


if __name__ == "__main__":
    main()
