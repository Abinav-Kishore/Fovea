"""SIH compliance benchmarks (Phase 4C).

This module implements NO new FOVEA allocation, hierarchy, LOD, chunk, or
perception logic -- it is a read-only measurement layer over the EXISTING,
UNMODIFIED engine (src/fovea_allocator.py, src/uniform_baseline.py,
src/spatial/hierarchy.py, lod.py, chunks.py, map_2_5d.py, src/perception/*).
Every function below runs one or more of those real functions against real
local SemanticPOSS/RELLIS-3D frames and reports exactly what came out --
nothing here is fabricated, interpolated, or assumed. Six functions, one per
SIH requirement:

    A. resolution_vs_distance_report   -- adaptive resolution vs. distance
    B. coverage_and_conservation_report -- spatial coverage + point conservation
    C. memory_benchmark_report          -- uniform high-res vs. FOVEA memory
    D. pipeline_latency_report          -- end-to-end backend stage timing
    E. perception_distance_bands_report -- distance-wise perception accuracy/mIoU
    F. terrain_obstacle_report          -- terrain/obstacle classification counts

`run_sih_compliance_benchmark` orchestrates all six across a real set of
frames and returns one JSON-safe report dict (see `main()` for the CLI that
writes it to results/sih_compliance_benchmark.json).
"""
from __future__ import annotations

import copy
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.config import load_config
from src.datasets import DatasetAdapter, get_adapter
from src.fovea_allocator import build_fovea_map
from src.map_structure import Cell
from src.metrics import benchmark_latency, process_memory_mb, summarize_method
from src.perception.evaluation import evaluate_predictions
from src.perception.ontology import map_names_to_canonical
from src.spatial.chunks import serialize_chunks_with_metadata
from src.spatial.hierarchy import build_fovea_hierarchy
from src.spatial.lod import LODSelectionConfig, select_lod_nodes
from src.spatial.map_2_5d import build_2_5d_map, validate_point_conservation
from src.uniform_baseline import build_uniform_map


def effective_cfg_for(dataset: str, cfg: dict, adapter: DatasetAdapter) -> dict:
    """Same per-dataset semantic_priority/dynamic_classes substitution
    src/api.py's `_dataset_context` already uses -- copied here rather than
    imported, since api.py's version is bound to its own module-level cache;
    the substitution itself is the same one-line pattern every dataset-aware
    test in this repo already does, not a new mechanism."""
    if dataset == "SemanticPOSS":
        return cfg
    effective = copy.deepcopy(cfg)
    effective["fovea"]["semantic_priority"] = adapter.semantic_priority_table()
    effective["fovea"]["dynamic_classes"] = list(adapter.dynamic_classes())
    return effective


def _band_edges(bands_cfg: list[dict]) -> list[tuple[float, float]]:
    lo = 0.0
    edges = []
    for b in bands_cfg:
        hi = float(b["max_m"])
        edges.append((lo, hi))
        lo = hi
    return edges


# ---------------------------------------------------------------------------
# A. Adaptive resolution vs. distance
# ---------------------------------------------------------------------------

# The SIH brief's own stated example is two anchor points -- "approximately
# 5cm near 10m progressing toward approximately 50cm by 100m" -- not a full
# per-band target curve. Reported verbatim; the two interior bands' figures
# below are the EXISTING config.yaml distance_adaptive.bands values, not
# independently SIH-specified targets, and are labeled as such.
SIH_STATED_ANCHORS = {"near_10m_target_cell_size_m": 0.05, "by_100m_target_cell_size_m": 0.50}


def resolution_vs_distance_report(cells: list[Cell], bands_cfg: list[dict]) -> dict:
    """Requirement A. Groups the ACTUAL FOVEA cells from an unmodified
    build_fovea_map() call by their own `band_id` (assigned during
    allocation by src.geometry.assign_bands -- authoritative, not
    re-derived here) and reports real min/median/max cell_size_m and the
    most common resolution_level per band."""
    by_band: dict[int, list[Cell]] = {}
    for c in cells:
        by_band.setdefault(c.band_id, []).append(c)

    band_reports = []
    for i, (lo, hi) in enumerate(_band_edges(bands_cfg)):
        band_cells = by_band.get(i, [])
        sizes = sorted(c.cell_size_m for c in band_cells)
        levels = Counter(c.resolution_level for c in band_cells)
        band_reports.append({
            "band": f"{lo:g}-{hi:g}m",
            "configured_band_base_cell_size_m": float(bands_cfg[i]["cell_size_m"]),
            "cell_count": len(band_cells),
            "min_cell_size_m": sizes[0] if sizes else None,
            "median_cell_size_m": float(np.median(sizes)) if sizes else None,
            "max_cell_size_m": sizes[-1] if sizes else None,
            "resolution_level_distribution": {int(k): int(v) for k, v in levels.items()},
            "representative_resolution_level": (levels.most_common(1)[0][0] if levels else None),
        })

    return {
        "sih_stated_anchors": SIH_STATED_ANCHORS,
        "note": (
            "SIH gives two example anchors (~5cm near 10m, ~50cm by 100m), not a "
            "full per-band curve -- the 10-30m/30-60m figures below are the "
            "EXISTING config's own values, not independently SIH-specified. "
            "By construction, min_cell_size_m <= max_cell_size_m <= "
            "configured_band_base_cell_size_m always holds within a band "
            "(FOVEA can only refine finer than the band's base, never coarser "
            "-- see src/fovea_allocator.py's module docstring); whether the "
            "band's max actually EQUALS its base depends on whether any cell "
            "in that band stayed unsplit, which is measured, not assumed."
        ),
        "bands": band_reports,
    }


# ---------------------------------------------------------------------------
# B. Coverage + point conservation
# ---------------------------------------------------------------------------

def coverage_and_conservation_report(cells: list[Cell], accounting: dict, points: np.ndarray, bands_cfg: list[dict]) -> dict:
    """Requirement B. Real spatial coverage per distance band, out-of-
    envelope/invalid point counts, and an independent point-conservation
    re-check via src.spatial.map_2_5d.validate_point_conservation (existing,
    reused -- not reimplemented)."""
    ranges = np.linalg.norm(points[:, :3], axis=1) if points.shape[0] else np.array([])
    max_range_m = float(ranges.max()) if len(ranges) else 0.0

    by_band: dict[int, list[Cell]] = {}
    for c in cells:
        by_band.setdefault(c.band_id, []).append(c)

    band_reports = []
    for i, (lo, hi) in enumerate(_band_edges(bands_cfg)):
        band_cells = by_band.get(i, [])
        band_reports.append({
            "band": f"{lo:g}-{hi:g}m",
            "cell_count": len(band_cells),
            "point_count": int(sum(c.point_count for c in band_cells)),
            "has_data": len(band_cells) > 0,
        })

    conservation = validate_point_conservation(accounting)

    return {
        "input_points": int(points.shape[0]),
        "max_observed_range_m": max_range_m,
        "coverage_reaches_100m": max_range_m >= 100.0,
        "coverage_note": (
            f"Real measured max point range this frame is {max_range_m:.1f}m -- "
            f"the representation covers real data out to that range, never a "
            f"claimed 100m the actual scan didn't reach."
        ),
        "bands": band_reports,
        "out_of_envelope_points": accounting["out_of_envelope_points"],
        "invalid_points": accounting["invalid_points"],
        "assigned_points": accounting["assigned_points"],
        "assignment_rate": accounting["assignment_rate"],
        "point_conservation": conservation,
    }


# ---------------------------------------------------------------------------
# C. Uniform high-res vs. FOVEA memory benchmark
# ---------------------------------------------------------------------------

def memory_benchmark_report(points: np.ndarray, labels: np.ndarray, cfg: dict, label_dict=None, repeats: int = 1) -> dict:
    """Requirement C. Same real frame, same cfg, both build functions
    EXISTING and UNMODIFIED (build_uniform_map / build_fovea_map). Cell
    counts/storage/assigned-points/build-time via src.metrics.summarize_method
    /benchmark_latency (existing, reused). FVCH1 payload size is FOVEA-only
    -- uniform has no hierarchy/chunk representation to serialize; that
    asymmetry is documented, not glossed over."""
    bands_cfg = cfg["distance_adaptive"]["bands"]

    uniform_cells, uniform_acc = build_uniform_map(points, labels, cfg, label_dict=label_dict)
    uniform_summary = summarize_method(uniform_cells, uniform_acc, bands_cfg)
    uniform_timing = benchmark_latency(build_uniform_map, points, labels, cfg, repeats, label_dict=label_dict)

    fovea_cells, fovea_acc = build_fovea_map(points, labels, cfg, label_dict=label_dict)
    fovea_summary = summarize_method(fovea_cells, fovea_acc, bands_cfg)
    fovea_timing = benchmark_latency(build_fovea_map, points, labels, cfg, repeats, label_dict=label_dict)

    roots, _hstats = build_fovea_hierarchy(fovea_cells, fovea_acc, cfg)
    selection = select_lod_nodes(roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=0.02))
    _payload, meta = serialize_chunks_with_metadata(
        selection.nodes, "ground_truth", "benchmark", "benchmark", "benchmark", "benchmark"
    )

    fovea_cell_count = fovea_summary["output_cell_count"]
    fovea_storage = fovea_summary["estimated_map_storage_bytes"]

    return {
        "assumptions": (
            "Same real frame (points/labels), same cfg -- uniform uses "
            "cfg['uniform']['cell_size_m'] (the existing high-resolution "
            "baseline cell size, currently "
            f"{cfg['uniform']['cell_size_m']}m), FOVEA uses its own adaptive "
            "allocation. Uniform has NO distance envelope, so it in fact "
            "covers MORE far-range area than FOVEA's envelope -- a real "
            "asymmetry in uniform's favor, stated here rather than hidden."
        ),
        "uniform": {
            "cell_size_m": cfg["uniform"]["cell_size_m"],
            "output_cell_count": uniform_summary["output_cell_count"],
            "estimated_storage_bytes": uniform_summary["estimated_map_storage_bytes"],
            "assigned_points": uniform_summary["assigned_points"],
            "mean_build_latency_s": uniform_timing["mean_latency_s"],
        },
        "fovea": {
            "output_cell_count": fovea_cell_count,
            "estimated_storage_bytes": fovea_storage,
            "assigned_points": fovea_summary["assigned_points"],
            "mean_build_latency_s": fovea_timing["mean_latency_s"],
            "fvch1_serialized_payload_bytes": meta.payload_bytes,
            "fvch1_chunk_count": meta.chunk_count,
            "fvch1_note": "FOVEA-only figure -- uniform has no chunk/streaming representation to compare against.",
        },
        "cell_count_reduction_factor_uniform_over_fovea": (
            uniform_summary["output_cell_count"] / fovea_cell_count if fovea_cell_count else None
        ),
        "estimated_storage_reduction_factor_uniform_over_fovea": (
            uniform_summary["estimated_map_storage_bytes"] / fovea_storage if fovea_storage else None
        ),
    }


# ---------------------------------------------------------------------------
# D. End-to-end pipeline latency
# ---------------------------------------------------------------------------

def pipeline_latency_report(points: np.ndarray, labels: np.ndarray, cfg: dict, label_dict=None, perception_model=None) -> dict:
    """Requirement D. Real backend stage timings on real local data: FOVEA
    build -> hierarchy build -> LOD selection -> chunk serialization (all
    existing, unmodified). Perception timing included only when a real
    loaded model is passed in. Frontend decode/FPS are NOT re-measured here
    (this is a Python-only benchmark, no browser) -- the real numbers from
    actual prior browser validation (Phase 3D / Phase 4AB) are cited with
    that provenance stated explicitly, never presented as freshly measured
    by this function. FPS is explicitly labeled non-representative
    (headless/software-rendered), per the task's own instruction."""
    mem_before = process_memory_mb()

    perception_ms = None
    if perception_model is not None:
        from src.perception.inference import run_perception

        result = run_perception(perception_model, points)
        perception_ms = result.total_perception_ms

    t0 = time.perf_counter()
    cells, accounting = build_fovea_map(points, labels, cfg, label_dict=label_dict)
    fovea_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    roots, _hstats = build_fovea_hierarchy(cells, accounting, cfg)
    hierarchy_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    selection = select_lod_nodes(roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=0.02))
    lod_s = time.perf_counter() - t2

    t3 = time.perf_counter()
    _payload, _meta = serialize_chunks_with_metadata(
        selection.nodes, "ground_truth", "benchmark", "benchmark", "benchmark", "benchmark"
    )
    serialize_s = time.perf_counter() - t3

    mem_after = process_memory_mb()
    backend_total_s = fovea_s + hierarchy_s + lod_s + serialize_s + ((perception_ms or 0.0) / 1000.0)

    return {
        "perception_ms": perception_ms,
        "fovea_build_s": fovea_s,
        "hierarchy_build_s": hierarchy_s,
        "lod_selection_s": lod_s,
        "chunk_serialization_s": serialize_s,
        "backend_total_s": backend_total_s,
        "process_rss_mb_before": mem_before,
        "process_rss_mb_after": mem_after,
        "process_rss_mb_delta": (
            (mem_after - mem_before) if (mem_before is not None and mem_after is not None) else None
        ),
        "frontend_cited_not_remeasured": {
            "decode_ms_range_from_phase_3d_browser_validation": [8, 16],
            "fps_from_phase_4ab_browser_validation": 29,
            "fps_is_representative_of_real_gpu_hardware": False,
            "fps_provenance": "headless Chromium, CPU-side SwiftShader software rendering -- explicitly NOT a hardware performance claim (see README Phase 4AB section).",
            "provenance": "Not re-measured in this benchmark run (Python-only, no browser). Cited from the real browser measurements already recorded in README's Phase 3D and Phase 4AB sections.",
        },
    }


# ---------------------------------------------------------------------------
# E. Distance-wise perception benchmark
# ---------------------------------------------------------------------------

def perception_distance_bands_report(dataset_name: str, frames: list[tuple[str, str]], adapter: DatasetAdapter, model) -> dict:
    """Requirement E. Reuses src.perception.evaluation.evaluate_predictions
    (existing, unmodified) -- it already produces exactly the 0-10/10-30/
    30-60/60-100m bands this requirement asks for. Real inference
    (src.perception.inference.run_perception, unmodified) against real
    ground truth per (sequence, frame_id). No absent-class/absent-band
    metric is invented -- evaluate_predictions' own convention (None, not 0)
    is passed through unchanged."""
    from src.perception.inference import run_perception

    per_frame = []
    for sequence, frame_id in frames:
        frame = adapter.load_frame(sequence, frame_id)
        if not frame.has_labels:
            per_frame.append({"sequence": sequence, "frame_id": frame_id, "available": False, "message": "no ground truth"})
            continue
        result = run_perception(model, frame.points)
        gt_names = [adapter.class_name(int(lid)) for lid in frame.semantic_labels]
        gt_canonical = np.asarray(map_names_to_canonical(gt_names, dataset_name), dtype=np.int64)
        range_m = np.linalg.norm(frame.points[:, :3], axis=1)
        ev = evaluate_predictions(result.predicted_canonical, gt_canonical, range_m, result.model_ontology)
        per_frame.append({
            "sequence": sequence,
            "frame_id": frame_id,
            "available": True,
            "num_points_total": ev.num_points_total,
            "overall_accuracy": ev.overall_accuracy,
            "mean_iou": ev.mean_iou,
            "model_coverage": ev.model_coverage,
            "distance_bands": {k: asdict(v) for k, v in ev.distance_bands.items()},
        })

    return {"dataset": dataset_name, "model_ontology": model.model_ontology, "per_frame": per_frame}


# ---------------------------------------------------------------------------
# F. Terrain / static-obstacle reporting
# ---------------------------------------------------------------------------

def terrain_obstacle_report(cells: list[Cell], accounting: dict, dataset_name: str) -> dict:
    """Requirement F. Reuses src.spatial.map_2_5d.build_2_5d_map (existing,
    unmodified) -- the terrain/obstacle rule tables are Phase 2's own
    (src/spatial/terrain.py, obstacle.py), not touched or reimplemented."""
    _perception_cells, stats = build_2_5d_map(cells, accounting, "ground_truth", dataset_name)
    return {
        "label_source": "ground_truth",
        "source": dataset_name,
        "note": (
            "Semantic-DERIVED classification via Phase 2's terrain.py/obstacle.py "
            "rule tables -- NOT validated against real terrain ground truth "
            "(neither dataset has any; see src/spatial/map_2_5d.py's own "
            "docstring). potential_dynamic_obstacle is a semantic-category "
            "proxy, not real motion detection (documented since Phase 0)."
        ),
        "total_cells": stats.total_cells,
        "drivable_cells": stats.drivable_cells,
        "non_drivable_cells": stats.non_drivable_cells,
        "obstacle_cells": stats.obstacle_cells,
        "unknown_terrain_cells": stats.unknown_terrain_cells,
        "static_obstacle_cells": stats.static_obstacle_cells,
        "potential_dynamic_obstacle_cells": stats.potential_dynamic_obstacle_cells,
    }


# ---------------------------------------------------------------------------
# Orchestration -- runs A-F across real local frames from both datasets.
# ---------------------------------------------------------------------------

DEFAULT_SEMANTICPOSS_FRAMES = [("00", "000000"), ("00", "000050"), ("00", "000100")]
DEFAULT_RELLIS_FRAMES = [("00000", "000100"), ("00002", "000500"), ("00004", "000050")]


def _dataset_section(dataset_name: str, frames: list[tuple[str, str]], cfg: dict, perception_model) -> dict:
    adapter = get_adapter(dataset_name, cfg)
    effective_cfg = effective_cfg_for(dataset_name, cfg, adapter)
    bands_cfg = effective_cfg["distance_adaptive"]["bands"]

    all_cells: list[Cell] = []
    per_frame_c: list[dict] = []
    per_frame_d: list[dict] = []
    per_frame_f: list[dict] = []
    last_cells, last_accounting, last_points, last_labels = None, None, None, None

    for sequence, frame_id in frames:
        frame = adapter.load_frame(sequence, frame_id)
        cells, accounting = build_fovea_map(
            frame.points, frame.semantic_labels, effective_cfg, label_dict=adapter.label_dict()
        )
        all_cells.extend(cells)
        last_cells, last_accounting, last_points, last_labels = cells, accounting, frame.points, frame.semantic_labels

        per_frame_c.append({
            "sequence": sequence, "frame_id": frame_id,
            **memory_benchmark_report(frame.points, frame.semantic_labels, effective_cfg, label_dict=adapter.label_dict()),
        })
        per_frame_d.append({
            "sequence": sequence, "frame_id": frame_id,
            **pipeline_latency_report(
                frame.points, frame.semantic_labels, effective_cfg,
                label_dict=adapter.label_dict(), perception_model=perception_model,
            ),
        })
        per_frame_f.append({
            "sequence": sequence, "frame_id": frame_id,
            **terrain_obstacle_report(cells, accounting, dataset_name),
        })

    return {
        "dataset": dataset_name,
        "frames_used": [f"{s}/{f}" for s, f in frames],
        "A_resolution_vs_distance": resolution_vs_distance_report(all_cells, bands_cfg),
        "B_coverage_and_conservation_last_frame": coverage_and_conservation_report(
            last_cells, last_accounting, last_points, bands_cfg
        ),
        "C_memory_benchmark_per_frame": per_frame_c,
        "D_pipeline_latency_per_frame": per_frame_d,
        "F_terrain_obstacle_per_frame": per_frame_f,
    }


def run_sih_compliance_benchmark(
    cfg: dict,
    semanticposs_frames: list[tuple[str, str]] = DEFAULT_SEMANTICPOSS_FRAMES,
    rellis_frames: list[tuple[str, str]] = DEFAULT_RELLIS_FRAMES,
    include_perception: bool = True,
) -> dict:
    """Runs requirements A-F across real local SemanticPOSS + RELLIS-3D
    frames (>=3 each by default) and returns one JSON-safe report dict.
    `include_perception=False` skips loading the SqueezeSeg checkpoint
    (requirement E, and the perception stage of D) -- useful for a fast
    FOVEA-only run when the checkpoint isn't available."""
    from pathlib import Path as _Path

    perception_model = None
    if include_perception:
        from src.perception.model import get_squeezeseg_model

        perception_model = get_squeezeseg_model(cfg)

    report: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    report["SemanticPOSS"] = _dataset_section("SemanticPOSS", semanticposs_frames, cfg, perception_model)

    rellis_present = _Path(cfg["datasets"]["RELLIS-3D"]["pointcloud_root"]).exists()
    if rellis_present:
        report["RELLIS-3D"] = _dataset_section("RELLIS-3D", rellis_frames, cfg, perception_model)
    else:
        report["RELLIS-3D"] = {"available": False, "message": "RELLIS-3D data not present on this machine."}

    if include_perception and perception_model is not None:
        sp_adapter = get_adapter("SemanticPOSS", cfg)
        report["E_perception_distance_bands"] = {
            "SemanticPOSS": perception_distance_bands_report(
                "SemanticPOSS", semanticposs_frames, sp_adapter, perception_model
            )
        }
        if rellis_present:
            rl_adapter = get_adapter("RELLIS-3D", cfg)
            report["E_perception_distance_bands"]["RELLIS-3D"] = perception_distance_bands_report(
                "RELLIS-3D", rellis_frames, rl_adapter, perception_model
            )

    return report


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="FOVEA Phase 4C -- SIH compliance benchmarks")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="results/sih_compliance_benchmark.json")
    parser.add_argument("--no-perception", action="store_true", help="Skip loading the SqueezeSeg checkpoint (requirement E)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Running SIH compliance benchmark on real local data (SemanticPOSS + RELLIS-3D if present)...")
    result = run_sih_compliance_benchmark(cfg, include_perception=not args.no_perception)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
