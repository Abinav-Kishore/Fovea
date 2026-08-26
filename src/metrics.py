"""Measured metrics for comparing Uniform vs Distance-Adaptive vs FOVEA.

Every number here is computed from an actual run against the actual loaded
frame -- nothing in this module is a fabricated or hardcoded benchmark value.
"""
from __future__ import annotations

import platform
import sys
import time
from collections import Counter

import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from src.map_structure import Cell

# Bytes per Cell, estimated from its actual field layout (8 floats + 4 ints +
# 1 short string reference + small dict). This is a documented estimate of
# raw record storage, not a claim about Python object overhead or a measured
# process RSS delta.
_ESTIMATED_BYTES_PER_CELL = 8 * 8 + 4 * 4 + 32  # 8 float64 fields, 4 int fields, ~32B for name+dict overhead


def process_memory_mb() -> float | None:
    if not _HAS_PSUTIL:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def distance_band_edges(bands: list[dict]) -> list[float]:
    edges = [0.0]
    for b in bands:
        edges.append(float(b["max_m"]))
    return edges


def stratify_by_distance(cells: list[Cell], bands: list[dict]) -> dict:
    """Cells/points/semantic coverage grouped by the configured distance bands,
    using each cell's actual center range (works even for the uniform baseline,
    which has no band concept baked in -- we just bucket its cells post hoc)."""
    edges = distance_band_edges(bands)
    result = {}
    ranges = np.array([c.range_m for c in cells]) if cells else np.array([])
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (ranges >= lo) & (ranges < hi)
        band_cells = [c for c, m in zip(cells, mask) if m]
        classes = Counter(c.majority_class_name for c in band_cells)
        result[f"{lo:g}-{hi:g}m"] = {
            "cell_count": len(band_cells),
            "point_count": int(sum(c.point_count for c in band_cells)),
            "semantic_classes_present": dict(classes),
        }
    beyond = ranges >= edges[-1] if len(ranges) else np.array([], dtype=bool)
    if beyond.any():
        band_cells = [c for c, m in zip(cells, beyond) if m]
        result[f">{edges[-1]:g}m"] = {
            "cell_count": len(band_cells),
            "point_count": int(sum(c.point_count for c in band_cells)),
            "semantic_classes_present": dict(Counter(c.majority_class_name for c in band_cells)),
        }
    return result


def summarize_method(cells: list[Cell], accounting: dict, bands: list[dict]) -> dict:
    resolution_counts = Counter(c.resolution_level for c in cells)
    cell_sizes = sorted(set(round(c.cell_size_m, 6) for c in cells))
    estimated_storage_bytes = len(cells) * _ESTIMATED_BYTES_PER_CELL
    semantic_classes_covered = len(set(c.majority_class_name for c in cells))

    return {
        "method": accounting.get("method"),
        "output_cell_count": len(cells),
        "cells_by_resolution_level": {int(k): int(v) for k, v in resolution_counts.items()},
        "distinct_cell_sizes_m": cell_sizes,
        "input_points": accounting.get("input_points"),
        "assigned_points": accounting.get("assigned_points"),
        "assignment_rate": accounting.get("assignment_rate"),
        "out_of_envelope_points": accounting.get("out_of_envelope_points", 0),
        "invalid_points": accounting.get("invalid_points", 0),
        "map_generation_latency_s": accounting.get("latency_s"),
        "estimated_map_storage_bytes": estimated_storage_bytes,
        "estimated_map_storage_kb": estimated_storage_bytes / 1024,
        "semantic_classes_covered": semantic_classes_covered,
        "distance_band_stats": stratify_by_distance(cells, bands),
    }


def benchmark_latency(build_fn, points, labels, cfg, repeats: int) -> dict:
    """Repeated-run timing for a single map-generation function. Reports real
    wall-clock latency only -- labelled as single-frame latency, never as FPS,
    since a single static frame is not a throughput measurement."""
    times = []
    n_cells_last = 0
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        cells, _ = build_fn(points, labels, cfg)
        times.append(time.perf_counter() - t0)
        n_cells_last = len(cells)
    times = np.array(times)
    return {
        "repeats": len(times),
        "mean_latency_s": float(times.mean()),
        "std_latency_s": float(times.std()),
        "min_latency_s": float(times.min()),
        "max_latency_s": float(times.max()),
        "output_cell_count": n_cells_last,
    }


def environment_info() -> dict:
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": __import__("numpy").__version__,
        "pandas_version": __import__("pandas").__version__,
    }
    mem = process_memory_mb()
    if mem is not None:
        info["process_rss_mb"] = round(mem, 2)
    return info
