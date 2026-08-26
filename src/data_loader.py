"""SemanticPOSS frame loader.

Verified format (inspected directly from the dataset files, not assumed):
  sequences/<seq>/velodyne/<frame>.bin  -- float32, reshape(-1,4) = x,y,z,intensity
  sequences/<seq>/labels/<frame>.label  -- uint32; semantic = value & 0xFFFF,
                                            instance = value >> 16
Point count in .bin and label count in .label match exactly for every frame
checked (e.g. frame 000000 of sequence 00: 66658 points == 66658 labels).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.semantic_labels import class_name

logger = logging.getLogger("fovea.data_loader")


@dataclass
class Frame:
    sequence: str
    frame_id: str
    points: np.ndarray        # (N, 4) float32: x, y, z, intensity
    semantic_labels: np.ndarray  # (N,) uint32
    instance_labels: np.ndarray  # (N,) uint32
    has_labels: bool
    bin_path: Path
    label_path: Path | None


def _sequence_dir(data_root: str | Path, sequence: str) -> Path:
    seq_dir = Path(data_root) / "sequences" / sequence
    if not seq_dir.exists():
        available = sorted(p.name for p in (Path(data_root) / "sequences").glob("*")) \
            if (Path(data_root) / "sequences").exists() else []
        raise FileNotFoundError(
            f"Sequence directory not found: {seq_dir}\n"
            f"Expected data_root/sequences/<sequence>. "
            f"data_root={data_root}. Available sequences: {available}"
        )
    return seq_dir


def list_frame_ids(data_root: str | Path, sequence: str) -> list[str]:
    velodyne_dir = _sequence_dir(data_root, sequence) / "velodyne"
    if not velodyne_dir.exists():
        raise FileNotFoundError(
            f"velodyne/ directory not found under {velodyne_dir.parent}. "
            f"Contents found: {sorted(p.name for p in velodyne_dir.parent.glob('*'))}"
        )
    return sorted(p.stem for p in velodyne_dir.glob("*.bin"))


def load_frame(data_root: str | Path, sequence: str, frame_id: str | int) -> Frame:
    """Load one SemanticPOSS frame: points + (if present) semantic/instance labels.

    Raises FileNotFoundError with an actionable message if the expected files
    are missing. Never silently substitutes a different frame or fabricates data.
    """
    seq_dir = _sequence_dir(data_root, sequence)
    frame_str = f"{int(frame_id):06d}" if str(frame_id).isdigit() else str(frame_id)

    bin_path = seq_dir / "velodyne" / f"{frame_str}.bin"
    if not bin_path.exists():
        found = sorted(p.name for p in (seq_dir / "velodyne").glob("*.bin"))[:5]
        raise FileNotFoundError(
            f"Point cloud file not found: {bin_path}\n"
            f"Example files that DO exist in that directory: {found}"
        )

    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(
            f"{bin_path} has {raw.size} float32 values, not divisible by 4 "
            f"(expected x,y,z,intensity per point). File may be corrupt or in a different format."
        )
    points = raw.reshape(-1, 4)
    n_points = points.shape[0]

    label_path = seq_dir / "labels" / f"{frame_str}.label"
    has_labels = label_path.exists()
    if has_labels:
        raw_labels = np.fromfile(label_path, dtype=np.uint32)
        if raw_labels.size != n_points:
            raise ValueError(
                f"Point/label count mismatch for frame {frame_str}: "
                f"{n_points} points in {bin_path.name} vs {raw_labels.size} labels in "
                f"{label_path.name}. Refusing to proceed with misaligned data."
            )
        semantic_labels = (raw_labels & 0xFFFF).astype(np.uint32)
        instance_labels = (raw_labels >> 16).astype(np.uint32)
    else:
        logger.warning(
            "No label file found at %s -- proceeding in NO-SEMANTIC-INPUT mode "
            "(geometry-only). This frame will not carry semantic information.",
            label_path,
        )
        semantic_labels = np.zeros(n_points, dtype=np.uint32)
        instance_labels = np.zeros(n_points, dtype=np.uint32)

    return Frame(
        sequence=sequence,
        frame_id=frame_str,
        points=points,
        semantic_labels=semantic_labels,
        instance_labels=instance_labels,
        has_labels=has_labels,
        bin_path=bin_path,
        label_path=label_path if has_labels else None,
    )


def frame_diagnostics(frame: Frame) -> dict:
    """Compute the diagnostic summary required before any FOVEA logic runs."""
    xyz = frame.points[:, :3]
    intensity = frame.points[:, 3]
    ranges = np.linalg.norm(xyz, axis=1)

    unique_ids, counts = np.unique(frame.semantic_labels, return_counts=True)
    class_counts = {
        int(cid): {"name": class_name(int(cid)), "count": int(cnt)}
        for cid, cnt in zip(unique_ids, counts)
    }

    diag = {
        "sequence": frame.sequence,
        "frame_id": frame.frame_id,
        "source_bin": str(frame.bin_path),
        "source_label": str(frame.label_path) if frame.label_path else None,
        "has_labels": frame.has_labels,
        "num_points": int(frame.points.shape[0]),
        "attributes": ["x", "y", "z", "intensity"],
        "x_range": [float(xyz[:, 0].min()), float(xyz[:, 0].max())],
        "y_range": [float(xyz[:, 1].min()), float(xyz[:, 1].max())],
        "z_range": [float(xyz[:, 2].min()), float(xyz[:, 2].max())],
        "intensity_range": [float(intensity.min()), float(intensity.max())],
        "range_stats_m": {
            "min": float(ranges.min()),
            "max": float(ranges.max()),
            "mean": float(ranges.mean()),
            "p99": float(np.percentile(ranges, 99)),
        },
        "num_unique_semantic_ids": int(len(unique_ids)),
        "class_counts": class_counts,
    }
    return diag


def log_frame_diagnostics(diag: dict) -> None:
    logger.info("=== Frame diagnostics: seq=%s frame=%s ===", diag["sequence"], diag["frame_id"])
    logger.info("source_bin=%s", diag["source_bin"])
    logger.info("source_label=%s", diag["source_label"])
    logger.info("num_points=%d  has_labels=%s", diag["num_points"], diag["has_labels"])
    logger.info("x_range=%s y_range=%s z_range=%s", diag["x_range"], diag["y_range"], diag["z_range"])
    logger.info("range_stats_m=%s", diag["range_stats_m"])
    logger.info("num_unique_semantic_ids=%d", diag["num_unique_semantic_ids"])
    for cid, info in sorted(diag["class_counts"].items()):
        logger.info("  class %3d (%-16s): %d points", cid, info["name"], info["count"])
