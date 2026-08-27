"""RELLIS-3D frame loader.

Verified format (inspected directly from real files in this repo, not
assumed -- see the numbers below, reproduced from an actual run):

  Rellis-3D/<seq>/vel_cloud_node_kitti_bin/<frame>.bin
    float32, reshape(-1,4) = x,y,z,intensity  (same layout as SemanticPOSS's
    .bin, despite the different directory name/dataset)

  Rellis-3D 2/<seq>/vel_cloud_node_semantickitti_label_id/<frame>.label
    uint32; semantic = value & 0xFFFF, instance = value >> 16 (SemanticKITTI
    convention, matching the directory name)

Checked across one frame from each of the 5 sequences (00000-00004):
point count == label count in every case (e.g. seq 00000 frame 001372:
34,745 points == 34,745 labels; byte sizes matched exactly: 555,920 bytes /
4 = 138,980 float32 values = 34,745 points x 4 columns).

RELLIS ships as two SEPARATE directory trees (point clouds and annotations,
downloaded/extracted independently) -- this module reads both in place and
never copies or merges them. See src/datasets/rellis_labels.py for the
class-id ontology (a completely different table from SemanticPOSS's).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.data_loader import Frame
from src.datasets.rellis_labels import class_name

logger = logging.getLogger("fovea.datasets.rellis")


def _bin_dir(rcfg: dict, sequence: str) -> Path:
    return Path(rcfg["pointcloud_root"]) / sequence / rcfg["pointcloud_subdir"]


def _label_dir(rcfg: dict, sequence: str) -> Path:
    return Path(rcfg["label_root"]) / sequence / rcfg["label_subdir"]


def list_sequences(rcfg: dict) -> list[str]:
    """RELLIS's sequence set is fixed/documented in config.yaml (the on-disk
    layout doesn't self-describe available sequences the way SemanticPOSS's
    sequences/ directory does), filtered to ones that actually exist on disk."""
    root = Path(rcfg["pointcloud_root"])
    return [s for s in rcfg["sequences"] if (root / s).exists()]


def list_frame_ids(rcfg: dict, sequence: str) -> list[str]:
    """Only frame ids where BOTH a .bin and a matching .label file exist --
    a frame with a point cloud but no label is never silently shown."""
    bin_dir = _bin_dir(rcfg, sequence)
    label_dir = _label_dir(rcfg, sequence)
    if not bin_dir.exists():
        raise FileNotFoundError(
            f"RELLIS point cloud directory not found: {bin_dir}\n"
            f"Expected pointcloud_root/<sequence>/{rcfg['pointcloud_subdir']}/*.bin"
        )
    bin_stems = {p.stem for p in bin_dir.glob("*.bin")}
    label_stems = {p.stem for p in label_dir.glob("*.label")} if label_dir.exists() else set()
    usable = sorted(bin_stems & label_stems)
    missing_labels = bin_stems - label_stems
    if missing_labels:
        logger.warning(
            "RELLIS sequence %s: %d frame(s) have a .bin file but no matching .label file "
            "-- excluded from the frame list, not silently shown as labeled.",
            sequence, len(missing_labels),
        )
    return usable


def load_frame(rcfg: dict, sequence: str, frame_id: str) -> Frame:
    """Load one RELLIS-3D frame into the SAME Frame representation
    SemanticPOSS uses (src.data_loader.Frame) -- this is the common
    representation the FOVEA engine expects; no dataset-specific fields
    are added or required downstream."""
    bin_dir = _bin_dir(rcfg, sequence)
    label_dir = _label_dir(rcfg, sequence)
    frame_str = f"{int(frame_id):06d}" if str(frame_id).isdigit() else str(frame_id)

    bin_path = bin_dir / f"{frame_str}.bin"
    if not bin_path.exists():
        found = sorted(p.name for p in bin_dir.glob("*.bin"))[:5] if bin_dir.exists() else []
        raise FileNotFoundError(
            f"RELLIS point cloud file not found: {bin_path}\n"
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

    label_path = label_dir / f"{frame_str}.label"
    has_labels = label_path.exists()
    if has_labels:
        raw_labels = np.fromfile(label_path, dtype=np.uint32)
        if raw_labels.size != n_points:
            raise ValueError(
                f"Point/label count mismatch for RELLIS frame {frame_str}: "
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
    """Same diagnostic summary as data_loader.frame_diagnostics, but resolves
    class names via RELLIS's own ontology (src.datasets.rellis_labels) instead
    of SemanticPOSS's. Kept as a small parallel function (not a shared/
    parameterized data_loader.frame_diagnostics) so data_loader.py stays
    completely untouched -- see module docstring."""
    xyz = frame.points[:, :3]
    intensity = frame.points[:, 3]
    ranges = np.linalg.norm(xyz, axis=1)

    unique_ids, counts = np.unique(frame.semantic_labels, return_counts=True)
    class_counts = {
        int(cid): {"name": class_name(int(cid)), "count": int(cnt)}
        for cid, cnt in zip(unique_ids, counts)
    }

    return {
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
