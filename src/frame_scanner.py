"""Scan a small number of frames to find good demo candidates: frames with
high semantic class diversity (so a high-priority object and an ordinary
terrain region can be compared at similar distance)."""
from __future__ import annotations

import numpy as np

from src.data_loader import list_frame_ids, load_frame


def scan_candidate_frames(data_root: str, sequence: str, max_frames: int = 30) -> list[dict]:
    frame_ids = list_frame_ids(data_root, sequence)[:max_frames]
    candidates = []
    for fid in frame_ids:
        frame = load_frame(data_root, sequence, fid)
        unique_classes = np.unique(frame.semantic_labels)
        candidates.append({
            "frame_id": fid,
            "num_points": int(frame.points.shape[0]),
            "num_unique_classes": int(len(unique_classes)),
        })
    candidates.sort(key=lambda c: c["num_unique_classes"], reverse=True)
    return candidates
