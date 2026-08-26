"""Coordinate/index helpers shared by all three mapping methods.

Spatial indexing convention (documented once, used everywhere, to avoid
alignment bugs): given an explicit origin (ox, oy) and a cell size s,
    cell_index(v) = floor((v - o) / s)
This is a standard integerized floor-division grid index. It is stable for
negative coordinates (floor, not truncation) and never produces duplicate or
misaligned cells because indices are derived once from raw floats, not
accumulated.
"""
from __future__ import annotations

import numpy as np


def radial_range(points_xyz: np.ndarray) -> np.ndarray:
    """3D Euclidean range from sensor origin -- same definition used by the
    dataset's own read_data.py (np.linalg.norm(points[:, :3], axis=1))."""
    return np.linalg.norm(points_xyz[:, :3], axis=1)


def cell_index(coord: np.ndarray, origin: float, cell_size: np.ndarray | float) -> np.ndarray:
    """Floor-division grid index. `cell_size` may be a scalar or a per-point array
    (needed for distance-adaptive / FOVEA, where cell size varies by point)."""
    return np.floor((coord - origin) / cell_size).astype(np.int64)


def cell_center(index: np.ndarray, origin: float, cell_size: np.ndarray | float) -> np.ndarray:
    return origin + (index.astype(np.float64) + 0.5) * cell_size


def assign_bands(ranges: np.ndarray, bands: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Assign each point to a distance band.

    `bands` is a list of {max_m, cell_size_m} sorted ascending by max_m, forming
    contiguous half-open intervals [0, b0.max), [b0.max, b1.max), ...
    Returns (band_id, cell_size_for_point). band_id == -1 for points beyond the
    last configured band's max_m (kept out of distance-adaptive/FOVEA maps,
    counted explicitly as out-of-envelope rather than silently dropped).
    """
    band_id = np.full(ranges.shape, -1, dtype=np.int64)
    cell_size = np.full(ranges.shape, np.nan, dtype=np.float64)
    lower = 0.0
    for i, band in enumerate(bands):
        upper = float(band["max_m"])
        mask = (ranges >= lower) & (ranges < upper) & (band_id == -1)
        band_id[mask] = i
        cell_size[mask] = float(band["cell_size_m"])
        lower = upper
    return band_id, cell_size
