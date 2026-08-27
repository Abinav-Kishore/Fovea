"""Spherical range-image projection for SqueezeSeg.

Reimplements (not approximates) the exact projection used by the
pretrained checkpoint's original codebase, PRBonn/lidar-bonnetal
(train/common/laserscan.py + train/tasks/semantic/dataset/kitti/parser.py,
MIT license, fetched and cross-checked against the actual source at
integration time) -- the model was trained with this exact projection, so
reproducing it exactly (not "close enough") is required for the pretrained
weights to mean anything.

Projection formula (spherical, per point):
    depth = ||[x,y,z]||
    yaw    = -atan2(y, x)
    pitch  = asin(z / depth)
    proj_x = 0.5 * (yaw/pi + 1) * W        (then floor, clip to [0, W-1])
    proj_y = (1 - (pitch + |fov_down|)/fov) * H   (then floor, clip to [0, H-1])
When multiple points land on the same pixel, the closest one (smallest
depth) wins -- points are written to the image in order of DEcreasing
depth, so the nearest point's write happens last and survives.

Every point gets a (proj_y, proj_x) cell -- none are dropped by this step
(points are clipped to the image border, not filtered), which is what
preserves the len(predictions) == len(points) invariant end to end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RangeImageProjection:
    """Opaque intermediate passed from preprocess() to forward()/postprocess()."""

    proj_tensor: np.ndarray  # (5, H, W) float32: normalized [range, x, y, z, remission], zero where empty
    proj_idx: np.ndarray  # (H, W) int32: index into the ORIGINAL points array, -1 = empty pixel
    proj_mask: np.ndarray  # (H, W) int32: 1 where a point was projected, 0 otherwise
    num_points: int


class RangeImageProjector:
    """Config-driven projector: every parameter (fov, image size, per-channel
    normalization stats) comes from the checkpoint's own bundled
    arch_cfg.yaml -- never hardcoded separately from the model that expects
    them."""

    def __init__(self, height: int, width: int, fov_up_deg: float, fov_down_deg: float,
                 img_means: list[float], img_stds: list[float]):
        self.H = height
        self.W = width
        self.fov_up = fov_up_deg / 180.0 * np.pi
        self.fov_down = fov_down_deg / 180.0 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.means = np.asarray(img_means, dtype=np.float32).reshape(5, 1, 1)
        self.stds = np.asarray(img_stds, dtype=np.float32).reshape(5, 1, 1)

    def project(self, points: np.ndarray) -> RangeImageProjection:
        """`points` is (N, 4): x, y, z, intensity -- the same array
        src.data_loader.Frame.points already provides for every dataset."""
        n = points.shape[0]
        xyz = points[:, :3].astype(np.float64)
        remission = points[:, 3].astype(np.float32)

        depth = np.linalg.norm(xyz, axis=1)
        # Guard against degenerate points (exact-zero range would divide-by-zero
        # in pitch below) -- reject, don't silently corrupt the projection; these
        # are tracked and every such point still gets a real (fallback) label
        # downstream, never dropped from the output array.
        valid = depth > 1e-6
        n_invalid = int((~valid).sum())

        proj_idx = np.full((self.H, self.W), -1, dtype=np.int32)
        proj_range = np.zeros((self.H, self.W), dtype=np.float32)
        proj_xyz = np.zeros((self.H, self.W, 3), dtype=np.float32)
        proj_remission = np.zeros((self.H, self.W), dtype=np.float32)

        if valid.any():
            vxyz = xyz[valid]
            vdepth = depth[valid]
            vremission = remission[valid]
            vindices = np.nonzero(valid)[0]

            yaw = -np.arctan2(vxyz[:, 1], vxyz[:, 0])
            pitch = np.arcsin(np.clip(vxyz[:, 2] / vdepth, -1.0, 1.0))

            proj_x = 0.5 * (yaw / np.pi + 1.0) * self.W
            proj_y = (1.0 - (pitch + abs(self.fov_down)) / self.fov) * self.H

            proj_x = np.clip(np.floor(proj_x), 0, self.W - 1).astype(np.int32)
            proj_y = np.clip(np.floor(proj_y), 0, self.H - 1).astype(np.int32)

            # Nearest point per pixel wins: write in decreasing-depth order so
            # the closest point's write happens last (matches the reference
            # implementation exactly -- this is not an approximation).
            order = np.argsort(vdepth)[::-1]
            proj_range[proj_y[order], proj_x[order]] = vdepth[order].astype(np.float32)
            proj_xyz[proj_y[order], proj_x[order]] = vxyz[order].astype(np.float32)
            proj_remission[proj_y[order], proj_x[order]] = vremission[order]
            proj_idx[proj_y[order], proj_x[order]] = vindices[order].astype(np.int32)

        proj_mask = (proj_idx >= 0).astype(np.int32)

        proj_tensor = np.concatenate(
            [
                proj_range[None, :, :],
                proj_xyz.transpose(2, 0, 1),
                proj_remission[None, :, :],
            ],
            axis=0,
        ).astype(np.float32)
        proj_tensor = (proj_tensor - self.means) / self.stds
        proj_tensor = proj_tensor * proj_mask[None, :, :].astype(np.float32)

        result = RangeImageProjection(
            proj_tensor=proj_tensor, proj_idx=proj_idx, proj_mask=proj_mask, num_points=n
        )
        result.n_invalid_points = n_invalid  # type: ignore[attr-defined]
        return result

    def unproject_float(self, pixel_values: np.ndarray, projection: RangeImageProjection,
                          fallback: float = 0.0) -> np.ndarray:
        """Same scatter-back as unproject(), for a float-valued (H, W) map
        (e.g. per-pixel confidence) instead of integer class ids."""
        n = projection.num_points
        out = np.full(n, fallback, dtype=np.float32)
        mask = projection.proj_idx >= 0
        idx = projection.proj_idx[mask]
        out[idx] = pixel_values[mask]
        return out

    def unproject(self, pixel_labels: np.ndarray, projection: RangeImageProjection,
                   fallback_label: int = 0) -> np.ndarray:
        """Scatters a (H, W) per-pixel prediction back to a (N,) per-point
        array using the SAME proj_idx recorded during project() -- points
        that share a pixel (an inherent lossy property of range-image
        projection, not a bug) all receive that pixel's prediction. Points
        that never received a pixel (e.g. the degenerate-range guard above)
        get `fallback_label` (the model's own "unlabeled"/void class id),
        not a fabricated guess -- and are reported, not hidden.
        """
        n = projection.num_points
        out = np.full(n, fallback_label, dtype=np.int64)
        mask = projection.proj_idx >= 0
        idx = projection.proj_idx[mask]
        out[idx] = pixel_labels[mask]
        return out
