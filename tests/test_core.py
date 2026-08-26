"""Basic unit tests for FOVEA v0.1's core spatial/allocation logic.

Run with: python -m pytest tests/ -v   (or python -m unittest discover tests)
"""
import unittest

import numpy as np

from src.geometry import assign_bands, cell_index, radial_range
from src.map_structure import aggregate_points_to_cells, compute_information_score
from src.semantic_labels import class_name, priority_lookup


class TestCoordinateToCellMapping(unittest.TestCase):
    def test_basic_quantization(self):
        coords = np.array([0.0, 0.049, 0.05, 0.999, -0.001, -0.05])
        idx = cell_index(coords, origin=0.0, cell_size=0.05)
        np.testing.assert_array_equal(idx, [0, 0, 1, 19, -1, -1])

    def test_negative_coordinates_floor_correctly(self):
        # floor(-0.5/0.2) = floor(-2.5) = -3, not truncated to -2
        idx = cell_index(np.array([-0.5]), origin=0.0, cell_size=0.2)
        self.assertEqual(idx[0], -3)

    def test_nonzero_origin_shifts_grid(self):
        idx_a = cell_index(np.array([5.0]), origin=0.0, cell_size=1.0)
        idx_b = cell_index(np.array([5.0]), origin=5.0, cell_size=1.0)
        self.assertNotEqual(idx_a[0], idx_b[0])


class TestDistanceBandAssignment(unittest.TestCase):
    bands = [
        {"max_m": 10, "cell_size_m": 0.05},
        {"max_m": 30, "cell_size_m": 0.10},
        {"max_m": 60, "cell_size_m": 0.25},
        {"max_m": 100, "cell_size_m": 0.50},
    ]

    def test_band_assignment_by_range(self):
        ranges = np.array([0.0, 9.99, 10.0, 29.9, 60.0, 150.0])
        band_id, cell_size = assign_bands(ranges, self.bands)
        np.testing.assert_array_equal(band_id, [0, 0, 1, 1, 3, -1])
        self.assertTrue(np.isnan(cell_size[-1]))  # beyond envelope -> no cell size

    def test_cell_size_monotonic_non_decreasing_with_distance(self):
        sizes = [b["cell_size_m"] for b in self.bands]
        self.assertTrue(all(sizes[i] <= sizes[i + 1] for i in range(len(sizes) - 1)))


class TestSemanticPriorityLookup(unittest.TestCase):
    def test_known_class(self):
        table = {"car": 0.9, "ground": 0.05}
        self.assertEqual(priority_lookup(7, table), 0.9)  # id 7 == "car"

    def test_unknown_id_falls_back_to_default(self):
        table = {"car": 0.9}
        self.assertEqual(class_name(999), "unknown")
        self.assertEqual(priority_lookup(999, table, default=0.1), 0.1)


class TestInformationScore(unittest.TestCase):
    def test_higher_priority_class_yields_higher_score_all_else_equal(self):
        table = {"car": 0.9, "ground": 0.05}
        weights = {"semantic": 1.0, "geometry": 0.0, "dynamic": 0.0, "uncertainty": 0.0}
        _, _, _, score_car = compute_information_score(7, 0.0, table, set(), 0.5, weights)
        _, _, _, score_ground = compute_information_score(22, 0.0, table, set(), 0.5, weights)
        self.assertGreater(score_car, score_ground)

    def test_uncertainty_excluded_when_weight_zero(self):
        table = {"car": 0.9}
        weights = {"semantic": 1.0, "geometry": 0.0, "dynamic": 0.0, "uncertainty": 0.0}
        _, _, _, score = compute_information_score(7, 0.0, table, set(), 0.5, weights)
        self.assertAlmostEqual(score, 0.9)


class TestCellInsertionAndPointConservation(unittest.TestCase):
    def _make_points(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        xy = rng.uniform(-5, 5, size=(n, 2))
        z = rng.uniform(0, 2, size=(n, 1))
        intensity = rng.uniform(0, 1, size=(n, 1))
        return np.hstack([xy, z, intensity]).astype(np.float32)

    def test_no_point_loss_for_valid_points(self):
        points = self._make_points()
        labels = np.zeros(points.shape[0], dtype=np.uint32)
        ix = cell_index(points[:, 0], 0.0, 0.5)
        iy = cell_index(points[:, 1], 0.0, 0.5)
        cell_size = np.full(points.shape[0], 0.5)
        res = np.zeros(points.shape[0], dtype=np.int64)
        band = np.zeros(points.shape[0], dtype=np.int64)
        weights = {"semantic": 0.5, "geometry": 0.5, "dynamic": 0.0, "uncertainty": 0.0}

        cells, accounting = aggregate_points_to_cells(
            points, labels, ix, iy, cell_size, res, band,
            origin=(0.0, 0.0), priority_table={}, dynamic_classes=set(),
            geometry_max_std=0.5, weights=weights,
        )
        self.assertEqual(accounting["input_points"], points.shape[0])
        self.assertEqual(accounting["assigned_points"], points.shape[0])
        self.assertEqual(sum(c.point_count for c in cells), points.shape[0])

    def test_nan_points_are_rejected_not_silently_dropped(self):
        points = self._make_points(n=10)
        points[0, 0] = np.nan
        labels = np.zeros(points.shape[0], dtype=np.uint32)
        ix = cell_index(points[:, 0], 0.0, 0.5)
        iy = cell_index(points[:, 1], 0.0, 0.5)
        cell_size = np.full(points.shape[0], 0.5)
        res = np.zeros(points.shape[0], dtype=np.int64)
        band = np.zeros(points.shape[0], dtype=np.int64)
        weights = {"semantic": 1.0, "geometry": 0.0, "dynamic": 0.0, "uncertainty": 0.0}

        cells, accounting = aggregate_points_to_cells(
            points, labels, ix, iy, cell_size, res, band,
            origin=(0.0, 0.0), priority_table={}, dynamic_classes=set(),
            geometry_max_std=0.5, weights=weights,
        )
        self.assertEqual(accounting["rejected_points"], 1)
        self.assertEqual(accounting["assigned_points"], 9)


class TestDeterministicResolutionSelection(unittest.TestCase):
    def test_same_input_same_output(self):
        """Aggregation must be deterministic: running twice on identical input
        yields identical cell sets (order-independent comparison)."""
        rng = np.random.default_rng(42)
        n = 500
        points = np.hstack([
            rng.uniform(-10, 10, size=(n, 2)),
            rng.uniform(0, 3, size=(n, 1)),
            rng.uniform(0, 1, size=(n, 1)),
        ]).astype(np.float32)
        labels = rng.integers(0, 5, size=n).astype(np.uint32)
        ix = cell_index(points[:, 0], 0.0, 0.5)
        iy = cell_index(points[:, 1], 0.0, 0.5)
        cell_size = np.full(n, 0.5)
        res = np.zeros(n, dtype=np.int64)
        band = np.zeros(n, dtype=np.int64)
        weights = {"semantic": 0.5, "geometry": 0.5, "dynamic": 0.0, "uncertainty": 0.0}
        args = (points, labels, ix, iy, cell_size, res, band)
        kwargs = dict(origin=(0.0, 0.0), priority_table={}, dynamic_classes=set(),
                      geometry_max_std=0.5, weights=weights)

        cells_a, _ = aggregate_points_to_cells(*args, **kwargs)
        cells_b, _ = aggregate_points_to_cells(*args, **kwargs)

        keys_a = sorted((c.ix, c.iy, c.point_count) for c in cells_a)
        keys_b = sorted((c.ix, c.iy, c.point_count) for c in cells_b)
        self.assertEqual(keys_a, keys_b)


if __name__ == "__main__":
    unittest.main()
