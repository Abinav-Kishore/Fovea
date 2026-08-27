"""Tests for the explicit 2.5D perception representation (src/spatial/).

Runs against REAL data (SemanticPOSS always; RELLIS-3D skipped if not
present) and REAL Phase 1 inference -- no mocking, matching this repo's
"nothing fabricated" convention.

Run with: python -m unittest discover tests -v
"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.datasets import get_adapter
from src.distance_baseline import build_distance_adaptive_map
from src.fovea_allocator import build_fovea_map
from src.map_structure import Cell
from src.spatial import (
    ObstacleClass,
    TerrainClass,
    available_terrain_sources,
    build_2_5d_map,
    classify_obstacle,
    classify_terrain,
    validate_cell_geometry,
    validate_no_duplicate_cells,
    validate_point_conservation,
)
from src.uniform_baseline import build_uniform_map

CFG = load_config("config.yaml")
SEQUENCE = "00"
FRAME_ID = "000000"
_RELLIS_PRESENT = Path(CFG["datasets"]["RELLIS-3D"]["pointcloud_root"]).exists()


class TestTerrainClassification(unittest.TestCase):
    def test_semanticposs_ground_is_drivable(self):
        self.assertEqual(classify_terrain("ground", "SemanticPOSS"), TerrainClass.DRIVABLE)

    def test_semanticposs_car_is_obstacle(self):
        self.assertEqual(classify_terrain("car", "SemanticPOSS"), TerrainClass.OBSTACLE)

    def test_semanticposs_plants_is_non_drivable(self):
        self.assertEqual(classify_terrain("plants", "SemanticPOSS"), TerrainClass.NON_DRIVABLE)

    def test_rellis_grass_is_drivable_offroad(self):
        self.assertEqual(classify_terrain("grass", "RELLIS-3D"), TerrainClass.DRIVABLE)

    def test_rellis_water_is_non_drivable_not_obstacle(self):
        # Hazardous terrain, but not a solid object to collide with.
        self.assertEqual(classify_terrain("water", "RELLIS-3D"), TerrainClass.NON_DRIVABLE)

    def test_semantickitti_sidewalk_is_non_drivable(self):
        self.assertEqual(classify_terrain("sidewalk", "semantickitti-20"), TerrainClass.NON_DRIVABLE)

    def test_ambiguous_class_is_unknown_not_guessed(self):
        self.assertEqual(classify_terrain("other-ground", "semantickitti-20"), TerrainClass.UNKNOWN)

    def test_unregistered_source_is_unknown(self):
        self.assertEqual(classify_terrain("car", "NotARealSource"), TerrainClass.UNKNOWN)

    def test_unregistered_class_name_is_unknown(self):
        self.assertEqual(classify_terrain("not-a-real-class", "SemanticPOSS"), TerrainClass.UNKNOWN)

    def test_all_three_sources_registered(self):
        self.assertEqual(set(available_terrain_sources()), {"SemanticPOSS", "RELLIS-3D", "semantickitti-20"})

    def test_every_semanticposs_class_covered(self):
        from src.semantic_labels import LABEL_DICT

        for name in LABEL_DICT.values():
            # every real class name must have an explicit entry -- no silent
            # fallback for a class we know exists in this dataset
            self.assertIn(name, __import__("src.spatial.terrain", fromlist=["_SEMANTICPOSS_TERRAIN"])._SEMANTICPOSS_TERRAIN)

    def test_every_rellis_class_covered(self):
        from src.datasets.rellis_labels import LABEL_DICT

        table = __import__("src.spatial.terrain", fromlist=["_RELLIS_TERRAIN"])._RELLIS_TERRAIN
        for name in LABEL_DICT.values():
            self.assertIn(name, table)


class TestObstacleClassification(unittest.TestCase):
    def test_non_obstacle_terrain_yields_non_obstacle(self):
        self.assertEqual(classify_obstacle(TerrainClass.DRIVABLE, 0.0), ObstacleClass.NON_OBSTACLE)
        self.assertEqual(classify_obstacle(TerrainClass.NON_DRIVABLE, 1.0), ObstacleClass.NON_OBSTACLE)

    def test_unknown_terrain_yields_unknown_obstacle(self):
        self.assertEqual(classify_obstacle(TerrainClass.UNKNOWN, 0.0), ObstacleClass.UNKNOWN)
        self.assertEqual(classify_obstacle(TerrainClass.UNKNOWN, 1.0), ObstacleClass.UNKNOWN)

    def test_obstacle_terrain_with_low_dynamic_score_is_static(self):
        self.assertEqual(classify_obstacle(TerrainClass.OBSTACLE, 0.0), ObstacleClass.STATIC_OBSTACLE)

    def test_obstacle_terrain_with_high_dynamic_score_is_potential_dynamic(self):
        self.assertEqual(classify_obstacle(TerrainClass.OBSTACLE, 1.0), ObstacleClass.POTENTIAL_DYNAMIC_OBSTACLE)


class TestBuild2_5DMapSemanticPOSS(unittest.TestCase):
    """Real SemanticPOSS frame, ground-truth mode."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.cells, cls.accounting = build_fovea_map(cls.frame.points, cls.frame.semantic_labels, CFG)
        cls.pcells, cls.stats = build_2_5d_map(cls.cells, cls.accounting, "ground_truth", "SemanticPOSS")

    def test_cell_count_matches_source_cells(self):
        self.assertEqual(len(self.pcells), len(self.cells))
        self.assertEqual(self.stats.total_cells, len(self.cells))
        self.assertEqual(self.stats.occupied_cells, len(self.cells))

    def test_cell_ids_are_unique_and_sequential(self):
        ids = [c.cell_id for c in self.pcells]
        self.assertEqual(ids, list(range(len(self.pcells))))

    def test_spatial_bounds_derived_correctly(self):
        for c in self.pcells[:200]:
            self.assertAlmostEqual(c.x_max - c.x_min, c.cell_size_m, places=6)
            self.assertAlmostEqual(c.y_max - c.y_min, c.cell_size_m, places=6)
            self.assertAlmostEqual((c.x_min + c.x_max) / 2, c.center_x, places=6)
            self.assertAlmostEqual((c.y_min + c.y_max) / 2, c.center_y, places=6)

    def test_elevation_statistics_are_real_and_ordered(self):
        for c in self.pcells:
            self.assertTrue(c.has_points)  # every Cell FOVEA produces has >=1 point, by construction
            self.assertIsNotNone(c.elevation_median)
            self.assertLessEqual(c.elevation_min, c.elevation_median)
            self.assertLessEqual(c.elevation_median, c.elevation_max)
            self.assertGreaterEqual(c.elevation_std, 0.0)

    def test_map_level_elevation_stats_are_real(self):
        self.assertIsNotNone(self.stats.mean_elevation_m)
        self.assertLessEqual(self.stats.elevation_min_m, self.stats.mean_elevation_m)
        self.assertLessEqual(self.stats.mean_elevation_m, self.stats.elevation_max_m)
        # cross-check against the raw point cloud's own z range -- the map's
        # elevation range must be a real subset of the actual data's range.
        z = self.frame.points[:, 2]
        self.assertGreaterEqual(self.stats.elevation_min_m, float(z.min()) - 1e-6)
        self.assertLessEqual(self.stats.elevation_max_m, float(z.max()) + 1e-6)

    def test_semantic_class_comes_from_ground_truth_labels(self):
        # Every PerceptionCell's semantic_class must be a real SemanticPOSS
        # class name, never a SemanticKITTI/RELLIS name leaking in.
        from src.semantic_labels import LABEL_DICT

        valid_names = set(LABEL_DICT.values()) | {"unknown"}
        names = {c.semantic_class for c in self.pcells}
        self.assertTrue(names.issubset(valid_names))

    def test_terrain_and_obstacle_counts_sum_correctly(self):
        s = self.stats
        self.assertEqual(s.drivable_cells + s.non_drivable_cells + s.obstacle_cells + s.unknown_terrain_cells, s.total_cells)
        self.assertEqual(s.static_obstacle_cells + s.potential_dynamic_obstacle_cells, s.obstacle_cells)

    def test_label_source_and_provenance_recorded(self):
        for c in self.pcells:
            self.assertEqual(c.label_source, "ground_truth")
            self.assertEqual(c.source, "SemanticPOSS")
        self.assertEqual(self.stats.label_source, "ground_truth")
        self.assertEqual(self.stats.source, "SemanticPOSS")

    def test_point_conservation(self):
        result = validate_point_conservation(self.accounting)
        self.assertTrue(result["conserved"])
        self.assertEqual(result["input_points"], self.frame.points.shape[0])

    def test_no_duplicate_cell_assignment(self):
        result = validate_no_duplicate_cells(self.cells)
        self.assertTrue(result["no_duplicates"])
        self.assertEqual(result["cell_count"], result["unique_cell_count"])

    def test_geometry_valid(self):
        result = validate_cell_geometry(self.pcells)
        self.assertTrue(result["valid"])
        self.assertEqual(result["bad_bounds_cell_ids"], [])
        self.assertEqual(result["bad_elevation_cell_ids"], [])
        self.assertEqual(result["bad_resolution_cell_ids"], [])

    def test_representation_coverage_matches_assignment_rate(self):
        expected = self.accounting["assigned_points"] / self.accounting["input_points"]
        self.assertAlmostEqual(self.stats.representation_coverage, expected, places=6)


class TestBuild2_5DMapUniformAndDistanceAdaptive(unittest.TestCase):
    """Confirms build_2_5d_map works identically for the other two methods
    -- it's method-agnostic, since it only consumes Cell/accounting."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)

    def test_uniform_method(self):
        cells, accounting = build_uniform_map(self.frame.points, self.frame.semantic_labels, CFG)
        pcells, stats = build_2_5d_map(cells, accounting, "ground_truth", "SemanticPOSS")
        self.assertEqual(len(pcells), len(cells))
        self.assertTrue(all(c.resolution_level == 0 for c in pcells))  # single level, per uniform baseline's design

    def test_distance_adaptive_method(self):
        cells, accounting = build_distance_adaptive_map(self.frame.points, self.frame.semantic_labels, CFG)
        pcells, stats = build_2_5d_map(cells, accounting, "ground_truth", "SemanticPOSS")
        self.assertEqual(len(pcells), len(cells))
        self.assertTrue(validate_point_conservation(accounting)["conserved"])


class TestPredictionVsGroundTruthSeparation(unittest.TestCase):
    """Real Phase 1 inference -- confirms the 2.5D layer keeps prediction
    and ground-truth modes distinct, never silently mixing them."""

    @classmethod
    def setUpClass(cls):
        from src.perception.inference import run_perception
        from src.perception.model import get_squeezeseg_model

        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.model = get_squeezeseg_model(CFG)
        cls.result = run_perception(cls.model, cls.frame.points)

        cls.model_cfg = copy.deepcopy(CFG)
        model_entry = CFG["perception"]["models"]["squeezeseg"]
        cls.model_cfg["fovea"]["semantic_priority"] = model_entry["semantic_priority"]
        cls.model_cfg["fovea"]["dynamic_classes"] = list(model_entry["dynamic_classes"])

        cls.gt_cells, cls.gt_accounting = build_fovea_map(cls.frame.points, cls.frame.semantic_labels, CFG)
        cls.pred_cells, cls.pred_accounting = build_fovea_map(
            cls.frame.points, cls.result.predicted_labels, cls.model_cfg, label_dict=cls.model.raw_label_names
        )

    def test_ground_truth_mode_labeled_correctly(self):
        pcells, stats = build_2_5d_map(self.gt_cells, self.gt_accounting, "ground_truth", "SemanticPOSS")
        self.assertTrue(all(c.label_source == "ground_truth" for c in pcells))
        self.assertTrue(all(c.source == "SemanticPOSS" for c in pcells))

    def test_prediction_mode_labeled_correctly(self):
        pcells, stats = build_2_5d_map(self.pred_cells, self.pred_accounting, "prediction", "semantickitti-20")
        self.assertTrue(all(c.label_source == "prediction" for c in pcells))
        self.assertTrue(all(c.source == "semantickitti-20" for c in pcells))
        # class names must be SemanticKITTI's, never SemanticPOSS's
        self.assertNotIn("1 person", {c.semantic_class for c in pcells})

    def test_modes_produce_different_terrain_distributions(self):
        gt_pcells, gt_stats = build_2_5d_map(self.gt_cells, self.gt_accounting, "ground_truth", "SemanticPOSS")
        pred_pcells, pred_stats = build_2_5d_map(self.pred_cells, self.pred_accounting, "prediction", "semantickitti-20")
        # Real, distinct label sources -> not expected to produce identical
        # terrain counts (both are real, neither is fabricated).
        self.assertNotEqual(gt_stats.obstacle_cells, pred_stats.obstacle_cells)


@unittest.skipUnless(_RELLIS_PRESENT, "RELLIS-3D data not present on this machine")
class TestBuild2_5DMapRellis(unittest.TestCase):
    """Real RELLIS-3D frames, ground-truth mode."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("RELLIS-3D", CFG)
        cls.cfg = copy.deepcopy(CFG)
        cls.cfg["fovea"]["semantic_priority"] = cls.adapter.semantic_priority_table()
        cls.cfg["fovea"]["dynamic_classes"] = list(cls.adapter.dynamic_classes())

    def test_three_real_rellis_frames(self):
        for seq, fid in [("00000", "000100"), ("00002", "000500"), ("00004", "000050")]:
            frame = self.adapter.load_frame(seq, fid)
            cells, accounting = build_fovea_map(
                frame.points, frame.semantic_labels, self.cfg, label_dict=self.adapter.label_dict()
            )
            pcells, stats = build_2_5d_map(cells, accounting, "ground_truth", "RELLIS-3D")

            self.assertEqual(len(pcells), len(cells))
            self.assertTrue(validate_point_conservation(accounting)["conserved"])
            self.assertTrue(validate_no_duplicate_cells(cells)["no_duplicates"])
            self.assertTrue(validate_cell_geometry(pcells)["valid"])
            self.assertGreater(stats.total_cells, 0)
            # RELLIS's off-road ontology should surface real drivable terrain
            # (grass/dirt/asphalt/concrete) -- not everything UNKNOWN.
            self.assertGreater(stats.drivable_cells, 0)


class TestMap2_5DApiResponse(unittest.TestCase):
    """Real FastAPI app, no mocking."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from src.api import app

        cls.client = TestClient(app)

    def test_ground_truth_2_5d_endpoint(self):
        r = self.client.get(f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{FRAME_ID}/map/fovea/2_5d")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["label_source"], "ground_truth")
        self.assertEqual(body["source"], "SemanticPOSS")
        self.assertEqual(len(body["cells"]), body["stats"]["total_cells"])
        self.assertTrue(body["point_conservation"]["conserved"])
        self.assertTrue(body["no_duplicate_cells"]["no_duplicates"])
        self.assertTrue(body["geometry_valid"])

    def test_prediction_2_5d_endpoint(self):
        r = self.client.get(
            f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{FRAME_ID}/map/fovea/2_5d?label_source=prediction"
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["label_source"], "prediction")
        self.assertEqual(body["source"], "semantickitti-20")

    def test_existing_ground_truth_map_endpoint_unaffected(self):
        r = self.client.get(f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{FRAME_ID}/map/fovea")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["summary"]["output_cell_count"], 41868)

    def test_include_cells_false_omits_cells_but_keeps_stats(self):
        r = self.client.get(
            f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{FRAME_ID}/map/fovea/2_5d?include_cells=false"
        )
        body = r.json()
        self.assertEqual(body["cells"], [])
        self.assertGreater(body["stats"]["total_cells"], 0)

    def test_invalid_label_source_rejected(self):
        r = self.client.get(
            f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{FRAME_ID}/map/fovea/2_5d?label_source=bogus"
        )
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
