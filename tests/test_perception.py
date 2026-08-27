"""Tests for the perception layer (src/perception/).

Runs REAL inference against REAL data -- no mocking of the model or the
frames, matching this repo's "nothing fabricated" convention. Model loading
(state_dict + checkpoint) happens once per test class to keep the suite fast.

Run with: python -m unittest discover tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import load_config
from src.datasets import get_adapter
from src.perception.evaluation import DISTANCE_BANDS_M, evaluate_predictions
from src.perception.inference import run_perception
from src.perception.model import SqueezeSegModel, get_squeezeseg_model
from src.perception.ontology import (
    CANONICAL_CLASSES,
    canonical_id_for_name,
    map_names_to_canonical,
    model_predictable_canonical_classes,
)
from src.perception.preprocessing import RangeImageProjector

CFG = load_config("config.yaml")
SEQUENCE = "00"
FRAME_ID = "000000"


class TestOntologyMapping(unittest.TestCase):
    def test_semanticposs_car_maps_to_canonical_car(self):
        self.assertEqual(CANONICAL_CLASSES[canonical_id_for_name("car", "SemanticPOSS")], "car")

    def test_semantickitti_car_maps_to_canonical_car(self):
        self.assertEqual(CANONICAL_CLASSES[canonical_id_for_name("car", "semantickitti-20")], "car")

    def test_rellis_vehicle_maps_to_canonical_car(self):
        self.assertEqual(CANONICAL_CLASSES[canonical_id_for_name("vehicle", "RELLIS-3D")], "car")

    def test_unmapped_name_falls_back_to_unknown(self):
        self.assertEqual(CANONICAL_CLASSES[canonical_id_for_name("not-a-real-class", "SemanticPOSS")], "unknown")

    def test_rellis_hazard_classes_are_unreachable_by_semantickitti_model(self):
        predictable = model_predictable_canonical_classes("semantickitti-20")
        for name in ("water", "mud", "puddle", "rubble", "log", "barrier"):
            cid = canonical_id_for_name(name, "RELLIS-3D")
            self.assertEqual(CANONICAL_CLASSES[cid], "hazard")
            self.assertNotIn(cid, predictable)

    def test_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            canonical_id_for_name("car", "NotARealDataset")

    def test_map_names_to_canonical_vectorized(self):
        # "1 person" is SemanticPOSS's actual class name (see LABEL_DICT) --
        # bare "person" is not a SemanticPOSS class and correctly falls back
        # to "unknown", which this also exercises.
        ids = map_names_to_canonical(["car", "1 person", "person"], "SemanticPOSS")
        self.assertEqual([CANONICAL_CLASSES[i] for i in ids], ["car", "person", "unknown"])


class TestRangeImageProjection(unittest.TestCase):
    def setUp(self):
        self.projector = RangeImageProjector(
            height=8, width=16, fov_up_deg=3.0, fov_down_deg=-25.0,
            img_means=[0, 0, 0, 0, 0], img_stds=[1, 1, 1, 1, 1],
        )

    def test_output_shapes(self):
        pts = np.random.default_rng(0).uniform(-10, 10, size=(200, 4)).astype(np.float32)
        pts[:, 2] = np.random.default_rng(1).uniform(-1, 1, size=200)  # keep z modest so depth>0
        proj = self.projector.project(pts)
        self.assertEqual(proj.proj_tensor.shape, (5, 8, 16))
        self.assertEqual(proj.proj_idx.shape, (8, 16))
        self.assertEqual(proj.proj_mask.shape, (8, 16))
        self.assertEqual(proj.num_points, 200)

    def test_every_point_recoverable_via_unprojection_or_accounted_for(self):
        pts = np.random.default_rng(2).uniform(-10, 10, size=(500, 4)).astype(np.float32)
        proj = self.projector.project(pts)
        # unproject a trivial per-pixel "label" = 7 everywhere, then every
        # point that DID get a pixel should read back 7; every point that
        # didn't (degenerate range) should read back the fallback.
        pixel_labels = np.full((8, 16), 7, dtype=np.int64)
        out = self.projector.unproject(pixel_labels, proj, fallback_label=-1)
        self.assertEqual(out.shape[0], 500)
        got_pixel = out == 7
        got_fallback = out == -1
        self.assertTrue((got_pixel | got_fallback).all())

    def test_degenerate_zero_range_point_does_not_crash(self):
        pts = np.array([[0.0, 0.0, 0.0, 0.5]], dtype=np.float32)  # exact zero range
        proj = self.projector.project(pts)
        self.assertEqual(proj.num_points, 1)
        # must not have been silently projected onto a pixel with a bogus depth
        self.assertEqual(proj.n_invalid_points, 1)


class TestSqueezeSegModelLoading(unittest.TestCase):
    """Architecture correctness is verified BY the strict state_dict load
    itself (see src/perception/model.py's module docstring) -- if the
    reimplemented architecture didn't match the checkpoint's parameter
    names/shapes exactly, this constructor would raise."""

    def test_model_loads_with_strict_state_dict_matching(self):
        model = SqueezeSegModel()
        self.assertEqual(model.model_ontology, "semantickitti-20")
        self.assertEqual(model.nclasses, 20)
        self.assertTrue(model.provides_confidence)

    def test_singleton_reuse(self):
        m1 = get_squeezeseg_model(CFG)
        m2 = get_squeezeseg_model(CFG)
        self.assertIs(m1, m2)


class TestRealInferenceSemanticPOSS(unittest.TestCase):
    """Real inference on a real, known SemanticPOSS frame."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.model = get_squeezeseg_model(CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.result = run_perception(cls.model, cls.frame.points)

    def test_prediction_length_matches_input_exactly(self):
        self.assertEqual(len(self.result.predicted_labels), self.frame.points.shape[0])
        self.assertEqual(len(self.result.predicted_canonical), self.frame.points.shape[0])

    def test_ground_truth_never_used_as_prediction(self):
        # The model must never have seen frame.semantic_labels -- spot check
        # that predictions are NOT simply a copy of ground truth (which would
        # trivially happen if someone wired ground truth into predict() by
        # mistake): predicted ids live in SemanticKITTI's raw id space
        # (10, 30, 40, ...) which barely overlaps SemanticPOSS's own sparse
        # id set (0, 4-17, 21, 22), so an accidental passthrough would be
        # detectable by wildly different value ranges. This just asserts the
        # prediction array is a distinct object/values, not aliased.
        self.assertFalse(np.array_equal(self.result.predicted_labels, self.frame.semantic_labels))

    def test_confidence_present_and_in_unit_range(self):
        self.assertIsNotNone(self.result.confidence)
        self.assertEqual(len(self.result.confidence), self.frame.points.shape[0])
        self.assertTrue((self.result.confidence >= 0).all())
        self.assertTrue((self.result.confidence <= 1.0 + 1e-5).all())

    def test_timing_structure(self):
        r = self.result
        for field in (r.preprocessing_ms, r.inference_ms, r.postprocessing_ms, r.total_perception_ms):
            self.assertGreaterEqual(field, 0.0)
        # total should be at least the sum of the three stages minus float slop
        self.assertGreaterEqual(r.total_perception_ms + 1.0, r.preprocessing_ms + r.inference_ms + r.postprocessing_ms)
        self.assertGreater(r.points_per_second, 0)

    def test_predictions_are_not_degenerate(self):
        """Regression guard for the intensity-scale bug found during
        integration (raw [0,255] SemanticPOSS intensity fed unscaled into a
        model trained on ~[0,1] KITTI remission collapsed >78% of
        predictions onto a single class). A healthy run should span several
        classes with no single class above ~50%."""
        n = len(self.result.predicted_canonical)
        _, counts = np.unique(self.result.predicted_canonical, return_counts=True)
        self.assertGreaterEqual(len(counts), 5, "predictions collapsed onto too few classes")
        self.assertLess(counts.max() / n, 0.5, "one class dominates >50% of predictions -- likely a preprocessing bug")


class TestEvaluationSemanticPOSS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.model = get_squeezeseg_model(CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.result = run_perception(cls.model, cls.frame.points)
        gt_names = [cls.adapter.class_name(int(l)) for l in cls.frame.semantic_labels]
        cls.gt_canonical = np.asarray(map_names_to_canonical(gt_names, "SemanticPOSS"), dtype=np.int64)
        cls.range_m = np.linalg.norm(cls.frame.points[:, :3], axis=1)
        cls.ev = evaluate_predictions(cls.result.predicted_canonical, cls.gt_canonical, cls.range_m, cls.result.model_ontology)

    def test_point_accounting(self):
        self.assertEqual(self.ev.num_points_total, self.frame.points.shape[0])
        self.assertEqual(self.ev.num_points_evaluated + self.ev.num_points_ignored, self.ev.num_points_total)

    def test_accuracy_and_miou_in_valid_range(self):
        self.assertGreaterEqual(self.ev.overall_accuracy, 0.0)
        self.assertLessEqual(self.ev.overall_accuracy, 1.0)
        self.assertGreaterEqual(self.ev.mean_iou, 0.0)
        self.assertLessEqual(self.ev.mean_iou, 1.0)

    def test_measured_result_is_plausible_cross_domain_transfer(self):
        """Not a fabricated target -- a sanity bound. This checkpoint was
        trained on a different sensor/domain (KITTI Velodyne HDL-64E, urban
        driving) than SemanticPOSS (Pandora40, campus) with zero
        fine-tuning; accuracy should be well above chance (~1/12 canonical
        classes) but nowhere near a same-domain benchmark result."""
        self.assertGreater(self.ev.overall_accuracy, 0.15)
        self.assertLess(self.ev.overall_accuracy, 0.95)

    def test_per_class_iou_keys_cover_all_canonical_classes(self):
        self.assertEqual(set(self.ev.per_class_iou.keys()), set(CANONICAL_CLASSES.values()))

    def test_confusion_matrix_shape(self):
        n = len(CANONICAL_CLASSES)
        self.assertEqual(len(self.ev.confusion_matrix), n)
        self.assertTrue(all(len(row) == n for row in self.ev.confusion_matrix))
        self.assertEqual(sum(sum(row) for row in self.ev.confusion_matrix), self.ev.num_points_evaluated)

    def test_distance_bands_cover_config(self):
        self.assertEqual(set(self.ev.distance_bands.keys()), {f"{lo:g}-{hi:g}" for lo, hi in DISTANCE_BANDS_M})
        total_band_points = sum(m.point_count for m in self.ev.distance_bands.values())
        # points beyond the last band (>=100m) are legitimately excluded, so this is <=, not ==
        self.assertLessEqual(total_band_points, self.ev.num_points_evaluated)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(np.zeros(5, dtype=np.int64), np.zeros(6, dtype=np.int64), np.zeros(6), "semantickitti-20")


class TestEvaluationUnavailableWithoutGroundTruth(unittest.TestCase):
    def test_synthetic_frame_without_labels_reports_not_evaluated(self):
        # Directly exercises the "no ground truth -> honest non-evaluation"
        # path without needing a real unlabeled frame on disk.
        pred = np.array([1, 1, 2, 2], dtype=np.int64)
        gt = np.array([0, 0, 0, 0], dtype=np.int64)  # all "unknown" -> fully ignored
        ev = evaluate_predictions(pred, gt, np.array([1.0, 2.0, 3.0, 4.0]), "semantickitti-20")
        self.assertEqual(ev.num_points_evaluated, 0)
        self.assertEqual(ev.overall_accuracy, 0.0)


class TestFoveaInferenceMode(unittest.TestCase):
    """FOVEA fed PREDICTED labels instead of ground truth -- same engine
    functions, verified to produce real, distinct output."""

    @classmethod
    def setUpClass(cls):
        import copy

        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.model = get_squeezeseg_model(CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.result = run_perception(cls.model, cls.frame.points)
        cls.effective_cfg = copy.deepcopy(CFG)
        model_entry = CFG["perception"]["models"]["squeezeseg"]
        cls.effective_cfg["fovea"]["semantic_priority"] = model_entry["semantic_priority"]
        cls.effective_cfg["fovea"]["dynamic_classes"] = list(model_entry["dynamic_classes"])

    def test_fovea_map_builds_from_predictions(self):
        from src.fovea_allocator import build_fovea_map

        cells, accounting = build_fovea_map(
            self.frame.points, self.result.predicted_labels, self.effective_cfg,
            label_dict=self.model.raw_label_names,
        )
        self.assertGreater(len(cells), 0)
        self.assertEqual(accounting["method"], "fovea")
        # class names on the resulting cells must come from the MODEL's
        # ontology (semantickitti-20 names), never SemanticPOSS's.
        names = {c.majority_class_name for c in cells}
        self.assertTrue(names.issubset(set(self.model.raw_label_names.values()) | {"unknown"}))

    def test_prediction_mode_and_ground_truth_mode_produce_different_fovea_cells(self):
        from src.fovea_allocator import build_fovea_map

        pred_cells, _ = build_fovea_map(
            self.frame.points, self.result.predicted_labels, self.effective_cfg,
            label_dict=self.model.raw_label_names,
        )
        gt_cells, _ = build_fovea_map(self.frame.points, self.frame.semantic_labels, CFG)
        # Different label sources -> genuinely different information scores
        # -> not expected to produce an identical cell count (both are real,
        # neither is fabricated -- this just confirms the two modes are
        # actually wired to different data, not silently sharing one path).
        self.assertNotEqual(len(pred_cells), len(gt_cells))


if __name__ == "__main__":
    unittest.main()
