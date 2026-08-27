"""Focused tests for the SIH compliance benchmarking layer
(src/benchmarking/sih_compliance.py) -- Phase 4C.

Real SemanticPOSS data always; RELLIS-3D skipped if not present on disk,
same convention as every other dataset-aware test file in this repo. No
mocking -- every number asserted on is produced by an actual, unmodified
engine function.

Run with: python -m pytest tests/test_sih_compliance.py -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarking.sih_compliance import (
    DEFAULT_RELLIS_FRAMES,
    DEFAULT_SEMANTICPOSS_FRAMES,
    SIH_STATED_ANCHORS,
    coverage_and_conservation_report,
    effective_cfg_for,
    memory_benchmark_report,
    perception_distance_bands_report,
    pipeline_latency_report,
    resolution_vs_distance_report,
    run_sih_compliance_benchmark,
    terrain_obstacle_report,
)
from src.config import load_config
from src.datasets import get_adapter
from src.fovea_allocator import build_fovea_map

CFG = load_config("config.yaml")
SEQUENCE, FRAME_ID = "00", "000000"
_RELLIS_PRESENT = Path(CFG["datasets"]["RELLIS-3D"]["pointcloud_root"]).exists()


class TestResolutionVsDistanceReport(unittest.TestCase):
    """Requirement A."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.cells, cls.accounting = build_fovea_map(cls.frame.points, cls.frame.semantic_labels, CFG)
        cls.bands_cfg = CFG["distance_adaptive"]["bands"]
        cls.report = resolution_vs_distance_report(cls.cells, cls.bands_cfg)

    def test_four_bands_reported(self):
        self.assertEqual(len(self.report["bands"]), 4)
        self.assertEqual([b["band"] for b in self.report["bands"]], ["0-10m", "10-30m", "30-60m", "60-100m"])

    def test_sih_anchors_match_actual_config_at_the_two_stated_points(self):
        # The SIH brief's own two anchors -- confirm the EXISTING config
        # actually has these values at the near/far band edges (not assumed).
        self.assertAlmostEqual(self.report["bands"][0]["configured_band_base_cell_size_m"], SIH_STATED_ANCHORS["near_10m_target_cell_size_m"])
        self.assertAlmostEqual(self.report["bands"][3]["configured_band_base_cell_size_m"], SIH_STATED_ANCHORS["by_100m_target_cell_size_m"])

    def test_cell_size_bounds_hold_within_each_populated_band(self):
        for b in self.report["bands"]:
            if b["cell_count"] == 0:
                continue
            self.assertLessEqual(b["min_cell_size_m"], b["median_cell_size_m"])
            self.assertLessEqual(b["median_cell_size_m"], b["max_cell_size_m"])
            # never coarser than the band's own configured base (FOVEA's hard envelope)
            self.assertLessEqual(b["max_cell_size_m"], b["configured_band_base_cell_size_m"] + 1e-9)

    def test_band_cell_counts_sum_to_total_cells(self):
        self.assertEqual(sum(b["cell_count"] for b in self.report["bands"]), len(self.cells))


class TestCoverageAndConservationReport(unittest.TestCase):
    """Requirement B."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.cells, cls.accounting = build_fovea_map(cls.frame.points, cls.frame.semantic_labels, CFG)
        cls.report = coverage_and_conservation_report(
            cls.cells, cls.accounting, cls.frame.points, CFG["distance_adaptive"]["bands"]
        )

    def test_point_conservation_holds(self):
        self.assertTrue(self.report["point_conservation"]["conserved"])

    def test_input_points_matches_real_frame(self):
        self.assertEqual(self.report["input_points"], self.frame.points.shape[0])

    def test_accounted_points_sum_to_input(self):
        accounted = (
            self.report["assigned_points"] + self.report["out_of_envelope_points"] + self.report["invalid_points"]
        )
        self.assertEqual(accounted, self.report["input_points"])

    def test_four_bands_with_real_point_counts(self):
        self.assertEqual(len(self.report["bands"]), 4)
        self.assertEqual(sum(b["point_count"] for b in self.report["bands"]), self.report["assigned_points"])


class TestMemoryBenchmarkReport(unittest.TestCase):
    """Requirement C."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.report = memory_benchmark_report(cls.frame.points, cls.frame.semantic_labels, CFG, repeats=1)

    def test_both_methods_produce_real_positive_numbers(self):
        for method in ("uniform", "fovea"):
            self.assertGreater(self.report[method]["output_cell_count"], 0)
            self.assertGreater(self.report[method]["estimated_storage_bytes"], 0)
            self.assertGreater(self.report[method]["assigned_points"], 0)
            self.assertGreaterEqual(self.report[method]["mean_build_latency_s"], 0.0)

    def test_fvch1_payload_is_real_and_fovea_only(self):
        self.assertGreater(self.report["fovea"]["fvch1_serialized_payload_bytes"], 0)
        self.assertGreater(self.report["fovea"]["fvch1_chunk_count"], 0)
        self.assertIn("FOVEA-only", self.report["fovea"]["fvch1_note"])

    def test_reduction_factors_are_real_ratios_not_hardcoded(self):
        # No claim about direction -- measured reality: FOVEA is not
        # guaranteed to have FEWER cells than a fixed-grid uniform baseline
        # (its adaptive fine regions can outnumber a coarser uniform grid's
        # cells even while its far-field is coarser) -- just check the ratio
        # is a real, positive, correctly-computed number.
        factor = self.report["cell_count_reduction_factor_uniform_over_fovea"]
        self.assertAlmostEqual(
            factor,
            self.report["uniform"]["output_cell_count"] / self.report["fovea"]["output_cell_count"],
            places=6,
        )


class TestPipelineLatencyReport(unittest.TestCase):
    """Requirement D."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.report = pipeline_latency_report(cls.frame.points, cls.frame.semantic_labels, CFG, perception_model=None)

    def test_stage_timings_are_real_and_positive(self):
        for key in ("fovea_build_s", "hierarchy_build_s", "lod_selection_s", "chunk_serialization_s"):
            self.assertGreater(self.report[key], 0.0)

    def test_backend_total_is_sum_of_stages_without_perception(self):
        expected = (
            self.report["fovea_build_s"] + self.report["hierarchy_build_s"]
            + self.report["lod_selection_s"] + self.report["chunk_serialization_s"]
        )
        self.assertAlmostEqual(self.report["backend_total_s"], expected, places=6)

    def test_perception_ms_is_none_when_no_model_passed(self):
        self.assertIsNone(self.report["perception_ms"])

    def test_frontend_figures_are_cited_not_fabricated_and_fps_labeled_non_representative(self):
        frontend = self.report["frontend_cited_not_remeasured"]
        self.assertFalse(frontend["fps_is_representative_of_real_gpu_hardware"])
        self.assertIn("Not re-measured", frontend["provenance"])


class TestPerceptionDistanceBandsReport(unittest.TestCase):
    """Requirement E -- real SqueezeSeg inference, one real frame (kept to
    one frame in the unit test for speed; the real benchmark run covers 3)."""

    @classmethod
    def setUpClass(cls):
        from src.perception.model import get_squeezeseg_model

        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.model = get_squeezeseg_model(CFG)
        cls.report = perception_distance_bands_report("SemanticPOSS", [(SEQUENCE, FRAME_ID)], cls.adapter, cls.model)

    def test_frame_evaluated_with_four_distance_bands(self):
        frame_result = self.report["per_frame"][0]
        self.assertTrue(frame_result["available"])
        self.assertEqual(set(frame_result["distance_bands"].keys()), {"0-10", "10-30", "30-60", "60-100"})

    def test_accuracy_and_miou_in_valid_range(self):
        frame_result = self.report["per_frame"][0]
        self.assertGreaterEqual(frame_result["overall_accuracy"], 0.0)
        self.assertLessEqual(frame_result["overall_accuracy"], 1.0)
        self.assertGreaterEqual(frame_result["mean_iou"], 0.0)
        self.assertLessEqual(frame_result["mean_iou"], 1.0)

    def test_no_fabricated_metric_for_bands_with_no_points(self):
        # Any band actually reporting point_count == 0 must have None
        # accuracy/mean_iou, never a fabricated 0.0 -- evaluate_predictions'
        # own convention, passed through unchanged.
        for band in self.report["per_frame"][0]["distance_bands"].values():
            if band["point_count"] == 0:
                self.assertIsNone(band["accuracy"])
                self.assertIsNone(band["mean_iou"])


class TestTerrainObstacleReport(unittest.TestCase):
    """Requirement F."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.cells, cls.accounting = build_fovea_map(cls.frame.points, cls.frame.semantic_labels, CFG)
        cls.report = terrain_obstacle_report(cls.cells, cls.accounting, "SemanticPOSS")

    def test_terrain_categories_sum_to_total(self):
        r = self.report
        self.assertEqual(
            r["drivable_cells"] + r["non_drivable_cells"] + r["obstacle_cells"] + r["unknown_terrain_cells"],
            r["total_cells"],
        )

    def test_obstacle_breakdown_sums_to_obstacle_cells(self):
        r = self.report
        self.assertEqual(r["static_obstacle_cells"] + r["potential_dynamic_obstacle_cells"], r["obstacle_cells"])

    def test_ground_truth_not_conflated_with_validated_terrain(self):
        self.assertIn("NOT validated against real terrain ground truth", self.report["note"])


class TestOrchestration(unittest.TestCase):
    """Lightweight end-to-end check of run_sih_compliance_benchmark, kept to
    one frame per dataset and no perception (for test speed) -- the real
    >=3+3-frame, real-perception run is the CLI (`python -m
    src.benchmarking.sih_compliance`), exercised separately, not as a unit test."""

    def test_orchestration_produces_both_dataset_sections(self):
        report = run_sih_compliance_benchmark(
            CFG, semanticposs_frames=[(SEQUENCE, FRAME_ID)], rellis_frames=[("00000", "000100")], include_perception=False
        )
        self.assertIn("SemanticPOSS", report)
        sp = report["SemanticPOSS"]
        for key in ("A_resolution_vs_distance", "B_coverage_and_conservation_last_frame", "C_memory_benchmark_per_frame", "D_pipeline_latency_per_frame", "F_terrain_obstacle_per_frame"):
            self.assertIn(key, sp)
        self.assertNotIn("E_perception_distance_bands", report)  # include_perception=False -> skipped, not fabricated

    @unittest.skipUnless(_RELLIS_PRESENT, "RELLIS-3D data not present on this machine")
    def test_rellis_section_present_when_data_available(self):
        report = run_sih_compliance_benchmark(
            CFG, semanticposs_frames=[(SEQUENCE, FRAME_ID)], rellis_frames=[("00000", "000100")], include_perception=False
        )
        self.assertIn("A_resolution_vs_distance", report["RELLIS-3D"])

    def test_effective_cfg_for_semanticposs_is_unchanged_cfg(self):
        adapter = get_adapter("SemanticPOSS", CFG)
        self.assertIs(effective_cfg_for("SemanticPOSS", CFG, adapter), CFG)

    def test_default_frame_lists_have_at_least_three_frames(self):
        self.assertGreaterEqual(len(DEFAULT_SEMANTICPOSS_FRAMES), 3)
        self.assertGreaterEqual(len(DEFAULT_RELLIS_FRAMES), 3)


if __name__ == "__main__":
    unittest.main()
