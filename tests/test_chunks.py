"""Tests for the streamable spatial chunk / serialization layer
(src/spatial/chunks.py) and its API endpoint -- Phase 3C.

Groups:
  - TestChunkBuildingSynthetic: hand-built hierarchy with independently
    hand-computed expected chunk fields (leaf chunk == exact Cell values;
    aggregate chunk == hand-computed weighted average).
  - TestChunkSerializationRoundTrip: binary format round trip, determinism,
    empty/multiple chunks, corrupt-payload rejection.
  - TestChunksApiEndpoint: real FastAPI app (TestClient, no mocking) --
    endpoint behavior, cache reuse/isolation, concurrency coalescing.
  - TestChunksRealFovea / TestChunksRealRellis: real SemanticPOSS/RELLIS-3D
    frames through the real, unmodified full pipeline.

Run with: python -m pytest tests/test_chunks.py -v
"""
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.datasets import get_adapter
from src.fovea_allocator import build_fovea_map
from src.geometry import cell_center
from src.map_structure import Cell
from src.spatial.chunks import (
    SpatialChunk,
    build_chunks,
    deserialize_chunk,
    node_id,
    serialize_chunk,
    serialize_chunks_with_metadata,
    verify_chunk_roundtrip,
)
from src.spatial.hierarchy import build_fovea_hierarchy
from src.spatial.lod import LODSelectionConfig, select_lod_nodes

CFG = load_config("config.yaml")
SEQUENCE = "00"
FRAME_ID = "000000"
_RELLIS_PRESENT = Path(CFG["datasets"]["RELLIS-3D"]["pointcloud_root"]).exists()

SYN_CFG = {"origin": {"x": 0.0, "y": 0.0}, "fovea": {"refinement_factor": 4}}
ORIGIN = (0.0, 0.0)


def _make_cell(cell_size_m, resolution_level, band_id, ix, iy, point_count, majority_class_name="car",
                majority_class_fraction=0.8, semantic_priority=0.9, geometry_score=0.4, dynamic_score=1.0,
                information_score=0.77, elevation_median=1.0, elevation_min=0.1, elevation_max=1.9) -> Cell:
    cx = float(cell_center(np.array([ix], dtype=np.float64), ORIGIN[0], cell_size_m)[0])
    cy = float(cell_center(np.array([iy], dtype=np.float64), ORIGIN[1], cell_size_m)[0])
    return Cell(
        cell_size_m=cell_size_m, resolution_level=resolution_level, band_id=band_id,
        ix=ix, iy=iy, center_x=cx, center_y=cy, point_count=point_count,
        elevation_median=elevation_median, elevation_std=0.1, elevation_min=elevation_min, elevation_max=elevation_max,
        majority_class_id=0, majority_class_name=majority_class_name, majority_class_fraction=majority_class_fraction,
        semantic_priority=semantic_priority, geometry_score=geometry_score, dynamic_score=dynamic_score,
        information_score=information_score, range_m=float(np.hypot(cx, cy)),
    )


class TestChunkBuildingSynthetic(unittest.TestCase):
    def test_leaf_chunk_matches_cell_exactly(self):
        cell = _make_cell(1.0, 0, 0, 5, 5, 10, majority_class_name="ground", semantic_priority=0.05)
        roots, _ = build_fovea_hierarchy([cell], {"assigned_points": 10}, SYN_CFG)
        chunks = build_chunks(roots, "ground_truth", "SemanticPOSS")

        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertTrue(c.is_leaf)
        self.assertEqual(c.point_count, 10)
        self.assertEqual(c.semantic_class, "ground")
        self.assertAlmostEqual(c.semantic_priority, 0.05, places=9)
        self.assertAlmostEqual(c.elevation_median, cell.elevation_median, places=9)
        self.assertAlmostEqual(c.center_x, cell.center_x, places=9)
        self.assertAlmostEqual(c.x_max - c.x_min, cell.cell_size_m, places=9)
        self.assertEqual(c.node_id, node_id(roots[0]))
        self.assertEqual(c.label_source, "ground_truth")
        self.assertEqual(c.source, "SemanticPOSS")
        # ground -> DRIVABLE terrain, not an obstacle -- reused, not reinvented, classification
        self.assertEqual(c.terrain_class, "drivable")
        self.assertEqual(c.obstacle_class, "non_obstacle")

    def test_aggregate_chunk_is_point_count_weighted(self):
        # Two fine leaves under one medium parent (never split further) --
        # the coarse root is left UNSPLIT here so it becomes one aggregate
        # chunk over both fine leaves directly.
        cells = [
            _make_cell(1.0, 2, 0, 0, 0, 3, majority_class_name="car", semantic_priority=0.9,
                       geometry_score=0.2, dynamic_score=1.0, information_score=0.9, elevation_median=2.0,
                       elevation_min=1.5, elevation_max=2.5, majority_class_fraction=1.0),
            _make_cell(1.0, 2, 0, 1, 0, 1, majority_class_name="ground", semantic_priority=0.05,
                       geometry_score=0.0, dynamic_score=0.0, information_score=0.05, elevation_median=0.0,
                       elevation_min=-0.2, elevation_max=0.2, majority_class_fraction=1.0),
        ]
        accounting = {"assigned_points": 4}
        roots, hstats = build_fovea_hierarchy(cells, accounting, SYN_CFG)
        self.assertEqual(len(roots), 1)
        self.assertFalse(roots[0].is_leaf)  # coarse node synthesized, has 1 medium child w/ 2 fine leaves

        chunks = build_chunks(roots, "ground_truth", "SemanticPOSS")
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertFalse(c.is_leaf)
        self.assertEqual(c.point_count, 4)  # 3 + 1, conservation holds through the chunk layer too

        # hand-computed weighted aggregates (weights 3 and 1, total 4)
        self.assertAlmostEqual(c.elevation_median, (2.0 * 3 + 0.0 * 1) / 4, places=6)
        self.assertAlmostEqual(c.semantic_priority, (0.9 * 3 + 0.05 * 1) / 4, places=6)
        self.assertAlmostEqual(c.dynamic_score, (1.0 * 3 + 0.0 * 1) / 4, places=6)
        self.assertAlmostEqual(c.information_score, (0.9 * 3 + 0.05 * 1) / 4, places=6)
        self.assertEqual(c.elevation_min, -0.2)  # real min across descendants, not fabricated
        self.assertEqual(c.elevation_max, 2.5)
        # "car" carries 3/4 of the weighted majority-class points -> wins the aggregate vote
        self.assertEqual(c.semantic_class, "car")
        self.assertAlmostEqual(c.majority_class_fraction, 3.0 / 4.0, places=6)

    def test_no_raw_points_in_a_chunk(self):
        """SpatialChunk must never carry a raw per-point array -- only the
        dataclass's declared summary fields exist."""
        cell = _make_cell(1.0, 0, 0, 0, 0, 500)
        roots, _ = build_fovea_hierarchy([cell], {"assigned_points": 500}, SYN_CFG)
        chunks = build_chunks(roots, "ground_truth", "SemanticPOSS")
        field_names = set(vars(chunks[0]).keys())
        for forbidden in ("points", "x", "y", "z", "xyz", "raw_points", "point_cloud"):
            self.assertNotIn(forbidden, field_names)


class TestChunkSerializationRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cells = [
            _make_cell(1.0, 0, 0, 0, 0, 10, majority_class_name="ground"),
            _make_cell(0.5, 1, 0, 4, 4, 7, majority_class_name="car", dynamic_score=1.0),
            _make_cell(0.25, 2, 0, 40, 40, 3, majority_class_name="1 person", dynamic_score=1.0, semantic_priority=1.0),
        ]
        accounting = {"assigned_points": sum(c.point_count for c in cells)}
        cls.roots, _ = build_fovea_hierarchy(cells, accounting, SYN_CFG)
        cls.chunks = build_chunks(cls.roots, "ground_truth", "SemanticPOSS")

    def test_round_trip_preserves_values(self):
        result = verify_chunk_roundtrip(self.chunks)
        self.assertTrue(result["roundtrip_ok"], result["mismatches"])
        self.assertEqual(result["chunk_count"], len(self.chunks))
        self.assertEqual(result["mismatch_count"], 0)

    def test_round_trip_exact_and_tolerant_fields(self):
        payload = serialize_chunk(self.chunks)
        restored = deserialize_chunk(payload)
        self.assertEqual(len(restored), len(self.chunks))
        for original, back in zip(self.chunks, restored):
            # integer/string fields: EXACT
            self.assertEqual(original.node_id, back.node_id)
            self.assertEqual(original.level, back.level)
            self.assertEqual(original.point_count, back.point_count)
            self.assertEqual(original.semantic_class, back.semantic_class)
            self.assertEqual(original.terrain_class, back.terrain_class)
            self.assertEqual(original.is_leaf, back.is_leaf)
            # float32-carried fields: tolerant
            self.assertAlmostEqual(original.center_x, back.center_x, places=4)
            self.assertAlmostEqual(original.information_score, back.information_score, places=4)

    def test_metadata_preservation(self):
        payload, meta = serialize_chunks_with_metadata(
            self.roots, "ground_truth", "SemanticPOSS", "SemanticPOSS", "00", "000000",
        )
        self.assertEqual(meta.chunk_count, len(deserialize_chunk(payload)))
        self.assertEqual(meta.payload_bytes, len(payload))
        self.assertEqual(meta.payload_format, "FVCH1")
        self.assertGreaterEqual(meta.serialization_latency_s, 0.0)
        self.assertEqual(meta.dataset, "SemanticPOSS")
        self.assertEqual(meta.sequence, "00")
        self.assertEqual(meta.frame_id, "000000")
        self.assertEqual(meta.selected_point_count, sum(c.point_count for c in deserialize_chunk(payload)))

    def test_deterministic_serialization(self):
        payload_a = serialize_chunk(self.chunks)
        payload_b = serialize_chunk(self.chunks)
        self.assertEqual(payload_a, payload_b)

    def test_empty_selection_round_trips_to_empty_list(self):
        payload = serialize_chunk([])
        self.assertEqual(deserialize_chunk(payload), [])
        result = verify_chunk_roundtrip([])
        self.assertTrue(result["roundtrip_ok"])
        self.assertEqual(result["chunk_count"], 0)

    def test_multiple_chunks_preserve_order_and_distinct_ids(self):
        payload = serialize_chunk(self.chunks)
        restored = deserialize_chunk(payload)
        self.assertEqual([c.node_id for c in restored], [c.node_id for c in self.chunks])
        self.assertEqual(len({c.node_id for c in restored}), len(self.chunks))

    def test_payload_smaller_than_naive_json_equivalent(self):
        import dataclasses
        import json

        payload = serialize_chunk(self.chunks)
        naive_json_bytes = len(json.dumps([dataclasses.asdict(c) for c in self.chunks]).encode("utf-8"))
        self.assertLess(len(payload), naive_json_bytes)

    def test_corrupt_payload_wrong_magic_rejected(self):
        payload = bytearray(serialize_chunk(self.chunks))
        payload[0:6] = b"XXXXXX"
        with self.assertRaises(ValueError):
            deserialize_chunk(bytes(payload))

    def test_corrupt_payload_truncated_rejected(self):
        payload = serialize_chunk(self.chunks)
        with self.assertRaises(ValueError):
            deserialize_chunk(payload[:-5])

    def test_corrupt_payload_too_short_rejected(self):
        with self.assertRaises(ValueError):
            deserialize_chunk(b"\x00\x01")

    def test_corrupt_payload_bad_string_index_rejected(self):
        import struct

        payload = bytearray(serialize_chunk(self.chunks))
        # Overwrite the FIRST chunk record's semantic_class_idx (first uint16
        # in the trailing 6-uint16 block) with an out-of-range value.
        idx_offset = len(payload) - len(self.chunks) * 94 + (94 - 12)  # start of the 6 H's in the first record
        struct.pack_into("<H", payload, idx_offset, 0xFFFF)
        with self.assertRaises(ValueError):
            deserialize_chunk(bytes(payload))


class TestChunksApiEndpoint(unittest.TestCase):
    """Real FastAPI app (TestClient), no mocking."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from src.api import app

        cls.client = TestClient(app)

    def test_endpoint_returns_valid_chunk_payload(self):
        r = self.client.get(f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{FRAME_ID}/chunks?detail_threshold=0.02")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["dataset"], "SemanticPOSS")
        self.assertEqual(body["label_source"], "ground_truth")
        self.assertEqual(body["payload_format"], "FVCH1")
        self.assertGreater(body["chunk_count"], 0)
        self.assertEqual(body["chunk_count"], body["selected_node_count"])
        self.assertGreater(body["payload_bytes"], 0)

        import base64

        raw = base64.b64decode(body["payload_base64"])
        self.assertEqual(len(raw), body["payload_bytes"])
        restored = deserialize_chunk(raw)
        self.assertEqual(len(restored), body["chunk_count"])

    def test_timing_fields_present_and_separate(self):
        r = self.client.get(f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{FRAME_ID}/chunks?detail_threshold=0.02")
        timing = r.json()["timing"]
        for field in (
            "fovea_build_latency_s", "hierarchy_build_latency_s",
            "lod_selection_latency_s", "chunk_serialization_latency_s",
        ):
            self.assertIn(field, timing)
            self.assertGreaterEqual(timing[field], 0.0)

    def test_max_distance_and_point_budget_query_params_respected(self):
        r_full = self.client.get(f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{FRAME_ID}/chunks?detail_threshold=0.02")
        r_culled = self.client.get(
            f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{FRAME_ID}/chunks?detail_threshold=0.02&max_distance=10"
        )
        r_budget = self.client.get(
            f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{FRAME_ID}/chunks?detail_threshold=0.02&point_budget=1000"
        )
        self.assertLess(r_culled.json()["chunk_count"], r_full.json()["chunk_count"])
        self.assertLessEqual(r_budget.json()["selected_point_count"], 1000)
        self.assertGreater(r_budget.json()["budget_dropped_node_count"], 0)

    def test_unknown_dataset_404s(self):
        r = self.client.get(f"/api/datasets/NotARealDataset/frames/{SEQUENCE}/{FRAME_ID}/chunks")
        self.assertEqual(r.status_code, 404)

    def test_unknown_frame_404s(self):
        r = self.client.get(f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/999999/chunks")
        self.assertEqual(r.status_code, 404)

    def test_cache_hit_returns_identical_payload_and_timing(self):
        frame_id = "000010"
        params = "detail_threshold=0.015"
        first = self.client.get(f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{frame_id}/chunks?{params}")
        second = self.client.get(f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{frame_id}/chunks?{params}")
        self.assertEqual(first.json()["payload_base64"], second.json()["payload_base64"])
        # Identical timing proves the second call reused the cached result
        # rather than re-running LOD selection / re-serializing (a fresh
        # computation would re-measure perf_counter() and almost certainly
        # differ) -- same technique test_api.py's own cache tests use.
        self.assertEqual(first.json()["timing"], second.json()["timing"])

    def test_different_params_are_different_cache_keys(self):
        frame_id = "000011"
        loose = self.client.get(f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{frame_id}/chunks?detail_threshold=1.0")
        tight = self.client.get(f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{frame_id}/chunks?detail_threshold=0.001")
        self.assertNotEqual(loose.json()["chunk_count"], tight.json()["chunk_count"])
        self.assertNotEqual(loose.json()["payload_base64"], tight.json()["payload_base64"])

    def test_concurrent_requests_for_fresh_params_do_not_duplicate_computation(self):
        frame_id = "000012"  # not requested by any other test with this exact param set
        url = f"/api/datasets/SemanticPOSS/frames/{SEQUENCE}/{frame_id}/chunks?detail_threshold=0.013"

        def fetch(_):
            return self.client.get(url)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(fetch, range(8)))

        for r in results:
            self.assertEqual(r.status_code, 200)
        timings = {tuple(r.json()["timing"].values()) for r in results}
        payloads = {r.json()["payload_base64"] for r in results}
        self.assertEqual(len(timings), 1, "concurrent requests measured different latencies -> recomputed independently")
        self.assertEqual(len(payloads), 1)

    def test_legacy_map_endpoints_unaffected(self):
        r = self.client.get(f"/api/frames/{SEQUENCE}/{FRAME_ID}/map/fovea?include_cells=false")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["summary"]["output_cell_count"], 41868)


class TestChunksRealFovea(unittest.TestCase):
    """Real SemanticPOSS frames through the real, unmodified full pipeline:
    build_fovea_map -> build_fovea_hierarchy -> select_lod_nodes -> chunks."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)

    def test_three_real_semanticposs_frames(self):
        for fid in ["000000", "000050", "000100"]:
            frame = self.adapter.load_frame(SEQUENCE, fid)
            cells, accounting = build_fovea_map(frame.points, frame.semantic_labels, CFG)
            roots, hstats = build_fovea_hierarchy(cells, accounting, CFG)
            selection = select_lod_nodes(roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=0.01))
            chunks = build_chunks(selection.nodes, "ground_truth", "SemanticPOSS")
            payload, meta = serialize_chunks_with_metadata(
                selection.nodes, "ground_truth", "SemanticPOSS", "SemanticPOSS", SEQUENCE, fid
            )

            self.assertEqual(len(chunks), selection.node_count)
            self.assertEqual(meta.chunk_count, len(chunks))
            self.assertEqual(meta.payload_bytes, len(payload))
            self.assertEqual(sum(c.point_count for c in chunks), selection.total_point_count)

            result = verify_chunk_roundtrip(chunks)
            self.assertTrue(result["roundtrip_ok"], result["mismatches"])

            # FOVEA's own output must be completely unaffected by chunking.
            cells2, accounting2 = build_fovea_map(frame.points, frame.semantic_labels, CFG)
            self.assertEqual(len(cells2), len(cells))
            self.assertEqual(accounting2["assigned_points"], accounting["assigned_points"])


@unittest.skipUnless(_RELLIS_PRESENT, "RELLIS-3D data not present on this machine")
class TestChunksRealRellis(unittest.TestCase):
    """Real RELLIS-3D frames -- confirms the chunk layer is dataset-agnostic
    (reads only HierarchyNode/Cell fields, same convention as map_2_5d.py)."""

    def test_three_real_rellis_frames(self):
        import copy

        adapter = get_adapter("RELLIS-3D", CFG)
        cfg = copy.deepcopy(CFG)
        cfg["fovea"]["semantic_priority"] = adapter.semantic_priority_table()
        cfg["fovea"]["dynamic_classes"] = list(adapter.dynamic_classes())

        for seq, fid in [("00000", "000100"), ("00002", "000500"), ("00004", "000050")]:
            frame = adapter.load_frame(seq, fid)
            cells, accounting = build_fovea_map(frame.points, frame.semantic_labels, cfg, label_dict=adapter.label_dict())
            roots, hstats = build_fovea_hierarchy(cells, accounting, cfg)
            selection = select_lod_nodes(roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=0.01))
            chunks = build_chunks(selection.nodes, "ground_truth", "RELLIS-3D")
            payload, meta = serialize_chunks_with_metadata(
                selection.nodes, "ground_truth", "RELLIS-3D", "RELLIS-3D", seq, fid
            )

            self.assertEqual(meta.chunk_count, len(chunks))
            self.assertEqual(meta.payload_bytes, len(payload))
            result = verify_chunk_roundtrip(chunks)
            self.assertTrue(result["roundtrip_ok"], result["mismatches"])
            self.assertGreater(meta.chunk_count, 0)


if __name__ == "__main__":
    unittest.main()
