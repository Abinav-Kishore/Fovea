"""Tests for the LOD (level-of-detail) selection layer (src/spatial/lod.py)
-- Phase 3B.

Two groups of tests:
  - TestLODSelectionSynthetic: hand-built hierarchies with geometrically
    real (ix/iy/cell_size/origin-consistent) node centers, so near/far,
    culling, budget, and determinism can be checked against independently
    computed distances -- not just internal self-consistency.
  - TestLODSelectionRealFovea / TestLODSelectionRealRellis: the real,
    unmodified FOVEA engine + hierarchy on real SemanticPOSS/RELLIS-3D
    frames, matching this repo's "nothing mocked" convention.

Run with: python -m pytest tests/test_lod.py -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.datasets import get_adapter
from src.fovea_allocator import build_fovea_map
from src.geometry import cell_center
from src.map_structure import Cell
from src.spatial.hierarchy import build_fovea_hierarchy, iter_nodes
from src.spatial.lod import (
    LODSelectionConfig,
    select_lod_nodes,
    verify_lod_frontier,
)

CFG = load_config("config.yaml")
SEQUENCE = "00"
FRAME_ID = "000000"
_RELLIS_PRESENT = Path(CFG["datasets"]["RELLIS-3D"]["pointcloud_root"]).exists()

SYN_CFG = {"origin": {"x": 0.0, "y": 0.0}, "fovea": {"refinement_factor": 4}}
ORIGIN = (0.0, 0.0)


def _make_cell(cell_size_m, resolution_level, band_id, ix, iy, point_count) -> Cell:
    """Real Cell with GEOMETRICALLY CONSISTENT center_x/center_y (via the
    real cell_center() formula, same as the actual engine uses), so LOD
    distance decisions in these tests are computed against real geometry,
    not placeholder coordinates."""
    cx = float(cell_center(np.array([ix], dtype=np.float64), ORIGIN[0], cell_size_m)[0])
    cy = float(cell_center(np.array([iy], dtype=np.float64), ORIGIN[1], cell_size_m)[0])
    return Cell(
        cell_size_m=cell_size_m, resolution_level=resolution_level, band_id=band_id,
        ix=ix, iy=iy, center_x=cx, center_y=cy, point_count=point_count,
        elevation_median=0.0, elevation_std=0.0, elevation_min=0.0, elevation_max=0.0,
        majority_class_id=0, majority_class_name="unknown", majority_class_fraction=1.0,
        semantic_priority=0.0, geometry_score=0.0, dynamic_score=0.0, information_score=0.0,
        range_m=float(np.hypot(cx, cy)),
    )


def _build_two_root_forest():
    """Root A: coarse(ix=0) -> medium(ix=0) -> 2 fine leaves(ix=0,1), all
    band 0, cell sizes 4.0/2.0/1.0 -- centered near (1-2, 0.5-2).
    Root B: same 3-level shape, but offset far away (coarse ix=100) so its
    coarse center sits ~400m from the origin -- centered near (400, 2).
    Both are independently derivable: fine leaves' ix // 2 == medium ix,
    medium ix // 2 == coarse ix (ratios from SYN_CFG's refinement_factor=4:
    half=2, factor//half=2 -- same arithmetic as test_hierarchy.py)."""
    cells = [
        _make_cell(1.0, 2, 0, 0, 0, 5),     # root A fine leaf 1
        _make_cell(1.0, 2, 0, 1, 0, 6),     # root A fine leaf 2
        _make_cell(1.0, 2, 0, 400, 0, 3),   # root B fine leaf 1 (under medium ix=200, coarse ix=100)
        _make_cell(1.0, 2, 0, 401, 0, 4),   # root B fine leaf 2
    ]
    accounting = {"assigned_points": sum(c.point_count for c in cells)}
    roots, stats = build_fovea_hierarchy(cells, accounting, SYN_CFG)
    return roots, stats, accounting


class TestLODSelectionSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots, cls.hstats, cls.accounting = _build_two_root_forest()
        cls.roots_by_ix = {r.ix: r for r in cls.roots}  # coarse ix=0 (A), ix=100 (B)

    def test_near_viewpoint_prefers_finest_descendants(self):
        # Viewpoint sits right next to root A's fine leaves (centers ~(0.5,0.5)/(1.5,0.5)).
        config = LODSelectionConfig(detail_threshold=0.5)
        sel = select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=config)
        a_selected = [n for n in sel.nodes if n.key[0] == 2 and n.ix in (0, 1)]
        self.assertEqual(len(a_selected), 2)  # both fine leaves of root A, not their ancestors
        selected_keys = {n.key for n in sel.nodes}
        coarse_a = self.roots_by_ix[0]
        medium_a = coarse_a.children[0]
        self.assertNotIn(coarse_a.key, selected_keys)
        self.assertNotIn(medium_a.key, selected_keys)

    def test_far_viewpoint_prefers_coarsest_ancestor(self):
        # Same viewpoint (near root A) is very far from root B (~400m away)
        # -> root B should be selected AT ITS COARSE ROOT, not descended.
        config = LODSelectionConfig(detail_threshold=0.5)
        sel = select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=config)
        coarse_b = self.roots_by_ix[100]
        selected_keys = {n.key for n in sel.nodes}
        self.assertIn(coarse_b.key, selected_keys)
        # neither of root B's fine leaves (nor its medium child) should appear
        for descendant in iter_nodes([coarse_b]):
            if descendant is not coarse_b:
                self.assertNotIn(descendant.key, selected_keys)

    def test_multiple_roots_both_represented(self):
        config = LODSelectionConfig(detail_threshold=0.5)
        sel = select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=config)
        self.assertEqual(sel.root_count, 2)
        # root A contributes 2 fine leaves + root B contributes 1 coarse node
        self.assertEqual(sel.node_count, 3)
        self.assertEqual(sel.total_point_count, 5 + 6 + (3 + 4))  # conservation still holds within selection

    def test_frontier_has_no_ancestor_descendant_overlap(self):
        config = LODSelectionConfig(detail_threshold=0.5)
        sel = select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=config)
        result = verify_lod_frontier(sel.nodes)
        self.assertTrue(result["valid"])
        self.assertTrue(result["no_duplicates"])
        self.assertTrue(result["no_ancestor_descendant_overlap"])

    def test_leaf_fallback_selects_leaf_even_at_zero_distance(self):
        # A never-split coarse root (genuine leaf, no children) with the
        # viewpoint essentially on top of it and an aggressive threshold that
        # would otherwise demand infinite refinement.
        leaf_cell = _make_cell(4.0, 0, 0, 500, 0, 9)
        roots, _ = build_fovea_hierarchy([leaf_cell], {"assigned_points": 9}, SYN_CFG)
        config = LODSelectionConfig(detail_threshold=0.0001)
        vx, vy = leaf_cell.center_x, leaf_cell.center_y
        sel = select_lod_nodes(roots, viewpoint=(vx, vy), config=config)
        self.assertEqual(sel.node_count, 1)
        self.assertTrue(sel.nodes[0].is_leaf)
        self.assertEqual(sel.nodes[0].point_count, 9)

    def test_max_distance_culls_far_root_entirely(self):
        config = LODSelectionConfig(detail_threshold=0.5, max_distance_m=50.0)
        sel = select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=config)
        selected_keys = {n.key for n in sel.nodes}
        coarse_b = self.roots_by_ix[100]
        for descendant in iter_nodes([coarse_b]):
            self.assertNotIn(descendant.key, selected_keys)
        self.assertEqual(sel.culled_root_count, 1)
        self.assertEqual(sel.node_count, 2)  # only root A's 2 fine leaves survive

    def test_deterministic_repeated_calls(self):
        config = LODSelectionConfig(detail_threshold=0.5)
        sel1 = select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=config)
        sel2 = select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=config)
        self.assertEqual([n.key for n in sel1.nodes], [n.key for n in sel2.nodes])

    def test_empty_hierarchy_returns_empty_selection_without_crashing(self):
        sel = select_lod_nodes([], viewpoint=(0.0, 0.0), config=LODSelectionConfig())
        self.assertEqual(sel.nodes, [])
        self.assertEqual(sel.node_count, 0)
        self.assertEqual(sel.total_point_count, 0)
        self.assertEqual(sel.root_count, 0)
        self.assertEqual(sel.culled_root_count, 0)
        self.assertGreaterEqual(sel.latency_s, 0.0)
        result = verify_lod_frontier(sel.nodes)
        self.assertTrue(result["valid"])

    def test_point_budget_drops_farthest_nodes_first(self):
        # Unconstrained: root A's 2 fine leaves (11 pts, close) + root B's
        # coarse node (7 pts, ~400m away) = 18 points, 3 nodes.
        config_unbudgeted = LODSelectionConfig(detail_threshold=0.5)
        unbudgeted = select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=config_unbudgeted)
        self.assertEqual(unbudgeted.total_point_count, 18)

        config = LODSelectionConfig(detail_threshold=0.5, point_budget=15)
        sel = select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=config)
        self.assertLessEqual(sel.total_point_count, 15)
        self.assertEqual(sel.budget_dropped_node_count, 1)
        coarse_b = self.roots_by_ix[100]
        self.assertNotIn(coarse_b.key, {n.key for n in sel.nodes})  # farthest node was dropped
        self.assertEqual(sel.total_point_count, 11)  # root A's 2 leaves remain intact
        self.assertTrue(verify_lod_frontier(sel.nodes)["valid"])

    def test_point_budget_zero_drops_everything(self):
        config = LODSelectionConfig(detail_threshold=0.5, point_budget=0)
        sel = select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=config)
        self.assertEqual(sel.node_count, 0)
        self.assertEqual(sel.total_point_count, 0)
        self.assertEqual(sel.budget_dropped_node_count, 3)

    def test_invalid_frontier_detection_ancestor_and_descendant_together(self):
        coarse_a = self.roots_by_ix[0]
        medium_a = coarse_a.children[0]
        fine_a = medium_a.children[0]
        result = verify_lod_frontier([medium_a, fine_a])  # ancestor + descendant, forbidden combo
        self.assertFalse(result["valid"])
        self.assertFalse(result["no_ancestor_descendant_overlap"])
        self.assertEqual(len(result["ancestor_descendant_violations"]), 1)

    def test_invalid_frontier_detection_duplicate_node(self):
        coarse_b = self.roots_by_ix[100]
        result = verify_lod_frontier([coarse_b, coarse_b])
        self.assertFalse(result["valid"])
        self.assertFalse(result["no_duplicates"])
        self.assertEqual(result["duplicate_keys"], [coarse_b.key])

    def test_valid_frontier_passes(self):
        config = LODSelectionConfig(detail_threshold=0.5)
        sel = select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=config)
        self.assertTrue(verify_lod_frontier(sel.nodes)["valid"])

    def test_lod_does_not_mutate_hierarchy(self):
        """LOD selection must be read-only: re-running hierarchy stats after
        an LOD call must show the exact same numbers (no node touched)."""
        before = {n.key: (n.point_count, n.is_leaf, len(n.children)) for n in iter_nodes(self.roots)}
        select_lod_nodes(self.roots, viewpoint=(1.0, 0.5), config=LODSelectionConfig(detail_threshold=0.5))
        select_lod_nodes(self.roots, viewpoint=(500.0, 500.0), config=LODSelectionConfig(max_distance_m=1.0))
        after = {n.key: (n.point_count, n.is_leaf, len(n.children)) for n in iter_nodes(self.roots)}
        self.assertEqual(before, after)


class TestLODSelectionRealFovea(unittest.TestCase):
    """Real SemanticPOSS frame through the real, unmodified build_fovea_map
    + build_fovea_hierarchy pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.cells, cls.accounting = build_fovea_map(cls.frame.points, cls.frame.semantic_labels, CFG)
        cls.roots, cls.hstats = build_fovea_hierarchy(cls.cells, cls.accounting, CFG)

    def test_sensor_origin_viewpoint_selects_valid_nonempty_frontier(self):
        sel = select_lod_nodes(self.roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=0.02))
        self.assertGreater(sel.node_count, 0)
        self.assertTrue(verify_lod_frontier(sel.nodes)["valid"])

    def test_lower_threshold_selects_at_least_as_many_nodes(self):
        loose = select_lod_nodes(self.roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=1.0))
        tight = select_lod_nodes(self.roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=0.01))
        self.assertGreaterEqual(tight.node_count, loose.node_count)

    def test_max_distance_reduces_or_equal_node_count(self):
        full = select_lod_nodes(self.roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=0.05))
        culled = select_lod_nodes(
            self.roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=0.05, max_distance_m=10.0)
        )
        self.assertLessEqual(culled.node_count, full.node_count)
        self.assertTrue(verify_lod_frontier(culled.nodes)["valid"])

    def test_fovea_outputs_unchanged_by_lod_selection(self):
        cells2, accounting2 = build_fovea_map(self.frame.points, self.frame.semantic_labels, CFG)
        select_lod_nodes(self.roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=0.05))
        self.assertEqual(len(cells2), len(self.cells))
        self.assertEqual(accounting2["output_cell_count"], self.accounting["output_cell_count"])
        self.assertEqual(accounting2["assigned_points"], self.accounting["assigned_points"])

    def test_lod_latency_is_recorded_and_small(self):
        sel = select_lod_nodes(self.roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=0.05))
        self.assertGreaterEqual(sel.latency_s, 0.0)
        self.assertLess(sel.latency_s, 5.0)  # generous upper bound, not a perf assertion


@unittest.skipUnless(_RELLIS_PRESENT, "RELLIS-3D data not present on this machine")
class TestLODSelectionRealRellis(unittest.TestCase):
    """Confirms the LOD layer is dataset-agnostic (reads only HierarchyNode
    fields) using real RELLIS-3D frames."""

    def test_three_real_rellis_frames(self):
        import copy

        adapter = get_adapter("RELLIS-3D", CFG)
        cfg = copy.deepcopy(CFG)
        cfg["fovea"]["semantic_priority"] = adapter.semantic_priority_table()
        cfg["fovea"]["dynamic_classes"] = list(adapter.dynamic_classes())

        for seq, fid in [("00000", "000100"), ("00002", "000500"), ("00004", "000050")]:
            frame = adapter.load_frame(seq, fid)
            cells, accounting = build_fovea_map(
                frame.points, frame.semantic_labels, cfg, label_dict=adapter.label_dict()
            )
            roots, hstats = build_fovea_hierarchy(cells, accounting, cfg)
            sel = select_lod_nodes(roots, viewpoint=(0.0, 0.0), config=LODSelectionConfig(detail_threshold=0.05))
            self.assertGreater(sel.node_count, 0)
            self.assertTrue(verify_lod_frontier(sel.nodes)["valid"])
            self.assertLessEqual(sel.total_point_count, hstats.root_total_points)


if __name__ == "__main__":
    unittest.main()
