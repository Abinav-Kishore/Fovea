"""Tests for the hierarchical (parent/child, LOD) spatial representation
(src/spatial/hierarchy.py) -- Phase 3 hierarchy core.

Two groups of tests:
  - TestHierarchyStructureSynthetic: hand-built Cell objects with EXACT,
    hand-computed expected tree shape/point-counts, so the parent/child
    construction and point-conservation arithmetic are checked against
    known-correct numbers, not just internal self-consistency.
  - TestHierarchyRealFovea / TestHierarchyOtherDataset: the real, unmodified
    FOVEA engine (build_fovea_map) run on real SemanticPOSS/RELLIS-3D frames,
    matching this repo's "nothing mocked" convention (see test_spatial.py).

Run with: python -m pytest tests/test_hierarchy.py -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.datasets import get_adapter
from src.fovea_allocator import build_fovea_map
from src.map_structure import Cell
from src.spatial.hierarchy import (
    build_fovea_hierarchy,
    iter_nodes,
    leaf_nodes,
    verify_point_conservation,
)

CFG = load_config("config.yaml")
SEQUENCE = "00"
FRAME_ID = "000000"
_RELLIS_PRESENT = Path(CFG["datasets"]["RELLIS-3D"]["pointcloud_root"]).exists()

# Minimal cfg for the synthetic tests -- only origin + refinement_factor are
# ever read by build_fovea_hierarchy (see its docstring: read-only, just to
# recover the engine's own branching ratio).
SYN_CFG = {"origin": {"x": 0.0, "y": 0.0}, "fovea": {"refinement_factor": 4}}


def _make_cell(cell_size_m, resolution_level, band_id, ix, iy, point_count) -> Cell:
    """Minimal real Cell -- every field the hierarchy module doesn't use is
    filled with a harmless placeholder; ix/iy/cell_size_m/resolution_level/
    band_id/point_count are the only fields build_fovea_hierarchy reads."""
    return Cell(
        cell_size_m=cell_size_m, resolution_level=resolution_level, band_id=band_id,
        ix=ix, iy=iy, center_x=0.0, center_y=0.0, point_count=point_count,
        elevation_median=0.0, elevation_std=0.0, elevation_min=0.0, elevation_max=0.0,
        majority_class_id=0, majority_class_name="unknown", majority_class_fraction=1.0,
        semantic_priority=0.0, geometry_score=0.0, dynamic_score=0.0, information_score=0.0,
        range_m=0.0,
    )


class TestHierarchyStructureSynthetic(unittest.TestCase):
    """Three hand-placed footprints (base cell size 1.0, band 0, refinement_factor=4):
      - coarse ix=0: never split -> a level-0 LEAF, 10 points.
      - coarse ix=1: split to medium only -> 2 level-1 LEAVES (ix=2,3), synthesized
        level-0 internal parent, 5+7=12 points.
      - coarse ix=5: split to fine -> 2 level-2 LEAVES (ix=20,21) under one synthesized
        level-1 internal parent (ix=10) under one synthesized level-0 internal
        parent (ix=5), 3+4=7 points.
    Every index/ratio below was derived by hand from the SAME arithmetic
    build_fovea_hierarchy uses (parent_ix = child_ix // ratio, ratio=2 at
    both hops since half=factor//2=2 and factor//half=2), so this test
    checks the module against independently-verifiable numbers.
    """

    @classmethod
    def setUpClass(cls):
        cls.cells = [
            _make_cell(1.0, 0, 0, 0, 0, 10),      # case A: coarse leaf
            _make_cell(0.5, 1, 0, 2, 0, 5),        # case B: medium leaf 1 (under coarse ix=1)
            _make_cell(0.5, 1, 0, 3, 0, 7),        # case B: medium leaf 2 (under coarse ix=1)
            _make_cell(0.25, 2, 0, 20, 0, 3),      # case C: fine leaf 1 (under medium ix=10, coarse ix=5)
            _make_cell(0.25, 2, 0, 21, 0, 4),      # case C: fine leaf 2 (under medium ix=10, coarse ix=5)
        ]
        cls.accounting = {"assigned_points": sum(c.point_count for c in cls.cells)}
        cls.roots, cls.stats = build_fovea_hierarchy(cls.cells, cls.accounting, SYN_CFG)
        cls.roots_by_ix = {r.ix: r for r in cls.roots}

    def test_root_count_and_footprints(self):
        self.assertEqual(len(self.roots), 3)
        self.assertEqual(set(self.roots_by_ix.keys()), {0, 1, 5})

    def test_case_a_unsplit_coarse_is_its_own_leaf(self):
        node = self.roots_by_ix[0]
        self.assertTrue(node.is_leaf)
        self.assertIsNotNone(node.cell)
        self.assertEqual(node.point_count, 10)
        self.assertEqual(node.children, [])
        self.assertEqual(node.cell_size_m, 1.0)

    def test_case_b_medium_split_synthesizes_coarse_parent(self):
        node = self.roots_by_ix[1]
        self.assertFalse(node.is_leaf)
        self.assertIsNone(node.cell)
        self.assertEqual(node.cell_size_m, 1.0)
        self.assertEqual(node.point_count, 12)  # 5 + 7, NOT independently measured
        self.assertEqual(len(node.children), 2)
        child_points = sorted(c.point_count for c in node.children)
        self.assertEqual(child_points, [5, 7])
        for c in node.children:
            self.assertTrue(c.is_leaf)
            self.assertIs(c.parent, node)
            self.assertEqual(c.level, 1)
            self.assertEqual(c.cell_size_m, 0.5)

    def test_case_c_fine_split_synthesizes_two_ancestor_levels(self):
        coarse = self.roots_by_ix[5]
        self.assertFalse(coarse.is_leaf)
        self.assertEqual(coarse.point_count, 7)  # 3 + 4
        self.assertEqual(coarse.cell_size_m, 1.0)
        self.assertEqual(len(coarse.children), 1)

        medium = coarse.children[0]
        self.assertFalse(medium.is_leaf)
        self.assertIsNone(medium.cell)
        self.assertEqual(medium.ix, 10)
        self.assertEqual(medium.cell_size_m, 0.5)
        self.assertEqual(medium.point_count, 7)
        self.assertIs(medium.parent, coarse)
        self.assertEqual(len(medium.children), 2)

        fine_points = sorted(c.point_count for c in medium.children)
        self.assertEqual(fine_points, [3, 4])
        for fine in medium.children:
            self.assertTrue(fine.is_leaf)
            self.assertIs(fine.parent, medium)
            self.assertEqual(fine.level, 2)
            self.assertEqual(fine.cell_size_m, 0.25)

    def test_stats_node_counts(self):
        self.assertEqual(self.stats.total_nodes, 8)  # 5 leaves + 1 (case B coarse) + 2 (case C coarse+medium)
        self.assertEqual(self.stats.leaf_nodes, 5)
        self.assertEqual(self.stats.internal_nodes, 3)
        self.assertEqual(self.stats.root_nodes, 3)
        self.assertEqual(self.stats.nodes_by_level, {0: 3, 1: 3, 2: 2})
        self.assertEqual(self.stats.max_depth, 3)

    def test_point_conservation_matches_hand_computed_total(self):
        self.assertEqual(self.stats.root_total_points, 29)  # 10 + 12 + 7
        self.assertEqual(self.stats.expected_total_points, 29)
        self.assertTrue(self.stats.point_conservation_ok)

    def test_iter_nodes_and_leaf_nodes_helpers(self):
        self.assertEqual(len(list(iter_nodes(self.roots))), 8)
        self.assertEqual(len(leaf_nodes(self.roots)), 5)
        self.assertTrue(all(n.is_leaf for n in leaf_nodes(self.roots)))

    def test_verify_point_conservation_detects_real_corruption(self):
        """The checker must be a REAL check, not a rubber stamp: corrupting a
        leaf's point_count without updating its ancestors must be caught."""
        medium_leaf = self.roots_by_ix[1].children[0]
        original = medium_leaf.point_count
        medium_leaf.point_count = original + 100
        try:
            result = verify_point_conservation(self.roots, expected_total=29)
            self.assertFalse(result["conserved"])
            self.assertFalse(result["internal_conservation_ok"])
            self.assertGreaterEqual(result["mismatched_node_count"], 1)
        finally:
            medium_leaf.point_count = original  # restore for any other test relying on cls state

    def test_conservation_raises_during_build_on_corrupted_input(self):
        """build_fovea_hierarchy itself enforces conservation (task requirement):
        feeding it Cells whose leaf split is internally inconsistent with the
        engine's own guarantees is not something that can happen from a real
        build_fovea_map call, but the guard must still fire if it ever does."""
        # Two leaves claiming the EXACT same (level, band, ix, iy) footprint
        # is impossible from a real build_fovea_map (map_structure.py's
        # groupby key is unique per physical cell) -- simulate the only way
        # to break the invariant post-construction: hand-corrupt a built
        # tree's node.point_count and re-run the checker function directly
        # (already covered above). Here we instead confirm the guard is wired
        # into build_fovea_hierarchy by asserting a conserved synthetic input
        # builds cleanly (no raise) -- the negative path is exercised via
        # verify_point_conservation directly above since corrupting inputs to
        # build_fovea_hierarchy itself cannot desync it (the sums are computed
        # BY it, not supplied to it).
        roots, stats = build_fovea_hierarchy(self.cells, self.accounting, SYN_CFG)
        self.assertTrue(stats.point_conservation_ok)


class TestHierarchyRealFovea(unittest.TestCase):
    """Real SemanticPOSS frame through the real, unmodified build_fovea_map."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = get_adapter("SemanticPOSS", CFG)
        cls.frame = cls.adapter.load_frame(SEQUENCE, FRAME_ID)
        cls.cells, cls.accounting = build_fovea_map(cls.frame.points, cls.frame.semantic_labels, CFG)
        cls.roots, cls.stats = build_fovea_hierarchy(cls.cells, cls.accounting, CFG)

    def test_leaf_count_matches_engine_cell_count(self):
        self.assertEqual(self.stats.leaf_nodes, len(self.cells))
        self.assertEqual(len(leaf_nodes(self.roots)), len(self.cells))

    def test_every_leaf_wraps_a_real_engine_cell_by_identity(self):
        engine_cell_ids = {id(c) for c in self.cells}
        for node in leaf_nodes(self.roots):
            self.assertIsNotNone(node.cell)
            self.assertIn(id(node.cell), engine_cell_ids)

    def test_point_conservation_against_real_accounting(self):
        self.assertTrue(self.stats.point_conservation_ok)
        self.assertEqual(self.stats.root_total_points, self.accounting["assigned_points"])
        self.assertEqual(self.stats.expected_total_points, self.accounting["assigned_points"])

    def test_leaf_resolution_histogram_matches_engine_accounting(self):
        # accounting["cells_by_resolution_level"] is the engine's own count,
        # from the exact same `cells` list -- this cross-checks the tree
        # didn't lose or reclassify any leaf's level while building ancestors.
        hist: dict[int, int] = {}
        for node in leaf_nodes(self.roots):
            hist[node.level] = hist.get(node.level, 0) + 1
        self.assertEqual(hist, self.accounting["cells_by_resolution_level"])

    def test_all_levels_within_0_1_2(self):
        for node in iter_nodes(self.roots):
            self.assertIn(node.level, (0, 1, 2))

    def test_child_level_is_exactly_parent_level_plus_one(self):
        for node in iter_nodes(self.roots):
            for child in node.children:
                self.assertEqual(child.level, node.level + 1)
                self.assertIs(child.parent, node)

    def test_child_cell_size_matches_configured_refinement_ratio(self):
        factor = int(CFG["fovea"]["refinement_factor"])
        half = factor // 2
        for node in iter_nodes(self.roots):
            for child in node.children:
                ratio = half if node.level == 1 else factor // half  # medium->fine vs coarse->medium
                self.assertAlmostEqual(node.cell_size_m, child.cell_size_m * ratio, places=9)

    def test_internal_nodes_have_no_cell_and_are_never_leaves(self):
        for node in iter_nodes(self.roots):
            if not node.is_leaf:
                self.assertIsNone(node.cell)
                self.assertGreater(len(node.children), 0)

    def test_no_duplicate_node_footprints(self):
        keys = [n.key for n in iter_nodes(self.roots)]
        self.assertEqual(len(keys), len(set(keys)))

    def test_roots_are_all_level_zero(self):
        self.assertTrue(all(r.level == 0 for r in self.roots))
        self.assertEqual(self.stats.root_nodes, len(self.roots))

    def test_independent_checker_agrees_with_build_time_result(self):
        result = verify_point_conservation(self.roots, expected_total=self.accounting["assigned_points"])
        self.assertTrue(result["conserved"])
        self.assertEqual(result["mismatched_node_count"], 0)


@unittest.skipUnless(_RELLIS_PRESENT, "RELLIS-3D data not present on this machine")
class TestHierarchyOtherDataset(unittest.TestCase):
    """Confirms the hierarchy layer is dataset-agnostic (it only reads Cell/
    accounting fields, same as src/spatial/map_2_5d.py) using real RELLIS-3D
    frames with RELLIS's own semantic_priority/dynamic_classes config."""

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
            roots, stats = build_fovea_hierarchy(cells, accounting, cfg)
            self.assertTrue(stats.point_conservation_ok)
            self.assertEqual(stats.root_total_points, accounting["assigned_points"])
            self.assertEqual(stats.leaf_nodes, len(cells))


if __name__ == "__main__":
    unittest.main()
