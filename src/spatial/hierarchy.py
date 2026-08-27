"""FOVEA v0.2 hierarchical spatial representation (Phase 3 -- hierarchy core).

This module adds NO new spatial allocation or scoring logic -- it wraps the
EXISTING, unmodified FOVEA engine's flat output (a list of
`src.map_structure.Cell` + its `accounting` dict, from
`src.fovea_allocator.build_fovea_map`) into an explicit 3-level parent/child
tree matching the quadtree-style resolution levels the engine already
computes (`Cell.resolution_level`: 0=coarse, 1=medium, 2=fine -- see
src/fovea_allocator.py's module docstring).

    Cell (existing, from map_structure.py) -- one row per FINAL/leaf group,
    at whichever resolution level the engine's split decision selected
        v
    HierarchyNode (this module) -- that same leaf Cell, PLUS synthesized
    ancestor nodes (coarse/medium) the flat engine never materializes,
    because it only ever emits the resolution level actually selected for a
    region, not the coarser footprints that decision passed through.
        v
    HierarchyStats (this module) -- node/level counts + point conservation

Why ancestors have to be synthesized: `build_fovea_map` decides, per coarse
group, whether to refine to medium, and per medium group, whether to refine
further to fine, then aggregates points ONLY at the final selected
resolution (src/fovea_allocator.py Step 4). A coarse cell that split to
medium therefore never gets a Cell of its own -- only its medium (or fine)
children do. This module reconstructs the coarse/medium footprints those
decisions were made over, purely from each leaf Cell's own
(band_id, ix, iy, cell_size_m, resolution_level), using the SAME integer
grid relationship the engine's own index arithmetic already relies on
(child_ix // ratio == parent_ix, because every finer grid nests exactly
inside the next coarser one -- see src/geometry.py's module docstring).

`cfg["fovea"]["refinement_factor"]` is read exactly once, read-only, solely
to recover that ratio -- it is never used to recompute or override an
information_score, weight, threshold, or distance band. No value this module
reads is written back, and no allocation decision is remade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from src.geometry import cell_center
from src.map_structure import Cell


@dataclass(eq=False)
class HierarchyNode:
    """One node of the 3-level FOVEA quadtree (coarse -> medium -> fine).

    Leaf nodes (`is_leaf=True`) correspond 1:1 to an existing engine `Cell`
    -- `cell` IS that exact object, untouched, and `point_count` is copied
    from it. Internal nodes (`is_leaf=False`, `cell=None`) are STRUCTURAL
    ONLY: they exist because the flat engine refined that footprint further
    and therefore never emitted a Cell for it directly. An internal node's
    `point_count` is always the sum of its children's `point_count`
    (enforced in `build_fovea_hierarchy`, checked in
    `verify_point_conservation`) -- so a point is represented at exactly one
    LEAF, never independently counted again by an ancestor.
    """

    level: int                      # 0=coarse (band base size), 1=medium (base/2), 2=fine (base/refinement_factor)
    band_id: int
    ix: int
    iy: int
    cell_size_m: float
    center_x: float
    center_y: float
    range_m: float
    is_leaf: bool
    cell: Optional[Cell]             # populated iff is_leaf -- the real engine Cell, never duplicated on an ancestor
    point_count: int                 # own points if leaf; sum of children's point_count if internal
    children: list["HierarchyNode"] = field(default_factory=list)
    parent: Optional["HierarchyNode"] = None

    @property
    def key(self) -> tuple[int, int, int, int]:
        """(level, band_id, ix, iy) -- a pure function of physical footprint
        + LOD, same uniqueness convention as Cell's own (cell_size, ix, iy)
        key in map_structure.py."""
        return (self.level, self.band_id, self.ix, self.iy)


def _node_key(level: int, band_id: int, ix: int, iy: int) -> tuple[int, int, int, int]:
    return (level, band_id, ix, iy)


def build_fovea_hierarchy(
    cells: list[Cell], accounting: dict, cfg: dict
) -> tuple[list[HierarchyNode], "HierarchyStats"]:
    """Builds the explicit parent/child tree from an EXISTING, unmodified
    `build_fovea_map(...)` result. Performs no allocation math of its own.

    Returns (roots, stats). `roots` is a FOREST of level-0 (coarse) nodes --
    one per (band, coarse-cell) footprint that has at least one point -- since
    FOVEA has no single map-wide root: each band's own distance-adaptive
    coarse cell is independently the top of its own subtree (this mirrors
    the engine itself, which makes the split-to-medium decision independently
    per coarse group -- see src/fovea_allocator.py Step 2).

    Raises AssertionError if point conservation does not hold (see
    `verify_point_conservation`) -- this is a regression guard on an
    invariant this function's own construction guarantees when correct, not
    a check on the caller's data.
    """
    ox, oy = float(cfg["origin"]["x"]), float(cfg["origin"]["y"])
    factor = int(cfg["fovea"]["refinement_factor"])  # READ-ONLY: recovers the engine's own branching ratio, changes nothing
    half = factor // 2
    # Ratio to divide a node's own (ix, iy) by to reach its PARENT's (ix, iy)
    # one level up -- mirrors src/fovea_allocator.py's own
    # `fc_ix // factor` (finest -> coarse) / `fc_ix // half` (finest ->
    # medium) index arithmetic exactly, just expressed as one hop at a time.
    parent_ratio = {2: half, 1: factor // half}

    nodes: dict[tuple, HierarchyNode] = {}

    # Pass 1: create every leaf (from the real Cells) and walk each one
    # upward, synthesizing structural ancestor nodes and parent/child links
    # as needed. Internal nodes' point_count is NOT finalized here -- see
    # Pass 2 below for why an incremental running sum during this walk would
    # under-count a node visited before all of its children exist.
    for cell in cells:
        leaf_key = _node_key(cell.resolution_level, cell.band_id, cell.ix, cell.iy)
        leaf = nodes.get(leaf_key)
        if leaf is None:
            leaf = HierarchyNode(
                level=cell.resolution_level, band_id=cell.band_id, ix=cell.ix, iy=cell.iy,
                cell_size_m=cell.cell_size_m, center_x=cell.center_x, center_y=cell.center_y,
                range_m=cell.range_m, is_leaf=True, cell=cell, point_count=cell.point_count,
            )
            nodes[leaf_key] = leaf

        child = leaf
        cur_level, cur_ix, cur_iy, cur_cs = cell.resolution_level, cell.ix, cell.iy, cell.cell_size_m
        while cur_level > 0:
            ratio = parent_ratio[cur_level]
            parent_level = cur_level - 1
            parent_ix, parent_iy = cur_ix // ratio, cur_iy // ratio
            parent_cs = cur_cs * ratio
            parent_key = _node_key(parent_level, cell.band_id, parent_ix, parent_iy)
            parent = nodes.get(parent_key)
            if parent is None:
                cx = float(cell_center(np.array([parent_ix], dtype=np.float64), ox, parent_cs)[0])
                cy = float(cell_center(np.array([parent_iy], dtype=np.float64), oy, parent_cs)[0])
                parent = HierarchyNode(
                    level=parent_level, band_id=cell.band_id, ix=parent_ix, iy=parent_iy,
                    cell_size_m=parent_cs, center_x=cx, center_y=cy, range_m=float(np.hypot(cx, cy)),
                    is_leaf=False, cell=None, point_count=0,
                )
                nodes[parent_key] = parent
            if child.parent is None:
                child.parent = parent
                parent.children.append(child)
            child = parent
            cur_level, cur_ix, cur_iy, cur_cs = parent_level, parent_ix, parent_iy, parent_cs

    # Pass 2: bottom-up point_count aggregation for internal nodes, run only
    # AFTER every leaf/parent link exists. Doing this incrementally during
    # Pass 1 (adding a child's count to its parent the moment the link is
    # made) would under-count any ancestor that gets a LATER sibling
    # attached afterwards -- e.g. two fine leaves sharing one medium parent:
    # the first leaf's walk would set the coarse grandparent's count from
    # the medium parent's count-so-far (one leaf), and never revisit it once
    # the second leaf later adds to the medium parent. Processing strictly
    # level-by-level, finest-internal-level first, avoids that.
    for level in (1, 0):
        for node in nodes.values():
            if node.level == level and not node.is_leaf:
                node.point_count = sum(c.point_count for c in node.children)

    roots = [node for node in nodes.values() if node.level == 0]

    conservation = verify_point_conservation(roots, expected_total=accounting.get("assigned_points"))
    assert conservation["conserved"], (
        f"FOVEA hierarchy point conservation violated: {conservation}"
    )

    stats = summarize_hierarchy(list(nodes.values()), roots, conservation)
    return roots, stats


def iter_nodes(roots: list[HierarchyNode]) -> Iterator[HierarchyNode]:
    """Preorder traversal of the whole forest (every level, leaf and internal)."""
    stack = list(roots)
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


def leaf_nodes(roots: list[HierarchyNode]) -> list[HierarchyNode]:
    """All leaf nodes (the ones carrying a real Cell) across the forest."""
    return [n for n in iter_nodes(roots) if n.is_leaf]


def verify_point_conservation(roots: list[HierarchyNode], expected_total: Optional[int] = None) -> dict:
    """Confirms the point-conservation invariant `build_fovea_hierarchy`'s
    construction is SUPPOSED to guarantee: every internal node's
    `point_count` exactly equals the sum of its children's `point_count`,
    and the whole forest's total equals the flat engine's own
    `accounting["assigned_points"]` when supplied. A real re-check of real
    numbers, not a fabricated pass/fail -- mirrors the spirit of
    `src.spatial.map_2_5d.validate_point_conservation`.
    """
    mismatched = []
    for node in iter_nodes(roots):
        if node.children:
            child_sum = sum(c.point_count for c in node.children)
            if child_sum != node.point_count:
                mismatched.append({"key": node.key, "node_point_count": node.point_count, "children_sum": child_sum})

    root_total = sum(r.point_count for r in roots)
    result = {
        "root_total_points": root_total,
        "mismatched_node_count": len(mismatched),
        "mismatched_nodes": mismatched[:20],  # capped for readability; count above is the real total
        "internal_conservation_ok": len(mismatched) == 0,
    }
    if expected_total is not None:
        result["expected_total_points"] = int(expected_total)
        result["matches_expected_total"] = root_total == int(expected_total)
    result["conserved"] = result["internal_conservation_ok"] and result.get("matches_expected_total", True)
    return result


@dataclass
class HierarchyStats:
    """Tree-level summary -- every number is a real count over the actual
    nodes built, nothing fabricated (same convention as Map2_5DStats)."""

    total_nodes: int
    leaf_nodes: int
    internal_nodes: int
    root_nodes: int
    nodes_by_level: dict[int, int]
    max_depth: int              # 1..3 -- number of distinct levels actually present in this forest
    root_total_points: int
    expected_total_points: Optional[int]
    point_conservation_ok: bool


def summarize_hierarchy(all_nodes: list[HierarchyNode], roots: list[HierarchyNode], conservation: dict) -> HierarchyStats:
    nodes_by_level: dict[int, int] = {}
    for n in all_nodes:
        nodes_by_level[n.level] = nodes_by_level.get(n.level, 0) + 1
    leaves = sum(1 for n in all_nodes if n.is_leaf)
    return HierarchyStats(
        total_nodes=len(all_nodes),
        leaf_nodes=leaves,
        internal_nodes=len(all_nodes) - leaves,
        root_nodes=len(roots),
        nodes_by_level=dict(sorted(nodes_by_level.items())),
        max_depth=(max(nodes_by_level.keys()) + 1) if nodes_by_level else 0,
        root_total_points=conservation["root_total_points"],
        expected_total_points=conservation.get("expected_total_points"),
        point_conservation_ok=conservation["conserved"],
    )
