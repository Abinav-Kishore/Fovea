"""FOVEA v0.2 LOD (level-of-detail) selection layer (Phase 3B).

This module adds NO new spatial allocation, scoring, or resolution-decision
logic -- it reads an EXISTING, unmodified hierarchy forest (the `roots`
returned by `src.spatial.hierarchy.build_fovea_hierarchy`, itself built from
an unmodified `src.fovea_allocator.build_fovea_map` call) and, given a
viewpoint, selects a NON-OVERLAPPING "render frontier": a set of
`HierarchyNode`s such that no selected node is an ancestor or descendant of
another selected node, together covering every non-culled root's footprint
exactly once.

This is a VIEWING decision, not a re-allocation. FOVEA's own
information_score / semantic_priority / weights / split thresholds /
distance bands / refinement_factor already decided, once, what resolution
levels EXIST in the hierarchy (src/fovea_allocator.py, unmodified). This
module never recomputes or overrides that -- it only picks, among the levels
that already exist for a given footprint, which one is worth rendering from
a particular viewpoint right now. Nothing here is written back onto a Cell
or HierarchyNode.

Distance metric: planar Euclidean distance from the viewpoint (x, y) to a
node's (center_x, center_y) -- the SAME planar convention map_structure.py's
own `Cell.range_m` already uses ("range of cell center from origin", see its
docstring), just measured from an arbitrary viewpoint instead of the fixed
sensor origin. No new distance convention is introduced.

Core near/far rule (apparent-size heuristic, independent of FOVEA's own
distance bands):

    detail_ratio = node.cell_size_m / max(distance_to_viewpoint, min_distance_m)
    descend to children iff node.children and detail_ratio > detail_threshold

A node close to the viewpoint has a small denominator -> large detail_ratio
-> keeps descending toward its finest available children ("near -> prefer
finer descendants"). A node far from the viewpoint has a large denominator
-> small detail_ratio -> stops descending and is selected as-is ("far ->
prefer coarser ancestors"). A node with no children (already the finest
FOVEA produced for that footprint) is always selected outright regardless of
detail_ratio ("leaf fallback") -- there is nothing finer to descend into.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.spatial.hierarchy import HierarchyNode


@dataclass
class LODSelectionConfig:
    """Every parameter here is independent of, and never overrides, FOVEA's
    own allocation config (`cfg["fovea"]`) -- this is a separate, purely
    view-dependent selection policy layered on top of an already-built
    hierarchy."""

    detail_threshold: float = 0.5
    # Cell looks "too big" (cell_size_m / distance > this) -> descend to children.
    # Smaller values refine more aggressively at any given distance.

    max_distance_m: Optional[float] = None
    # Nodes (and their entire subtree) farther than this from the viewpoint
    # are culled outright -- not selected, not descended into. None = no culling.

    min_distance_m: float = 1e-6
    # Floor on the distance used in detail_ratio, so a viewpoint sitting
    # exactly on a node's center never divides by zero.

    point_budget: Optional[int] = None
    # Optional simple cap on sum(node.point_count for node in frontier). If the
    # initial (detail_threshold + max_distance_m) frontier exceeds this, the
    # FARTHEST-from-viewpoint frontier nodes are dropped (excluded from the
    # frontier entirely, deterministically, farthest first) until the total
    # fits or nothing is left. This is deliberately simple prioritized
    # culling, NOT resolution/LOD-level renegotiation: a dropped node's
    # points are just not rendered this frame, its resolution level is never
    # changed. None = no budget enforcement.


@dataclass
class LODSelection:
    """Result of one `select_lod_nodes` call. Every count is a real tally
    over the actual `nodes` returned, nothing fabricated (same convention as
    `HierarchyStats`/`Map2_5DStats`)."""

    nodes: list[HierarchyNode]        # deterministic, non-overlapping render frontier
    viewpoint: tuple[float, float]
    config: LODSelectionConfig
    total_point_count: int
    node_count: int
    nodes_by_level: dict[int, int]
    root_count: int
    culled_root_count: int            # whole roots excluded outright by max_distance_m
    culled_subtree_count: int         # non-root subtrees culled by max_distance_m during descent
    budget_dropped_node_count: int    # frontier nodes removed by point_budget enforcement
    latency_s: float                  # THIS SELECTION ONLY -- excludes FOVEA/hierarchy build time


def _distance(node: HierarchyNode, vx: float, vy: float) -> float:
    return float(np.hypot(node.center_x - vx, node.center_y - vy))


def select_lod_nodes(
    roots: list[HierarchyNode],
    viewpoint: tuple[float, float],
    config: Optional[LODSelectionConfig] = None,
) -> LODSelection:
    """Selects a non-overlapping render frontier from an existing FOVEA
    hierarchy forest. Reads `HierarchyNode.cell_size_m/center_x/center_y/
    children/point_count` only -- never mutates a node, never touches a
    Cell's information_score/semantic_priority/weights, never re-derives a
    band or the refinement_factor.

    `roots` must come from an existing `build_fovea_hierarchy(...)` call
    (or be `[]`). `viewpoint` is `(x, y)` in the same frame as
    `HierarchyNode.center_x/center_y` (sensor/vehicle origin frame, per
    config.yaml's `origin`).
    """
    config = config or LODSelectionConfig()
    t0 = time.perf_counter()
    vx, vy = float(viewpoint[0]), float(viewpoint[1])

    frontier: list[HierarchyNode] = []
    culled_root_count = 0
    culled_subtree_count = 0

    def visit(node: HierarchyNode, is_root: bool) -> None:
        nonlocal culled_root_count, culled_subtree_count
        d = _distance(node, vx, vy)
        if config.max_distance_m is not None and d > config.max_distance_m:
            if is_root:
                culled_root_count += 1
            else:
                culled_subtree_count += 1
            return

        if node.children:
            detail_ratio = node.cell_size_m / max(d, config.min_distance_m)
            if detail_ratio > config.detail_threshold:
                for child in node.children:
                    visit(child, is_root=False)
                return

        # Either a leaf (nothing finer exists -- "leaf fallback") or the
        # viewpoint is far/coarse enough that this level is good enough.
        frontier.append(node)

    for root in roots:
        visit(root, is_root=True)

    frontier.sort(key=lambda n: n.key)  # deterministic regardless of dict/traversal order

    budget_dropped_node_count = 0
    if config.point_budget is not None:
        total = sum(n.point_count for n in frontier)
        if total > config.point_budget:
            # Deterministic priority: farthest-from-viewpoint first, tie-broken
            # by the node's own (level, band_id, ix, iy) key.
            drop_order = sorted(frontier, key=lambda n: (-_distance(n, vx, vy), n.key))
            dropped_ids: set[int] = set()
            running = total
            i = 0
            while running > config.point_budget and i < len(drop_order):
                n = drop_order[i]
                dropped_ids.add(id(n))
                running -= n.point_count
                budget_dropped_node_count += 1
                i += 1
            frontier = [n for n in frontier if id(n) not in dropped_ids]

    nodes_by_level: dict[int, int] = {}
    for n in frontier:
        nodes_by_level[n.level] = nodes_by_level.get(n.level, 0) + 1

    return LODSelection(
        nodes=frontier,
        viewpoint=(vx, vy),
        config=config,
        total_point_count=sum(n.point_count for n in frontier),
        node_count=len(frontier),
        nodes_by_level=dict(sorted(nodes_by_level.items())),
        root_count=len(roots),
        culled_root_count=culled_root_count,
        culled_subtree_count=culled_subtree_count,
        budget_dropped_node_count=budget_dropped_node_count,
        latency_s=time.perf_counter() - t0,
    )


def verify_lod_frontier(nodes: list[HierarchyNode]) -> dict:
    """Independent structural check of a candidate frontier -- does NOT trust
    that it came from `select_lod_nodes`. Confirms (a) no footprint is
    selected twice and (b) no selected node is an ancestor of another
    selected node (checked by walking each node's `.parent` chain, which
    catches every such pair from the descendant's side). A real
    re-derivation from the actual node objects, not a fabricated pass/fail
    -- same convention as `src.spatial.hierarchy.verify_point_conservation`.
    """
    seen_keys: set[tuple] = set()
    duplicate_keys: list[tuple] = []
    for n in nodes:
        if n.key in seen_keys:
            duplicate_keys.append(n.key)
        seen_keys.add(n.key)

    id_set = {id(n) for n in nodes}
    violations = []
    for n in nodes:
        p = n.parent
        while p is not None:
            if id(p) in id_set:
                violations.append({"descendant": n.key, "ancestor": p.key})
                break
            p = p.parent

    return {
        "node_count": len(nodes),
        "duplicate_keys": duplicate_keys,
        "ancestor_descendant_violations": violations,
        "no_duplicates": len(duplicate_keys) == 0,
        "no_ancestor_descendant_overlap": len(violations) == 0,
        "valid": len(duplicate_keys) == 0 and len(violations) == 0,
    }
