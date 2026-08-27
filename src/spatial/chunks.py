"""FOVEA v0.2 streamable spatial chunk / serialization layer (Phase 3C).

This module adds NO new FOVEA allocation, hierarchy-construction, or LOD-
selection logic. Pipeline:

    existing hierarchy (src/spatial/hierarchy.py, unmodified)
        v
    existing select_lod_nodes() (src/spatial/lod.py, unmodified)
        v
    selected HierarchyNodes (the render frontier)
        v
    SpatialChunk (THIS module) -- bounds/center/resolution/point_count/
    elevation/semantic/terrain/obstacle/information summary, NO raw points
        v
    serialize_chunk(...) -> bytes   (binary, see "Binary format" below)
    deserialize_chunk(bytes) -> list[SpatialChunk]   (exact inverse)
        v
    HTTP response (src/api.py)

This layer never decides FOVEA resolution and never re-runs or overrides
LOD selection -- it only packages whatever `select_lod_nodes` already chose.

Where a chunk's numbers come from
----------------------------------
Every SpatialChunk field is either copied directly from an existing
HierarchyNode, a trivial geometric derivation (bounds = center +/- half
cell size, same convention as `src.spatial.map_2_5d.PerceptionCell`), or --
for a node with NO Cell of its own (an internal/ancestor node the LOD layer
chose to render as one coarser whole; see hierarchy.py's docstring on why
such nodes exist) -- a POINT-COUNT-WEIGHTED AGGREGATE over that node's own
leaf descendants' already-computed Cell fields (`src.spatial.hierarchy.
leaf_nodes`). This is never a re-read of raw points and never a
re-derivation of information_score/semantic_priority/weights/thresholds/
distance bands/refinement_factor -- see `_build_chunk` below. For a leaf
node (the common case) this reduces to exactly that Cell's own values,
unchanged. `SpatialChunk.is_leaf` tells a consumer which case it got: True
= an exact single-Cell measurement, False = a weighted aggregate over
N>=1 leaf Cells the LOD layer folded into one coarser chunk.

Binary format ("FVCH1")
------------------------
Chosen over one-JSON-object-per-chunk because JSON repeats every field
NAME for every chunk -- tens of thousands of chunks x ~25 named floats/ints
is a multi-megabyte payload for numbers that are 4-8 bytes each on the
wire -- and because six string-valued fields (semantic_class, terrain_class,
obstacle_class, resolution_label, label_source, source) repeat almost
verbatim across nearly every chunk (a handful of distinct semantic classes
across tens of thousands of cells). FVCH1 fixes both: fixed-width
struct-packed numeric records (`_CHUNK_STRUCT`) plus ONE interned string
table shared by the whole chunk set (each record stores 6 small integer
indices into it, never the strings themselves).

Numeric fields use float32 (not float64) -- a documented size/precision
tradeoff (~7 significant decimal digits), the same choice glTF/Three.js
vertex buffers make, appropriate because these are rendering/inspection
values, never fed back into further FOVEA computation. `point_count`,
`ix`, `iy`, `band_id`, `level`, `is_leaf` are integer-typed and round-trip
EXACTLY; string fields round-trip EXACTLY (interned, not re-encoded lossily).
See `verify_chunk_roundtrip` for a same-process round-trip check with the
appropriate (integer-exact / float-tolerant) comparison per field.

Layout: 6-byte magic b"FVCH1\\x00" | uint32 chunk_count | uint32
string_count | string_count x (uint16 length + UTF-8 bytes) | chunk_count x
`_CHUNK_STRUCT` records.

If this ever proves too little: the JSON-metadata-first fallback the task
allows for is simply `[dataclasses.asdict(c) for c in chunks]` -- every
SpatialChunk field is already JSON-safe (str/int/float/bool) -- kept as a
documented option, not implemented, since FVCH1 was not "excessively
complex" to build and measurably smaller (see README / benchmark script).
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass

import numpy as np

from src.spatial.hierarchy import HierarchyNode, leaf_nodes
from src.spatial.obstacle import classify_obstacle
from src.spatial.terrain import classify_terrain
from src.theme import LEVEL_LABELS


@dataclass
class SpatialChunk:
    """The minimum a renderer/inspector needs for ONE selected LOD node --
    no raw point arrays. See module docstring for where each field comes
    from and what `is_leaf=False` means for the aggregated ones."""

    node_id: str  # deterministic "level:band_id:ix:iy" -- derivable, not a persistent cross-frame id
    level: int
    resolution_label: str
    is_leaf: bool

    band_id: int
    ix: int
    iy: int
    cell_size_m: float

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    center_x: float
    center_y: float
    range_m: float

    point_count: int

    elevation_min: float
    elevation_max: float
    elevation_median: float  # point-count-weighted mean across leaf descendants when is_leaf=False

    semantic_class: str      # point-count-weighted majority vote across leaf descendants when is_leaf=False
    semantic_priority: float
    majority_class_fraction: float
    geometry_score: float
    dynamic_score: float
    information_score: float

    terrain_class: str
    obstacle_class: str

    label_source: str
    source: str


@dataclass
class ChunkMetadata:
    """Summary of one whole chunk SET (one `select_lod_nodes` call's worth),
    not one chunk -- what an API response needs alongside the payload."""

    dataset: str
    sequence: str
    frame_id: str
    label_source: str
    source: str
    chunk_count: int
    selected_point_count: int
    payload_format: str
    payload_bytes: int
    serialization_latency_s: float


def node_id(node: HierarchyNode) -> str:
    level, band_id, ix, iy = node.key
    return f"{level}:{band_id}:{ix}:{iy}"


def _build_chunk(node: HierarchyNode, label_source: str, source: str) -> SpatialChunk:
    leaves = leaf_nodes([node])  # [node] itself if it's already a leaf
    if not leaves:
        # Structurally unreachable: build_fovea_hierarchy never creates a
        # node without at least one leaf descendant (every internal node is
        # only ever synthesized as the ancestor of some real leaf Cell) --
        # kept as an explicit guard rather than a silent empty-aggregate.
        raise ValueError(f"hierarchy node {node.key} has no leaf descendants -- cannot build a chunk from it")

    total_points = sum(leaf.point_count for leaf in leaves)
    assert total_points == node.point_count, (
        f"chunk aggregation mismatch at {node.key}: leaves sum to {total_points}, "
        f"node.point_count is {node.point_count} -- hierarchy point conservation violated"
    )

    if len(leaves) == 1:
        # Fast path for the overwhelmingly common case (a chunk that IS a
        # single leaf Cell, no aggregation needed) -- skips the weighted-
        # average machinery below entirely; the result is identical to it
        # (weight of 1.0 on a single value), just without the per-chunk
        # numpy array overhead, which matters at tens of thousands of chunks.
        c0 = leaves[0].cell
        elevation_min, elevation_max, elevation_median = c0.elevation_min, c0.elevation_max, c0.elevation_median
        semantic_priority, geometry_score = c0.semantic_priority, c0.geometry_score
        dynamic_score, information_score = c0.dynamic_score, c0.information_score
        semantic_class, majority_class_fraction = c0.majority_class_name, c0.majority_class_fraction
    elif total_points > 0:
        w = np.array([leaf.point_count for leaf in leaves], dtype=np.float64)
        elevation_median = float(np.average([leaf.cell.elevation_median for leaf in leaves], weights=w))
        semantic_priority = float(np.average([leaf.cell.semantic_priority for leaf in leaves], weights=w))
        geometry_score = float(np.average([leaf.cell.geometry_score for leaf in leaves], weights=w))
        dynamic_score = float(np.average([leaf.cell.dynamic_score for leaf in leaves], weights=w))
        information_score = float(np.average([leaf.cell.information_score for leaf in leaves], weights=w))
        elevation_min = min(leaf.cell.elevation_min for leaf in leaves)
        elevation_max = max(leaf.cell.elevation_max for leaf in leaves)

        class_points: dict[str, float] = {}
        for leaf in leaves:
            c = leaf.cell
            class_points[c.majority_class_name] = (
                class_points.get(c.majority_class_name, 0.0) + c.point_count * c.majority_class_fraction
            )
        semantic_class = max(class_points, key=class_points.get)
        majority_class_fraction = class_points[semantic_class] / total_points
    else:
        # Never actually reachable -- every leaf HierarchyNode wraps a real
        # Cell, and map_structure.py only ever creates a Cell for an
        # occupied (point_count >= 1) group -- kept explicit rather than a
        # silent divide-by-zero if that invariant is ever violated upstream.
        elevation_min = elevation_max = elevation_median = 0.0
        semantic_priority = geometry_score = dynamic_score = information_score = 0.0
        semantic_class, majority_class_fraction = "unknown", 0.0

    terrain = classify_terrain(semantic_class, source)
    obstacle = classify_obstacle(terrain, dynamic_score)

    half = node.cell_size_m / 2.0
    return SpatialChunk(
        node_id=node_id(node),
        level=node.level,
        resolution_label=LEVEL_LABELS.get(node.level, str(node.level)),
        is_leaf=node.is_leaf,
        band_id=node.band_id,
        ix=node.ix,
        iy=node.iy,
        cell_size_m=node.cell_size_m,
        x_min=node.center_x - half,
        x_max=node.center_x + half,
        y_min=node.center_y - half,
        y_max=node.center_y + half,
        center_x=node.center_x,
        center_y=node.center_y,
        range_m=node.range_m,
        point_count=node.point_count,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        elevation_median=elevation_median,
        semantic_class=semantic_class,
        semantic_priority=semantic_priority,
        majority_class_fraction=majority_class_fraction,
        geometry_score=geometry_score,
        dynamic_score=dynamic_score,
        information_score=information_score,
        terrain_class=terrain.value,
        obstacle_class=obstacle.value,
        label_source=label_source,
        source=source,
    )


def build_chunks(nodes: list[HierarchyNode], label_source: str, source: str) -> list[SpatialChunk]:
    """Turns an EXISTING LOD frontier (`LODSelection.nodes` from
    `select_lod_nodes`, unmodified) into one `SpatialChunk` per node, in the
    same order. `label_source`/`source` follow the same convention as
    `src.spatial.map_2_5d.build_2_5d_map` ("ground_truth"/dataset name, or
    "prediction"/model ontology)."""
    return [_build_chunk(n, label_source, source) for n in nodes]


# ---------------------------------------------------------------------------
# Binary serialization -- see module docstring for the format rationale.
# ---------------------------------------------------------------------------

_MAGIC = b"FVCH1\x00"

# level(B) is_leaf(B) band_id(i) ix(i) iy(i) cell_size_m(f)
# x_min x_max y_min y_max center_x center_y range_m (f x7)
# point_count(I)
# elevation_min elevation_max elevation_median (f x3)
# semantic_priority majority_class_fraction geometry_score dynamic_score information_score (f x5)
# semantic_class_idx terrain_class_idx obstacle_class_idx resolution_label_idx label_source_idx source_idx (H x6)
_CHUNK_STRUCT = struct.Struct("<BBiiiffffffffIffffffffHHHHHH")


def serialize_chunk(chunks: list[SpatialChunk]) -> bytes:
    """Encodes a whole chunk set into one self-contained FVCH1 binary blob
    (decodable by `deserialize_chunk` alone, no external context needed)."""
    string_table: list[str] = []
    string_index: dict[str, int] = {}

    def intern(s: str) -> int:
        idx = string_index.get(s)
        if idx is None:
            idx = len(string_table)
            string_table.append(s)
            string_index[s] = idx
        return idx

    records = bytearray()
    for c in chunks:
        records += _CHUNK_STRUCT.pack(
            c.level,
            1 if c.is_leaf else 0,
            c.band_id,
            c.ix,
            c.iy,
            c.cell_size_m,
            c.x_min,
            c.x_max,
            c.y_min,
            c.y_max,
            c.center_x,
            c.center_y,
            c.range_m,
            c.point_count,
            c.elevation_min,
            c.elevation_max,
            c.elevation_median,
            c.semantic_priority,
            c.majority_class_fraction,
            c.geometry_score,
            c.dynamic_score,
            c.information_score,
            intern(c.semantic_class),
            intern(c.terrain_class),
            intern(c.obstacle_class),
            intern(c.resolution_label),
            intern(c.label_source),
            intern(c.source),
        )

    out = bytearray()
    out += _MAGIC
    out += struct.pack("<II", len(chunks), len(string_table))
    for s in string_table:
        b = s.encode("utf-8")
        if len(b) > 0xFFFF:
            raise ValueError(f"chunk string field too long to encode ({len(b)} bytes): {s[:50]!r}...")
        out += struct.pack("<H", len(b))
        out += b
    out += records
    return bytes(out)


def deserialize_chunk(data: bytes) -> list[SpatialChunk]:
    """Exact inverse of `serialize_chunk`. Raises `ValueError` on anything
    that isn't a well-formed FVCH1 payload -- wrong magic, truncated header/
    string table/records, a declared length that doesn't match the actual
    byte count, or a string-table index a record points at that doesn't
    exist. Never silently returns a partial or garbage chunk list."""
    if len(data) < len(_MAGIC) + 8:
        raise ValueError(f"corrupt chunk payload: too short to contain a header ({len(data)} bytes)")
    if data[: len(_MAGIC)] != _MAGIC:
        raise ValueError("corrupt chunk payload: missing FVCH1 magic header")

    offset = len(_MAGIC)
    chunk_count, string_count = struct.unpack_from("<II", data, offset)
    offset += 8

    string_table: list[str] = []
    for _ in range(string_count):
        if offset + 2 > len(data):
            raise ValueError("corrupt chunk payload: string table truncated (length prefix cut off)")
        (length,) = struct.unpack_from("<H", data, offset)
        offset += 2
        if offset + length > len(data):
            raise ValueError("corrupt chunk payload: string table truncated (string data cut off)")
        try:
            string_table.append(data[offset : offset + length].decode("utf-8"))
        except UnicodeDecodeError as e:
            raise ValueError(f"corrupt chunk payload: invalid UTF-8 in string table: {e}")
        offset += length

    expected_total = offset + chunk_count * _CHUNK_STRUCT.size
    if expected_total != len(data):
        raise ValueError(
            f"corrupt chunk payload: expected {expected_total} bytes for {chunk_count} chunk(s) "
            f"+ header/string-table, got {len(data)}"
        )

    def lookup(idx: int, field: str) -> str:
        if not (0 <= idx < len(string_table)):
            raise ValueError(f"corrupt chunk payload: {field} string index {idx} out of range (table size {len(string_table)})")
        return string_table[idx]

    chunks: list[SpatialChunk] = []
    for _ in range(chunk_count):
        (
            level, is_leaf, band_id, ix, iy, cell_size_m,
            x_min, x_max, y_min, y_max, center_x, center_y, range_m,
            point_count,
            elevation_min, elevation_max, elevation_median,
            semantic_priority, majority_class_fraction, geometry_score, dynamic_score, information_score,
            sem_idx, terrain_idx, obstacle_idx, reslabel_idx, labelsrc_idx, source_idx,
        ) = _CHUNK_STRUCT.unpack_from(data, offset)
        offset += _CHUNK_STRUCT.size

        chunks.append(
            SpatialChunk(
                node_id=f"{level}:{band_id}:{ix}:{iy}",
                level=level,
                resolution_label=lookup(reslabel_idx, "resolution_label"),
                is_leaf=bool(is_leaf),
                band_id=band_id,
                ix=ix,
                iy=iy,
                cell_size_m=cell_size_m,
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                center_x=center_x, center_y=center_y, range_m=range_m,
                point_count=point_count,
                elevation_min=elevation_min, elevation_max=elevation_max, elevation_median=elevation_median,
                semantic_class=lookup(sem_idx, "semantic_class"),
                semantic_priority=semantic_priority,
                majority_class_fraction=majority_class_fraction,
                geometry_score=geometry_score,
                dynamic_score=dynamic_score,
                information_score=information_score,
                terrain_class=lookup(terrain_idx, "terrain_class"),
                obstacle_class=lookup(obstacle_idx, "obstacle_class"),
                label_source=lookup(labelsrc_idx, "label_source"),
                source=lookup(source_idx, "source"),
            )
        )

    return chunks


def serialize_chunks_with_metadata(
    nodes: list[HierarchyNode], label_source: str, source: str, dataset: str, sequence: str, frame_id: str
) -> tuple[bytes, ChunkMetadata]:
    """Convenience wrapper the API layer uses: builds chunks from an
    existing LOD frontier, serializes them, and reports the metadata an HTTP
    response needs -- timing this stage ONLY (not FOVEA/hierarchy/LOD, which
    the caller already timed separately)."""
    t0 = time.perf_counter()
    chunks = build_chunks(nodes, label_source, source)
    payload = serialize_chunk(chunks)
    latency = time.perf_counter() - t0

    meta = ChunkMetadata(
        dataset=dataset,
        sequence=sequence,
        frame_id=frame_id,
        label_source=label_source,
        source=source,
        chunk_count=len(chunks),
        selected_point_count=sum(c.point_count for c in chunks),
        payload_format="FVCH1",
        payload_bytes=len(payload),
        serialization_latency_s=latency,
    )
    return payload, meta


# ---------------------------------------------------------------------------
# Independent round-trip verification -- mirrors the spirit of
# src.spatial.hierarchy.verify_point_conservation / src.spatial.lod.
# verify_lod_frontier: a real re-check of real numbers, not a rubber stamp.
# ---------------------------------------------------------------------------

_EXACT_FIELDS = (
    "node_id", "level", "resolution_label", "is_leaf", "band_id", "ix", "iy",
    "point_count", "semantic_class", "terrain_class", "obstacle_class", "label_source", "source",
)
_FLOAT_FIELDS = (
    "cell_size_m", "x_min", "x_max", "y_min", "y_max", "center_x", "center_y", "range_m",
    "elevation_min", "elevation_max", "elevation_median", "semantic_priority",
    "majority_class_fraction", "geometry_score", "dynamic_score", "information_score",
)


def verify_chunk_roundtrip(chunks: list[SpatialChunk], rtol: float = 1e-5, atol: float = 1e-4) -> dict:
    """Serializes `chunks`, deserializes the result, and compares every
    field against the original: integer/string fields must match EXACTLY,
    float32-carried fields within `rtol`/`atol` (see module docstring on why
    floats are float32 on the wire). Not just "did it not crash" -- a real
    per-field, per-chunk comparison."""
    payload = serialize_chunk(chunks)
    restored = deserialize_chunk(payload)

    mismatches = []
    if len(restored) != len(chunks):
        mismatches.append({"field": "chunk_count", "expected": len(chunks), "got": len(restored)})

    for i, (a, b) in enumerate(zip(chunks, restored)):
        for field in _EXACT_FIELDS:
            va, vb = getattr(a, field), getattr(b, field)
            if va != vb:
                mismatches.append({"chunk_index": i, "field": field, "expected": va, "got": vb})
        for field in _FLOAT_FIELDS:
            va, vb = getattr(a, field), getattr(b, field)
            if not np.isclose(va, vb, rtol=rtol, atol=atol):
                mismatches.append({"chunk_index": i, "field": field, "expected": va, "got": vb})

    return {
        "chunk_count": len(chunks),
        "payload_bytes": len(payload),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "roundtrip_ok": len(mismatches) == 0,
    }
