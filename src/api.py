"""FOVEA FastAPI backend.

This layer performs ZERO allocation/scoring logic of its own. Every number it
returns comes from the existing engine modules (uniform_baseline,
distance_baseline, fovea_allocator, map_structure, metrics, benchmark,
config) plus the dataset adapters in src/datasets/ (SemanticPOSS, RELLIS-3D).
It only does three jobs:

  1. Wire HTTP requests to the same functions src/app.py (Streamlit) already
     calls, with a small in-process cache so repeated requests for the same
     (dataset, sequence, frame_id) don't recompute all three maps.
  2. Resolve which dataset is active per-request and hand the engine the
     right label dictionary / semantic-priority table / dynamic-class set
     for it -- via src.datasets.get_adapter, never a special-cased branch.
  3. Convert internal dataclasses / plain dicts into JSON-safe Pydantic
     response models -- internal dataclasses are never returned directly.

Two URL families exist for the same underlying logic:
  - /api/datasets/{dataset}/...  (primary, works for SemanticPOSS or RELLIS-3D)
  - /api/...                     (legacy, SemanticPOSS-only, kept for backward
                                   compatibility -- same handlers, dataset
                                   fixed to "SemanticPOSS")
Both call the same internal `_*_response` functions -- there is exactly one
implementation, not a parallel one per dataset.

Run with:  uvicorn src.api:app --reload --port 8000
"""
from __future__ import annotations

import base64
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import load_config
from src.data_loader import Frame
from src.datasets import DatasetAdapter, available_datasets, get_adapter
from src.distance_baseline import build_distance_adaptive_map
from src.fovea_allocator import build_fovea_map, explain_allocation
from src.map_structure import Cell
from src.metrics import summarize_method
from src.perception.evaluation import evaluate_predictions
from src.perception.inference import run_perception
from src.perception.ontology import CANONICAL_CLASSES, map_names_to_canonical
from src.spatial import (
    build_2_5d_map,
    validate_cell_geometry,
    validate_no_duplicate_cells,
    validate_point_conservation,
)
from src.spatial.chunks import serialize_chunks_with_metadata
from src.spatial.hierarchy import build_fovea_hierarchy
from src.spatial.lod import LODSelectionConfig, select_lod_nodes
from src.theme import LEVEL_LABELS
from src.uniform_baseline import build_uniform_map

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FOVEA API",
    description="Adaptive variable-resolution 2.5D LiDAR mapping -- serves the "
    "existing Python FOVEA engine's real outputs (SemanticPOSS or RELLIS-3D) "
    "to the React frontend. Reuses the same functions as the Streamlit debug "
    "dashboard; computes nothing new.",
    version="0.2.0",
)

_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CFG = load_config("config.yaml")
_DEFAULT_DATASET = _CFG.get("datasets", {}).get("default", "SemanticPOSS")
_METHOD_FNS = {
    "uniform": build_uniform_map,
    "distance_adaptive": build_distance_adaptive_map,
    "fovea": build_fovea_map,
}

# ---------------------------------------------------------------------------
# Dataset resolution: one adapter + one "effective cfg" per dataset name,
# computed once and reused (config is static for the life of the process).
# The effective cfg is a copy of _CFG with fovea.semantic_priority /
# fovea.dynamic_classes swapped to that dataset's own table -- this is the
# ONLY place per-dataset config substitution happens; every engine function
# call below still just reads cfg["fovea"]["semantic_priority"] the exact
# same way it always has, so uniform_baseline.py / distance_baseline.py /
# fovea_allocator.py needed zero changes to their cfg-reading code.
# ---------------------------------------------------------------------------

_dataset_context_cache: dict[str, tuple[DatasetAdapter, dict]] = {}
_dataset_context_guard = threading.Lock()


def _dataset_context(dataset: str) -> tuple[DatasetAdapter, dict]:
    if dataset not in available_datasets():
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dataset '{dataset}'. Available: {available_datasets()}",
        )
    if dataset in _dataset_context_cache:
        return _dataset_context_cache[dataset]
    with _dataset_context_guard:
        if dataset in _dataset_context_cache:
            return _dataset_context_cache[dataset]
        import copy

        adapter = get_adapter(dataset, _CFG)
        effective_cfg = copy.deepcopy(_CFG)
        effective_cfg["fovea"]["semantic_priority"] = adapter.semantic_priority_table()
        effective_cfg["fovea"]["dynamic_classes"] = list(adapter.dynamic_classes())
        _dataset_context_cache[dataset] = (adapter, effective_cfg)
        return adapter, effective_cfg


# ---------------------------------------------------------------------------
# In-process cache: (dataset, sequence, frame_id) -> {"frame": Frame, "maps": {...}}
# Bounded FIFO eviction -- simple and sufficient for a single-user local demo.
# Cache keys MUST include dataset, or SemanticPOSS and RELLIS-3D frames that
# happen to share a (sequence, frame_id) string would collide and serve each
# other's data.
#
# LOCKING: FastAPI runs plain `def` endpoints in a thread pool, so multiple
# HTTP requests genuinely run concurrently. The frontend fires 5 requests at
# once for a single frame, and React StrictMode's dev-mode double-effect can
# double that again. Without a lock, concurrent requests for the same key
# would each independently reload the frame and rebuild all three maps -- a
# lock per cache key makes them coalesce: the first caller computes, the
# rest wait on the lock and reuse the result already in the cache. This is
# unchanged from the prior single-dataset implementation, just re-keyed.
# ---------------------------------------------------------------------------

_CACHE_MAX_FRAMES = 8  # slightly higher now that two datasets share the cache
CacheKey = tuple[str, str, str]
_frame_cache: dict[CacheKey, dict[str, Any]] = {}
_frame_cache_order: list[CacheKey] = []
_frame_locks: dict[CacheKey, threading.Lock] = {}
_locks_guard = threading.Lock()  # only guards creation of per-key locks themselves (fast, uncontended)


def _normalize_frame_id(adapter: DatasetAdapter, sequence: str, frame_id: str) -> str:
    try:
        ids = adapter.list_frames(sequence)
    except FileNotFoundError as e:
        # A sequence id valid for one dataset (e.g. SemanticPOSS's "00") is
        # not valid for another (RELLIS-3D's are "00000".."00004") -- this is
        # a normal 404, not a server error, and must be reported as such
        # rather than propagating as an unhandled exception (which a browser
        # sees as a hard connection failure, not a clean HTTP error).
        raise HTTPException(status_code=404, detail=str(e))
    if frame_id in ids:
        return frame_id
    if frame_id.isdigit():
        padded = f"{int(frame_id):06d}"
        if padded in ids:
            return padded
    raise HTTPException(
        status_code=404,
        detail=f"Frame '{frame_id}' not found in {adapter.name} sequence '{sequence}'.",
    )


def _get_frame_entry(dataset: str, sequence: str, frame_id: str) -> dict[str, Any]:
    adapter, _ = _dataset_context(dataset)
    frame_id = _normalize_frame_id(adapter, sequence, frame_id)
    key: CacheKey = (dataset, sequence, frame_id)

    if key in _frame_cache:  # fast path -- no locking needed once warm
        return _frame_cache[key]

    with _locks_guard:
        lock = _frame_locks.setdefault(key, threading.Lock())

    with lock:
        if key in _frame_cache:  # another thread may have just finished loading it
            return _frame_cache[key]

        try:
            frame = adapter.load_frame(sequence, frame_id)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))

        entry: dict[str, Any] = {
            "frame": frame,
            "maps": {},
            "map_locks": {method: threading.Lock() for method in _METHOD_FNS},
            "hierarchy_lock": threading.Lock(),
        }
        _frame_cache[key] = entry
        _frame_cache_order.append(key)
        if len(_frame_cache_order) > _CACHE_MAX_FRAMES:
            oldest = _frame_cache_order.pop(0)
            _frame_cache.pop(oldest, None)
            _frame_locks.pop(oldest, None)
        return entry


def _get_map(dataset: str, sequence: str, frame_id: str, method: str) -> tuple[list[Cell], dict, dict, Frame]:
    """Returns (cells, accounting, summary, frame). `summary` is cached
    alongside cells/accounting -- deterministic given them, so recomputing it
    on every request is pure waste once the map itself is cached."""
    if method not in _METHOD_FNS:
        raise HTTPException(status_code=400, detail=f"Unknown method '{method}'. Use one of {list(_METHOD_FNS)}.")
    adapter, effective_cfg = _dataset_context(dataset)
    entry = _get_frame_entry(dataset, sequence, frame_id)

    if method in entry["maps"]:  # fast path
        cells, accounting, summary = entry["maps"][method]
        return cells, accounting, summary, entry["frame"]

    with entry["map_locks"][method]:
        if method not in entry["maps"]:  # re-check after acquiring the lock
            frame: Frame = entry["frame"]
            cells, accounting = _METHOD_FNS[method](
                frame.points, frame.semantic_labels, effective_cfg, label_dict=adapter.label_dict()
            )
            summary = summarize_method(cells, accounting, effective_cfg["distance_adaptive"]["bands"])
            entry["maps"][method] = (cells, accounting, summary)

    cells, accounting, summary = entry["maps"][method]
    return cells, accounting, summary, entry["frame"]


# ---------------------------------------------------------------------------
# Hierarchy + LOD + chunk serialization (src/spatial/hierarchy.py, lod.py,
# chunks.py -- all unmodified by this layer). Same cache-once-per-key /
# lock-per-key coalescing discipline as _get_map/_get_prediction_map above:
# the first request for a given key computes and caches, concurrent/later
# requests for the SAME key reuse the cached result rather than recomputing.
#
# Ground-truth FOVEA mode only -- the coarse/medium/fine hierarchy is
# specifically FOVEA's own 3-level structure (src/fovea_allocator.py), so
# this reuses the SAME cached `_get_map(..., "fovea")` result every other
# FOVEA-mode endpoint already uses. Nothing here recomputes it.
# ---------------------------------------------------------------------------

def _get_hierarchy(dataset: str, sequence: str, frame_id: str):
    """Returns (roots, hierarchy_stats, hierarchy_build_latency_s), built
    ONCE per frame from the existing, unmodified build_fovea_map output and
    cached alongside it. `hierarchy_build_latency_s` is measured the one
    time this actually runs and reused on cache hits -- same convention as
    accounting["latency_s"] elsewhere in this module (a real one-time
    measurement, not re-timed per request)."""
    _, effective_cfg = _dataset_context(dataset)
    entry = _get_frame_entry(dataset, sequence, frame_id)

    if "hierarchy" in entry:
        return entry["hierarchy"]

    with entry["hierarchy_lock"]:
        if "hierarchy" not in entry:
            cells, accounting, _, _ = _get_map(dataset, sequence, frame_id, "fovea")
            t0 = time.perf_counter()
            roots, hstats = build_fovea_hierarchy(cells, accounting, effective_cfg)
            entry["hierarchy"] = (roots, hstats, time.perf_counter() - t0)

    return entry["hierarchy"]


def _get_chunk_payload(
    dataset: str,
    sequence: str,
    frame_id: str,
    detail_threshold: float,
    max_distance_m: Optional[float],
    point_budget: Optional[int],
    viewpoint_x: float,
    viewpoint_y: float,
):
    """Returns (selection, payload_bytes, chunk_metadata) for one exact LOD
    parameter combination, cached per (dataset, sequence, frame_id, *params)
    key -- concurrent/repeat requests with the SAME viewpoint/thresholds
    reuse the cached selection+payload instead of re-running select_lod_nodes
    or re-serializing. Different parameters are a DIFFERENT cache key
    (materially change the result), so they never collide."""
    entry = _get_frame_entry(dataset, sequence, frame_id)
    roots, _, _ = _get_hierarchy(dataset, sequence, frame_id)

    key = (
        round(float(detail_threshold), 6),
        None if max_distance_m is None else round(float(max_distance_m), 6),
        point_budget,
        round(float(viewpoint_x), 6),
        round(float(viewpoint_y), 6),
    )
    cache = entry.setdefault("chunk_cache", {})
    locks = entry.setdefault("chunk_locks", {})

    if key in cache:
        return cache[key]

    with _locks_guard:
        lock = locks.setdefault(key, threading.Lock())

    with lock:
        if key not in cache:
            lod_cfg = LODSelectionConfig(
                detail_threshold=detail_threshold, max_distance_m=max_distance_m, point_budget=point_budget
            )
            selection = select_lod_nodes(roots, viewpoint=(viewpoint_x, viewpoint_y), config=lod_cfg)
            payload, meta = serialize_chunks_with_metadata(
                selection.nodes, "ground_truth", dataset, dataset, sequence, frame_id
            )
            cache[key] = (selection, payload, meta)

    return cache[key]


# ---------------------------------------------------------------------------
# Perception (real AI inference, src/perception/) -- an INFERENCE MODE
# alongside the ground_truth mode above. Same frame cache, same per-key
# locking/coalescing discipline; only the source of `semantic_labels` fed
# into the (unmodified) engine functions differs: a model's predictions
# instead of a dataset's ground truth. Ground truth is never used as a
# prediction anywhere below -- see src/perception/base.py's module docstring.
# ---------------------------------------------------------------------------

_perception_context_cache: dict[str, tuple[Any, dict]] = {}
_perception_context_guard = threading.Lock()


def _perception_context(model_name: str) -> tuple[Any, dict]:
    """Returns (model_instance, effective_cfg) for `model_name`, memoized --
    loading a checkpoint takes real time, done once per process. effective_cfg
    is a copy of _CFG with fovea.semantic_priority/dynamic_classes swapped to
    this model's own ontology's table (config.yaml -> perception.models.*),
    the exact same substitution pattern _dataset_context uses for datasets."""
    models_cfg = _CFG.get("perception", {}).get("models", {})
    if model_name not in models_cfg:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown perception model '{model_name}'. Available: {list(models_cfg)}",
        )
    if model_name in _perception_context_cache:
        return _perception_context_cache[model_name]
    with _perception_context_guard:
        if model_name in _perception_context_cache:
            return _perception_context_cache[model_name]
        if model_name != "squeezeseg":
            raise HTTPException(status_code=501, detail=f"Model '{model_name}' has no loader implementation.")
        from src.perception.model import get_squeezeseg_model

        model = get_squeezeseg_model(_CFG)
        import copy

        effective_cfg = copy.deepcopy(_CFG)
        model_entry = models_cfg[model_name]
        effective_cfg["fovea"]["semantic_priority"] = model_entry["semantic_priority"]
        effective_cfg["fovea"]["dynamic_classes"] = list(model_entry["dynamic_classes"])
        _perception_context_cache[model_name] = (model, effective_cfg)
        return model, effective_cfg


def _get_prediction(dataset: str, sequence: str, frame_id: str, model_name: str):
    """Runs (and caches) real inference for one frame -- coalesced under
    concurrent requests exactly like _get_map coalesces map builds."""
    entry = _get_frame_entry(dataset, sequence, frame_id)
    cache = entry.setdefault("prediction", {})
    locks = entry.setdefault("prediction_locks", {})

    if model_name in cache:
        return cache[model_name]

    with _locks_guard:
        lock = locks.setdefault(model_name, threading.Lock())

    with lock:
        if model_name not in cache:
            model, _ = _perception_context(model_name)
            frame: Frame = entry["frame"]
            cache[model_name] = run_perception(model, frame.points)

    return cache[model_name]


def _get_prediction_map(dataset: str, sequence: str, frame_id: str, model_name: str, method: str):
    """FOVEA/uniform/distance-adaptive map built from PREDICTED labels
    instead of ground truth -- same engine functions, same label_dict
    plumbing added for RELLIS-3D, just fed the model's predictions and the
    model's own semantic-priority table (_perception_context) instead of a
    dataset's. Returns (cells, accounting, summary, frame, perception_result)."""
    if method not in _METHOD_FNS:
        raise HTTPException(status_code=400, detail=f"Unknown method '{method}'. Use one of {list(_METHOD_FNS)}.")
    entry = _get_frame_entry(dataset, sequence, frame_id)
    perception_result = _get_prediction(dataset, sequence, frame_id, model_name)
    model, model_cfg = _perception_context(model_name)

    maps_cache = entry.setdefault("prediction_maps", {})
    map_locks = entry.setdefault("prediction_map_locks", {})
    key = (model_name, method)

    if key in maps_cache:
        cells, accounting, summary = maps_cache[key]
        return cells, accounting, summary, entry["frame"], perception_result

    with _locks_guard:
        lock = map_locks.setdefault(key, threading.Lock())

    with lock:
        if key not in maps_cache:
            frame: Frame = entry["frame"]
            cells, accounting = _METHOD_FNS[method](
                frame.points, perception_result.predicted_labels, model_cfg, label_dict=model.raw_label_names
            )
            summary = summarize_method(cells, accounting, model_cfg["distance_adaptive"]["bands"])
            maps_cache[key] = (cells, accounting, summary)

    cells, accounting, summary = maps_cache[key]
    return cells, accounting, summary, entry["frame"], perception_result


# ---------------------------------------------------------------------------
# Pydantic response models -- the API boundary. No internal dataclass is ever
# returned directly.
# ---------------------------------------------------------------------------

class CellModel(BaseModel):
    """1:1 JSON-safe mirror of src.map_structure.Cell, plus a couple of
    display-only derived fields (index, resolution_label). Every value is
    read from the actual Cell instance -- nothing here is computed fresh."""
    index: int
    cell_size_m: float
    resolution_level: int
    resolution_label: str
    band_id: int
    ix: int
    iy: int
    center_x: float
    center_y: float
    point_count: int
    elevation_median: float
    elevation_std: float
    majority_class_id: int
    majority_class_name: str
    majority_class_fraction: float
    semantic_priority: float
    geometry_score: float
    dynamic_score: float
    uncertainty_score: Optional[float] = Field(
        default=None,
        description="Always null (N/A) -- neither SemanticPOSS nor RELLIS-3D provides a "
        "confidence field, so no value is invented. See weights_used.uncertainty for the "
        "configured weight (0.0 by default).",
    )
    information_score: float
    range_m: float
    weights_used: dict[str, float]


def _cell_to_model(cell: Cell, index: int) -> CellModel:
    return CellModel(
        index=index,
        cell_size_m=cell.cell_size_m,
        resolution_level=cell.resolution_level,
        resolution_label=LEVEL_LABELS.get(cell.resolution_level, str(cell.resolution_level)),
        band_id=cell.band_id,
        ix=cell.ix,
        iy=cell.iy,
        center_x=cell.center_x,
        center_y=cell.center_y,
        point_count=cell.point_count,
        elevation_median=cell.elevation_median,
        elevation_std=cell.elevation_std,
        majority_class_id=cell.majority_class_id,
        majority_class_name=cell.majority_class_name,
        majority_class_fraction=cell.majority_class_fraction,
        semantic_priority=cell.semantic_priority,
        geometry_score=cell.geometry_score,
        dynamic_score=cell.dynamic_score,
        uncertainty_score=None,  # never fabricated -- see field description
        information_score=cell.information_score,
        range_m=cell.range_m,
        weights_used=dict(cell.weights_used),
    )


class MapResponse(BaseModel):
    dataset: str
    sequence: str
    frame_id: str
    method: Literal["uniform", "distance_adaptive", "fovea"]
    cells: list[CellModel]
    accounting: dict[str, Any]
    summary: dict[str, Any]


class PointsResponse(BaseModel):
    dataset: str
    sequence: str
    frame_id: str
    total_points_in_frame: int
    returned_points: int
    subsampled: bool
    x: list[float]
    y: list[float]
    z: list[float]
    intensity: list[float]
    semantic_id: list[int]


class SemanticClassModel(BaseModel):
    class_id: int
    class_name: str
    semantic_priority: float
    is_dynamic_relevant: bool


class DatasetInfo(BaseModel):
    name: str
    is_default: bool


class StatusResponse(BaseModel):
    status: str
    dataset_root: str
    dataset_root_exists: bool
    available_sequences: list[str]
    available_datasets: list[str]
    default_dataset: str
    environment: dict[str, Any]
    cached_frames: list[str]


class ExplainResponse(BaseModel):
    dataset: str
    sequence: str
    frame_id: str
    method: str
    cell_index: int
    cell: CellModel
    explanation: str


class BenchmarkResponse(BaseModel):
    dataset: str
    available: bool
    source: str
    data: Optional[dict[str, Any]] = None
    message: Optional[str] = None


class PerceptionModelInfo(BaseModel):
    name: str
    ontology: str
    provides_confidence: bool
    description: str


class ClassCount(BaseModel):
    class_name: str
    point_count: int
    fraction: float


class PerceptionResponse(BaseModel):
    """Real inference summary -- deliberately no per-point arrays (the
    frontend already has the FOVEA map/points endpoints for that); this is
    the compact "what did the model do" view: timing, point count, and
    class distributions only."""

    dataset: str
    sequence: str
    frame_id: str
    model_name: str
    model_ontology: str
    num_points: int
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    total_perception_ms: float
    points_per_second: float
    provides_confidence: bool
    mean_confidence: Optional[float] = Field(
        default=None, description="Null if the model doesn't provide confidence -- never fabricated."
    )
    predicted_class_distribution: list[ClassCount]  # model's own raw ontology
    predicted_canonical_distribution: list[ClassCount]  # mapped through src/perception/ontology.py


class DistanceBandMetricsModel(BaseModel):
    band: str
    point_count: int
    accuracy: Optional[float] = None
    mean_iou: Optional[float] = None


class EvaluationResponse(BaseModel):
    """Predictions vs. ground truth, canonical-ontology space. `available`
    is false (with `message` explaining why) when the frame has no ground
    truth, or when the model structurally can't be scored meaningfully --
    never a fabricated number in either case."""

    dataset: str
    sequence: str
    frame_id: str
    model_name: str
    available: bool
    message: Optional[str] = None
    num_points_total: Optional[int] = None
    num_points_ignored: Optional[int] = None
    num_points_evaluated: Optional[int] = None
    overall_accuracy: Optional[float] = None
    mean_iou: Optional[float] = None
    per_class_iou: Optional[dict[str, Optional[float]]] = None
    distance_bands: Optional[dict[str, DistanceBandMetricsModel]] = None
    model_coverage: Optional[float] = Field(
        default=None,
        description="Fraction of evaluated ground-truth points whose canonical class the model "
        "can structurally ever predict -- low coverage means the domain/ontology mismatch, not "
        "just model weakness, explains a low score.",
    )
    excluded_canonical_classes: Optional[list[str]] = Field(
        default=None, description="Canonical classes present in ground truth that this model's ontology cannot express at all."
    )


class PerceptionCellModel(BaseModel):
    """1:1 mirror of src.spatial.PerceptionCell."""

    cell_id: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    center_x: float
    center_y: float
    cell_size_m: float
    resolution_level: int
    band_id: int
    range_m: float
    has_points: bool
    point_count: int
    elevation_median: Optional[float]
    elevation_min: Optional[float]
    elevation_max: Optional[float]
    elevation_std: Optional[float]
    semantic_class: str
    semantic_priority: float
    geometry_score: float
    dynamic_score: float
    information_score: float
    terrain_class: str
    obstacle_class: str
    label_source: str
    source: str


class Map25DStatsModel(BaseModel):
    """1:1 mirror of src.spatial.Map2_5DStats."""

    total_cells: int
    occupied_cells: int
    unknown_terrain_cells: int
    drivable_cells: int
    non_drivable_cells: int
    obstacle_cells: int
    static_obstacle_cells: int
    potential_dynamic_obstacle_cells: int
    mean_elevation_m: Optional[float]
    elevation_min_m: Optional[float]
    elevation_max_m: Optional[float]
    elevation_relief_std_m: Optional[float]
    input_points: int
    represented_points: int
    out_of_envelope_points: int
    invalid_points: int
    representation_coverage: float
    label_source: str
    source: str


class Map25DResponse(BaseModel):
    """Explicit 2.5D perception representation -- elevation, terrain,
    obstacle, and information fields per cell, plus map-level statistics
    and objective (ground-truth-independent) representation-correctness
    checks. Wraps the existing FOVEA/uniform/distance-adaptive map output;
    computes no new spatial allocation."""

    dataset: str
    sequence: str
    frame_id: str
    method: Literal["uniform", "distance_adaptive", "fovea"]
    label_source: Literal["ground_truth", "prediction"]
    source: str
    cells: list[PerceptionCellModel]
    stats: Map25DStatsModel
    point_conservation: dict[str, Any]
    no_duplicate_cells: dict[str, Any]
    geometry_valid: bool


class LODConfigModel(BaseModel):
    """Echoes back exactly the parameters that selected this response's LOD
    frontier -- see src.spatial.lod.LODSelectionConfig."""

    detail_threshold: float
    max_distance_m: Optional[float]
    point_budget: Optional[int]
    viewpoint_x: float
    viewpoint_y: float


class ChunkTimingModel(BaseModel):
    """Each stage measured SEPARATELY (Phase 3C requirement). Every field is
    a real one-time measurement from when that stage actually ran for this
    (frame, parameter-combination) -- reused on cache hits, the same
    convention `accounting["latency_s"]` already uses elsewhere in this API.
    Not a per-request wall-clock breakdown."""

    fovea_build_latency_s: float
    hierarchy_build_latency_s: float
    lod_selection_latency_s: float
    chunk_serialization_latency_s: float


class ChunksResponse(BaseModel):
    """Streamable spatial chunks for an existing FOVEA hierarchy + LOD
    frontier (src/spatial/chunks.py). `payload_base64` is the FVCH1 binary
    payload (see src/spatial/chunks.py's module docstring) base64-encoded
    for transport inside a JSON body -- NOT a list of tens of thousands of
    per-chunk JSON objects. `payload_bytes` is the RAW binary size (before
    base64's ~33% wire-size inflation, which exists only because this
    endpoint returns JSON like every other endpoint here -- see README)."""

    dataset: str
    sequence: str
    frame_id: str
    label_source: str
    source: str
    lod_config: LODConfigModel
    selected_node_count: int
    selected_point_count: int
    culled_root_count: int
    culled_subtree_count: int
    budget_dropped_node_count: int
    chunk_count: int
    payload_format: str
    payload_bytes: int
    payload_base64: str
    timing: ChunkTimingModel


# ---------------------------------------------------------------------------
# Internal, dataset-parameterized response builders -- both the new
# /api/datasets/{dataset}/... routes and the legacy SemanticPOSS-only routes
# call these. There is exactly one implementation.
# ---------------------------------------------------------------------------

def _sequences_response(dataset: str) -> dict:
    adapter, _ = _dataset_context(dataset)
    return {"dataset": dataset, "sequences": adapter.list_sequences()}


def _frames_response(dataset: str, sequence: str) -> dict:
    adapter, _ = _dataset_context(dataset)
    try:
        frame_ids = adapter.list_frames(sequence)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"dataset": dataset, "sequence": sequence, "frame_count": len(frame_ids), "frame_ids": frame_ids}


def _diagnostics_response(dataset: str, sequence: str, frame_id: str) -> dict:
    adapter, _ = _dataset_context(dataset)
    entry = _get_frame_entry(dataset, sequence, frame_id)
    diag = adapter.frame_diagnostics(entry["frame"])
    return {"dataset": dataset, **diag}


def _points_response(dataset: str, sequence: str, frame_id: str, max_points: int | None) -> PointsResponse:
    _, effective_cfg = _dataset_context(dataset)
    entry = _get_frame_entry(dataset, sequence, frame_id)
    frame: Frame = entry["frame"]
    cap = max_points if max_points is not None else int(effective_cfg["visualization"]["point_subsample_max"])
    n = frame.points.shape[0]

    pts, labels = frame.points, frame.semantic_labels
    subsampled = n > cap
    if subsampled:
        idx = np.random.default_rng(0).choice(n, size=cap, replace=False)
        pts, labels = pts[idx], labels[idx]

    return PointsResponse(
        dataset=dataset,
        sequence=sequence,
        frame_id=frame.frame_id,
        total_points_in_frame=n,
        returned_points=int(pts.shape[0]),
        subsampled=subsampled,
        x=pts[:, 0].tolist(),
        y=pts[:, 1].tolist(),
        z=pts[:, 2].tolist(),
        intensity=pts[:, 3].tolist(),
        semantic_id=[int(v) for v in labels.tolist()],
    )


def _map_response(dataset: str, sequence: str, frame_id: str, method: str, include_cells: bool) -> MapResponse:
    cells, accounting, summary, frame = _get_map(dataset, sequence, frame_id, method)
    return MapResponse(
        dataset=dataset,
        sequence=sequence,
        frame_id=frame.frame_id,
        method=method,  # type: ignore[arg-type]
        cells=[_cell_to_model(c, i) for i, c in enumerate(cells)] if include_cells else [],
        accounting=accounting,
        summary=summary,
    )


def _explain_response(dataset: str, sequence: str, frame_id: str, index: int, method: str) -> ExplainResponse:
    cells, _, _, frame = _get_map(dataset, sequence, frame_id, method)
    if not cells or index < 0 or index >= len(cells):
        raise HTTPException(status_code=404, detail=f"Cell index {index} out of range (0..{len(cells) - 1}).")
    cell = cells[index]
    return ExplainResponse(
        dataset=dataset,
        sequence=sequence,
        frame_id=frame.frame_id,
        method=method,
        cell_index=index,
        cell=_cell_to_model(cell, index),
        explanation=explain_allocation(cell),
    )


def _semantic_classes_response(dataset: str) -> dict:
    adapter, effective_cfg = _dataset_context(dataset)
    priority_table = effective_cfg["fovea"]["semantic_priority"]
    dynamic_classes = set(effective_cfg["fovea"]["dynamic_classes"])
    classes = [
        SemanticClassModel(
            class_id=cid,
            class_name=name,
            semantic_priority=float(priority_table.get(name, 0.1)),
            is_dynamic_relevant=name in dynamic_classes,
        )
        for cid, name in sorted(adapter.label_dict().items())
    ]
    return {"dataset": dataset, "classes": classes}


def _benchmark_result_path(dataset: str) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    if dataset == "SemanticPOSS":
        return repo_root / "results" / "benchmark_result.json"  # unchanged filename -- preserves v0.1 file
    slug = dataset.lower().replace(" ", "_").replace("/", "_")
    return repo_root / "results" / f"benchmark_result_{slug}.json"


def _benchmark_response(dataset: str, refresh: bool, num_frames: int, repeats: int) -> BenchmarkResponse:
    adapter, effective_cfg = _dataset_context(dataset)
    result_path = _benchmark_result_path(dataset)

    if refresh:
        from src.benchmark import run_benchmark

        sequence = adapter.list_sequences()[0] if dataset != _DEFAULT_DATASET else _CFG["data"]["sequence"]
        try:
            benchmark_adapter = None if dataset == "SemanticPOSS" else adapter
            data = run_benchmark(effective_cfg, sequence, num_frames, repeats, adapter=benchmark_adapter)
        except RuntimeError as e:
            return BenchmarkResponse(dataset=dataset, available=False, source="live_run_failed", message=str(e))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return BenchmarkResponse(dataset=dataset, available=True, source="live_run", data=data)

    if not result_path.exists():
        return BenchmarkResponse(
            dataset=dataset,
            available=False,
            source="none",
            message=f"Not measured -- no {result_path.name} found. "
            f"Call this endpoint with ?refresh=true to generate one from real {dataset} frames.",
        )

    with open(result_path) as f:
        data = json.load(f)
    return BenchmarkResponse(dataset=dataset, available=True, source=str(result_path.relative_to(result_path.parent.parent)), data=data)


def _class_distribution(labels: np.ndarray, name_lookup) -> list[ClassCount]:
    n = len(labels)
    ids, counts = np.unique(labels, return_counts=True)
    rows = [
        ClassCount(class_name=name_lookup(int(i)), point_count=int(c), fraction=float(c) / n if n else 0.0)
        for i, c in zip(ids, counts)
    ]
    return sorted(rows, key=lambda r: -r.point_count)


def _perception_response(dataset: str, sequence: str, frame_id: str, model_name: str) -> PerceptionResponse:
    model, _ = _perception_context(model_name)
    entry = _get_frame_entry(dataset, sequence, frame_id)
    result = _get_prediction(dataset, sequence, frame_id, model_name)

    mean_confidence = float(result.confidence.mean()) if result.confidence is not None else None

    return PerceptionResponse(
        dataset=dataset,
        sequence=sequence,
        frame_id=entry["frame"].frame_id,
        model_name=result.model_name,
        model_ontology=result.model_ontology,
        num_points=result.num_points,
        preprocessing_ms=result.preprocessing_ms,
        inference_ms=result.inference_ms,
        postprocessing_ms=result.postprocessing_ms,
        total_perception_ms=result.total_perception_ms,
        points_per_second=result.points_per_second,
        provides_confidence=model.provides_confidence,
        mean_confidence=mean_confidence,
        predicted_class_distribution=_class_distribution(
            result.predicted_labels, lambda i: model.raw_label_names.get(i, "unknown")
        ),
        predicted_canonical_distribution=_class_distribution(
            result.predicted_canonical, lambda i: CANONICAL_CLASSES.get(i, "unknown")
        ),
    )


def _evaluation_response(dataset: str, sequence: str, frame_id: str, model_name: str) -> EvaluationResponse:
    adapter, _ = _dataset_context(dataset)
    entry = _get_frame_entry(dataset, sequence, frame_id)
    frame: Frame = entry["frame"]

    if not frame.has_labels:
        return EvaluationResponse(
            dataset=dataset,
            sequence=sequence,
            frame_id=frame.frame_id,
            model_name=model_name,
            available=False,
            message=f"Frame {frame.frame_id} has no ground-truth labels -- cannot evaluate. "
            "Ground truth is only ever used for evaluation, never as a prediction.",
        )

    result = _get_prediction(dataset, sequence, frame_id, model_name)
    gt_names = [adapter.class_name(int(label_id)) for label_id in frame.semantic_labels]
    gt_canonical = np.asarray(map_names_to_canonical(gt_names, dataset), dtype=np.int64)
    range_m = np.linalg.norm(frame.points[:, :3], axis=1)

    ev = evaluate_predictions(result.predicted_canonical, gt_canonical, range_m, result.model_ontology)

    return EvaluationResponse(
        dataset=dataset,
        sequence=sequence,
        frame_id=frame.frame_id,
        model_name=model_name,
        available=True,
        num_points_total=ev.num_points_total,
        num_points_ignored=ev.num_points_ignored,
        num_points_evaluated=ev.num_points_evaluated,
        overall_accuracy=ev.overall_accuracy,
        mean_iou=ev.mean_iou,
        per_class_iou=ev.per_class_iou,
        distance_bands={
            k: DistanceBandMetricsModel(band=v.band, point_count=v.point_count, accuracy=v.accuracy, mean_iou=v.mean_iou)
            for k, v in ev.distance_bands.items()
        },
        model_coverage=ev.model_coverage,
        excluded_canonical_classes=ev.excluded_canonical_classes,
    )


def _map_2_5d_response(
    dataset: str,
    sequence: str,
    frame_id: str,
    method: str,
    label_source: str,
    model_name: str,
    include_cells: bool,
) -> Map25DResponse:
    """Wraps an EXISTING map build (ground-truth or prediction mode -- same
    _get_map/_get_prediction_map this module already uses for the plain
    /map/{method} endpoints) with the explicit 2.5D representation layer
    (src/spatial/). No new spatial allocation happens here."""
    if label_source == "ground_truth":
        cells, accounting, _, frame = _get_map(dataset, sequence, frame_id, method)
        source = dataset
    elif label_source == "prediction":
        cells, accounting, _, frame, _ = _get_prediction_map(dataset, sequence, frame_id, model_name, method)
        source = _perception_context(model_name)[0].model_ontology
    else:
        raise HTTPException(status_code=400, detail=f"Unknown label_source '{label_source}'. Use 'ground_truth' or 'prediction'.")

    perception_cells, stats = build_2_5d_map(cells, accounting, label_source, source)
    geometry = validate_cell_geometry(perception_cells)

    return Map25DResponse(
        dataset=dataset,
        sequence=sequence,
        frame_id=frame.frame_id,
        method=method,  # type: ignore[arg-type]
        label_source=label_source,  # type: ignore[arg-type]
        source=source,
        cells=[PerceptionCellModel(**vars(c) | {"terrain_class": c.terrain_class.value, "obstacle_class": c.obstacle_class.value}) for c in perception_cells] if include_cells else [],
        stats=Map25DStatsModel(**vars(stats)),
        point_conservation=validate_point_conservation(accounting),
        no_duplicate_cells=validate_no_duplicate_cells(cells),
        geometry_valid=geometry["valid"],
    )


def _chunks_response(
    dataset: str,
    sequence: str,
    frame_id: str,
    detail_threshold: float,
    max_distance_m: Optional[float],
    point_budget: Optional[int],
    viewpoint_x: float,
    viewpoint_y: float,
) -> ChunksResponse:
    """Wraps the EXISTING hierarchy (src/spatial/hierarchy.py) + LOD
    selector (src/spatial/lod.py) + chunk serializer (src/spatial/chunks.py)
    -- this function makes no FOVEA/hierarchy/LOD decision of its own, only
    cache lookups and response assembly. Ground-truth FOVEA mode only."""
    cells, accounting, _, frame = _get_map(dataset, sequence, frame_id, "fovea")
    _, _, hierarchy_latency_s = _get_hierarchy(dataset, sequence, frame_id)
    selection, payload, meta = _get_chunk_payload(
        dataset, sequence, frame_id, detail_threshold, max_distance_m, point_budget, viewpoint_x, viewpoint_y
    )

    return ChunksResponse(
        dataset=dataset,
        sequence=sequence,
        frame_id=frame.frame_id,
        label_source="ground_truth",
        source=dataset,
        lod_config=LODConfigModel(
            detail_threshold=detail_threshold,
            max_distance_m=max_distance_m,
            point_budget=point_budget,
            viewpoint_x=viewpoint_x,
            viewpoint_y=viewpoint_y,
        ),
        selected_node_count=selection.node_count,
        selected_point_count=selection.total_point_count,
        culled_root_count=selection.culled_root_count,
        culled_subtree_count=selection.culled_subtree_count,
        budget_dropped_node_count=selection.budget_dropped_node_count,
        chunk_count=meta.chunk_count,
        payload_format=meta.payload_format,
        payload_bytes=meta.payload_bytes,
        payload_base64=base64.b64encode(payload).decode("ascii"),
        timing=ChunkTimingModel(
            fovea_build_latency_s=float(accounting.get("latency_s", 0.0)),
            hierarchy_build_latency_s=hierarchy_latency_s,
            lod_selection_latency_s=selection.latency_s,
            chunk_serialization_latency_s=meta.serialization_latency_s,
        ),
    )


# ---------------------------------------------------------------------------
# Endpoints -- dataset-aware (primary)
# ---------------------------------------------------------------------------

@app.get("/api/status", response_model=StatusResponse)
def get_status():
    from src.metrics import environment_info

    root = Path(_CFG["data"]["root"])
    default_adapter, _ = _dataset_context(_DEFAULT_DATASET)
    return StatusResponse(
        status="ok",
        dataset_root=str(root),
        dataset_root_exists=root.exists(),
        available_sequences=default_adapter.list_sequences(),
        available_datasets=available_datasets(),
        default_dataset=_DEFAULT_DATASET,
        environment=environment_info(),
        cached_frames=[f"{d}/{s}/{f}" for d, s, f in _frame_cache_order],
    )


@app.get("/api/datasets")
def get_datasets():
    return {
        "datasets": [
            DatasetInfo(name=name, is_default=(name == _DEFAULT_DATASET)) for name in available_datasets()
        ],
        "default": _DEFAULT_DATASET,
    }


@app.get("/api/datasets/{dataset}/sequences")
def get_dataset_sequences(dataset: str):
    return _sequences_response(dataset)


@app.get("/api/datasets/{dataset}/sequences/{sequence}/frames")
def get_dataset_frames(dataset: str, sequence: str):
    return _frames_response(dataset, sequence)


@app.get("/api/datasets/{dataset}/frames/{sequence}/{frame_id}/diagnostics")
def get_dataset_diagnostics(dataset: str, sequence: str, frame_id: str):
    return _diagnostics_response(dataset, sequence, frame_id)


@app.get("/api/datasets/{dataset}/frames/{sequence}/{frame_id}/points", response_model=PointsResponse)
def get_dataset_points(dataset: str, sequence: str, frame_id: str, max: int = Query(default=None, ge=1, le=200_000)):
    return _points_response(dataset, sequence, frame_id, max)


@app.get("/api/datasets/{dataset}/frames/{sequence}/{frame_id}/map/{method}", response_model=MapResponse)
def get_dataset_map(
    dataset: str,
    sequence: str,
    frame_id: str,
    method: Literal["uniform", "distance_adaptive", "fovea"],
    include_cells: bool = Query(default=True),
):
    return _map_response(dataset, sequence, frame_id, method, include_cells)


@app.get("/api/datasets/{dataset}/frames/{sequence}/{frame_id}/cell/{index}/explain", response_model=ExplainResponse)
def get_dataset_cell_explanation(
    dataset: str,
    sequence: str,
    frame_id: str,
    index: int,
    method: Literal["uniform", "distance_adaptive", "fovea"] = Query(default="fovea"),
):
    return _explain_response(dataset, sequence, frame_id, index, method)


@app.get("/api/datasets/{dataset}/semantic-classes")
def get_dataset_semantic_classes(dataset: str):
    return _semantic_classes_response(dataset)


@app.get("/api/datasets/{dataset}/benchmark", response_model=BenchmarkResponse)
def get_dataset_benchmark(
    dataset: str,
    refresh: bool = Query(default=False),
    num_frames: int = Query(default=3, ge=1, le=20),
    repeats: int = Query(default=1, ge=1, le=5),
):
    return _benchmark_response(dataset, refresh, num_frames, repeats)


@app.get("/api/config")
def get_config():
    return _CFG


# ---------------------------------------------------------------------------
# Endpoints -- perception (real AI inference)
# ---------------------------------------------------------------------------

@app.get("/api/perception/models")
def get_perception_models():
    models_cfg = _CFG.get("perception", {}).get("models", {})
    infos = []
    for name in models_cfg:
        if name != "squeezeseg":
            continue  # only one loader implementation exists today (see _perception_context)
        model, _ = _perception_context(name)
        infos.append(
            PerceptionModelInfo(
                name=model.name,
                ontology=model.model_ontology,
                provides_confidence=model.provides_confidence,
                description=(
                    "Pretrained SqueezeSeg (Bonnetal/RangeNet++ checkpoint, trained on SemanticKITTI) -- "
                    "a range-image CNN. Real inference, not ground-truth passthrough; see README for "
                    "provenance, preprocessing, and measured accuracy/mIoU."
                ),
            )
        )
    return {"models": infos, "default": _CFG.get("perception", {}).get("default_model")}


@app.get(
    "/api/datasets/{dataset}/frames/{sequence}/{frame_id}/map/{method}/predicted",
    response_model=MapResponse,
)
def get_dataset_map_predicted(
    dataset: str,
    sequence: str,
    frame_id: str,
    method: Literal["uniform", "distance_adaptive", "fovea"],
    model: str = Query(default="squeezeseg", description="Perception model name, see /api/perception/models"),
    include_cells: bool = Query(default=True),
):
    """INFERENCE MODE: the same FOVEA/uniform/distance-adaptive engine, fed
    the model's PREDICTED labels instead of dataset ground truth. The
    ground-truth-mode endpoint above (.../map/{method}) is completely
    unchanged by this -- this is an additional path, not a modification."""
    cells, accounting, summary, frame, _ = _get_prediction_map(dataset, sequence, frame_id, model, method)
    return MapResponse(
        dataset=dataset,
        sequence=sequence,
        frame_id=frame.frame_id,
        method=method,
        cells=[_cell_to_model(c, i) for i, c in enumerate(cells)] if include_cells else [],
        accounting=accounting,
        summary=summary,
    )


@app.get(
    "/api/datasets/{dataset}/frames/{sequence}/{frame_id}/map/{method}/2_5d",
    response_model=Map25DResponse,
)
def get_dataset_map_2_5d(
    dataset: str,
    sequence: str,
    frame_id: str,
    method: Literal["uniform", "distance_adaptive", "fovea"],
    label_source: Literal["ground_truth", "prediction"] = Query(default="ground_truth"),
    model: str = Query(default="squeezeseg", description="Only used when label_source=prediction."),
    include_cells: bool = Query(default=True),
):
    """Explicit 2.5D perception representation (src/spatial/) -- elevation
    bounds, terrain/traversability, and obstacle classification per cell,
    plus map-level statistics and point-conservation/geometry validation.
    Wraps the SAME map build the plain /map/{method} (ground truth) and
    /map/{method}/predicted (prediction) endpoints already use -- adds a
    representation layer, not a new computation. Default behavior
    (label_source=ground_truth) leaves every existing endpoint untouched."""
    return _map_2_5d_response(dataset, sequence, frame_id, method, label_source, model, include_cells)


@app.get(
    "/api/datasets/{dataset}/frames/{sequence}/{frame_id}/chunks",
    response_model=ChunksResponse,
)
def get_dataset_chunks(
    dataset: str,
    sequence: str,
    frame_id: str,
    detail_threshold: float = Query(
        default=0.01, gt=0.0, description="LOD apparent-size threshold (cell_size_m / distance) -- see src/spatial/lod.py."
    ),
    max_distance: Optional[float] = Query(
        default=None, ge=0.0, description="Culls whole subtrees beyond this planar distance (m) from the viewpoint. Omit for no culling."
    ),
    point_budget: Optional[int] = Query(
        default=None, ge=0, description="Caps total selected point_count -- drops the farthest chunks first if exceeded. Omit for no budget."
    ),
    viewpoint_x: float = Query(default=0.0, description="Viewpoint X, sensor/vehicle origin frame (same frame as config.yaml's origin)."),
    viewpoint_y: float = Query(default=0.0, description="Viewpoint Y, sensor/vehicle origin frame."),
):
    """Streamable spatial chunks (src/spatial/chunks.py) for an existing
    FOVEA hierarchy (src/spatial/hierarchy.py) + LOD frontier
    (src/spatial/lod.py, select_lod_nodes) -- ground-truth FOVEA mode. This
    endpoint performs no FOVEA/hierarchy/LOD decision-making of its own,
    only cache lookup + chunk serialization; LOD logic is never duplicated
    here. See ChunksResponse/ChunkTimingModel for what's returned."""
    return _chunks_response(dataset, sequence, frame_id, detail_threshold, max_distance, point_budget, viewpoint_x, viewpoint_y)


@app.get(
    "/api/datasets/{dataset}/frames/{sequence}/{frame_id}/perception",
    response_model=PerceptionResponse,
)
def get_dataset_perception(
    dataset: str, sequence: str, frame_id: str, model: str = Query(default="squeezeseg")
):
    return _perception_response(dataset, sequence, frame_id, model)


@app.get(
    "/api/datasets/{dataset}/frames/{sequence}/{frame_id}/evaluation",
    response_model=EvaluationResponse,
)
def get_dataset_evaluation(
    dataset: str, sequence: str, frame_id: str, model: str = Query(default="squeezeseg")
):
    return _evaluation_response(dataset, sequence, frame_id, model)


# ---------------------------------------------------------------------------
# Endpoints -- legacy (backward compatible, SemanticPOSS-only)
# Same handlers as above with dataset fixed to "SemanticPOSS" -- not a
# separate implementation.
# ---------------------------------------------------------------------------

@app.get("/api/sequences")
def get_sequences():
    return _sequences_response("SemanticPOSS")


@app.get("/api/sequences/{sequence}/frames")
def get_frames(sequence: str):
    return _frames_response("SemanticPOSS", sequence)


@app.get("/api/frames/{sequence}/{frame_id}/diagnostics")
def get_diagnostics(sequence: str, frame_id: str):
    return _diagnostics_response("SemanticPOSS", sequence, frame_id)


@app.get("/api/frames/{sequence}/{frame_id}/points", response_model=PointsResponse)
def get_points(sequence: str, frame_id: str, max: int = Query(default=None, ge=1, le=200_000)):
    return _points_response("SemanticPOSS", sequence, frame_id, max)


@app.get("/api/frames/{sequence}/{frame_id}/map/{method}", response_model=MapResponse)
def get_map(
    sequence: str,
    frame_id: str,
    method: Literal["uniform", "distance_adaptive", "fovea"],
    include_cells: bool = Query(default=True, description="Set false to fetch only accounting/summary."),
):
    return _map_response("SemanticPOSS", sequence, frame_id, method, include_cells)


@app.get("/api/frames/{sequence}/{frame_id}/cell/{index}/explain", response_model=ExplainResponse)
def get_cell_explanation(
    sequence: str,
    frame_id: str,
    index: int,
    method: Literal["uniform", "distance_adaptive", "fovea"] = Query(default="fovea"),
):
    return _explain_response("SemanticPOSS", sequence, frame_id, index, method)


@app.get("/api/semantic-classes")
def get_semantic_classes():
    return _semantic_classes_response("SemanticPOSS")


@app.get("/api/benchmark", response_model=BenchmarkResponse)
def get_benchmark(
    refresh: bool = Query(default=False, description="If true, run a small fresh benchmark and overwrite results/benchmark_result.json"),
    num_frames: int = Query(default=3, ge=1, le=20),
    repeats: int = Query(default=1, ge=1, le=5),
):
    return _benchmark_response("SemanticPOSS", refresh, num_frames, repeats)
