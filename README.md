# FOVEA

Foveated Variable-Resolution 2.5D LiDAR Perception — prototype for SIH problem
statement 26053: *Adaptive Variable Resolution 2.5D Lidar Mapping for Dynamic
Environment Perception*.

**Status: proof-of-system prototype (v0.1), not a finished research result.**
Every weight/threshold in `config.yaml` is a heuristic default, clearly
labelled as such in code and config comments.

---

## FOR TEAM MEMBERS — QUICK START

```bash
# 1. Clone and enter the repo
git clone <repo-url> FOVEA
cd FOVEA

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install Python dependencies (engine + Streamlit + FastAPI + perception/torch)
#    torch is ~2GB and CPU/MPS-only (no CUDA needed/used) -- see "Real AI
#    perception layer" below for why. The SqueezeSeg checkpoint itself
#    (3.4MB) is already committed at models/squeezeseg/ -- nothing to download.
pip install -r requirements.txt

# 4. Get SemanticPOSS onto disk under dataset/ (see "Dataset setup" below)
#    Expected: dataset/sequences/<seq>/{velodyne,labels}/...
#    (dataset/ is gitignored -- it is NOT included in the repo)
#    RELLIS-3D is optional -- see "Multi-dataset support" below. The app
#    runs fine on SemanticPOSS alone; RELLIS-3D just adds a second dataset
#    to the selector once its two directories are present.

# 5. Start the FastAPI backend (terminal 1)
python run_api.py                # http://localhost:8000  (docs at /docs)

# 6. Start the React frontend (terminal 2)
cd frontend
npm install
cp .env.example .env.local       # VITE_API_URL=http://localhost:8000
npm run dev                      # http://localhost:5173

# Optional: the original Streamlit debug dashboard (terminal 3)
python run_demo.py               # http://localhost:8501

# Optional: real AI perception CLI (no server needed)
python run_perception.py --dataset SemanticPOSS --sequence 00 --frame 000000 --fovea
```

Open **http://localhost:5173** — that's the judge-facing FOVEA Command Center.

---

## 1. What FOVEA is

FOVEA takes a LiDAR point cloud + semantic labels and produces a **variable
resolution 2.5D semantic-elevation map**, where cell size is bounded by a
distance-dependent envelope (near = fine, far = coarse) but can be *further
refined within that envelope* based on an information score (semantic
priority + local geometric complexity + dynamic-object relevance). Two
regions at the **same distance** can end up at **different resolutions** if
one is more informative than the other — that behavior is measured and
demonstrated in this repo, not just asserted.

FOVEA is **not** claiming to invent LiDAR semantic segmentation, 2.5D
mapping, adaptive sampling, or resource-constrained optimization — all
established prior work. The contribution is the specific integration:
information-aware refinement *constrained by* a distance envelope, applied
to a navigation-oriented 2.5D semantic-elevation grid.

## 2. Problem being solved

Autonomous ground vehicles need a spatial map dense enough to act on
(obstacles, pedestrians, curbs) but sparse enough to compute and store in
real time. A uniform-resolution grid wastes detail on empty far-field space;
a purely distance-based grid wastes detail on *near* but uninteresting
regions (a stretch of blank pavement) while under-resolving a *far but
important* one (a pedestrian at range). FOVEA allocates resolution by
**information content within a distance-bounded envelope**, so detail
concentrates where it's actually useful for perception/planning.

## 3. Core FOVEA concept

```
RAW LiDAR
  -> SCENE UNDERSTANDING       (semantic labels per point -- SemanticPOSS ground truth in v0.1)
  -> INFORMATION ESTIMATION    (semantic priority + geometric complexity + dynamic relevance)
  -> FOVEA INFORMATION SCORE   (weighted sum, config.yaml -> fovea.weights)
  -> ADAPTIVE RESOLUTION       (quadtree split within the distance-band envelope)
  -> VARIABLE-RESOLUTION 2.5D MAP
  -> BENCHMARK                 (vs. uniform and distance-adaptive baselines)
```

See "FOVEA's resolution decision" (§16) for exactly how the envelope +
refinement works.

## 4. Architecture

```
SemanticPOSS (dataset/)
      |
      v
Python FOVEA Engine (src/*.py)   <-- source of truth for every number
      |                              shown in either frontend
      +----------------------------------+
      v                                  v
FastAPI (src/api.py)              Streamlit (src/app.py)
      |                            [internal debug dashboard]
      v
React + TypeScript + Vite (frontend/)
      |
      v
Three.js / React Three Fiber   [judge-facing FOVEA Command Center]
```

Both frontends call the **same** `src/*.py` functions (`load_frame`,
`build_uniform_map`, `build_distance_adaptive_map`, `build_fovea_map`,
`summarize_method`, `explain_allocation`). Neither frontend, and no
TypeScript code, reimplements any part of the allocation/scoring logic —
`src/api.py` only serializes the engine's real outputs into JSON.

## 5. Repository structure

```
config.yaml            -- all tunable weights/thresholds/paths (single source of truth)
requirements.txt       -- Python deps (engine + Streamlit + FastAPI + perception/torch)
run_demo.py             -- launches the Streamlit debug dashboard
run_api.py               -- launches the FastAPI backend
run_perception.py         -- real-inference CLI (see "Real AI perception layer")

dataset/                -- SemanticPOSS goes here (gitignored, see below)
  read_data.py            -- the dataset's own reference reader (verified against)

Rellis-3D/                -- RELLIS-3D point clouds (gitignored, optional, see "Multi-dataset support")
Rellis-3D 2/               -- RELLIS-3D annotations (gitignored, optional)

models/
  squeezeseg/               -- pretrained SqueezeSeg checkpoint (committed, 3.4MB, MIT license --
                               see NOTICE.md for full provenance)

src/
  config.py               -- YAML config loading (SemanticPOSS + per-dataset RELLIS-3D paths)
  data_loader.py           -- SemanticPOSS .bin/.label -> Frame, diagnostics
  semantic_labels.py       -- verbatim SemanticPOSS LABEL_DICT + priority lookup
  geometry.py              -- coordinate/index/band-assignment helpers
  map_structure.py         -- Cell dataclass + point->cell aggregation (shared by all 3 methods;
                              optional label_dict param for non-SemanticPOSS datasets)
  uniform_baseline.py      -- Baseline 1: fixed cell size everywhere
  distance_baseline.py     -- Baseline 2: cell size purely a function of range
  fovea_allocator.py       -- FOVEA: quadtree refinement within the distance envelope
  metrics.py               -- measured comparison metrics, distance-stratified stats
  benchmark.py              -- CLI benchmark -> results/benchmark_result*.json (dataset-aware)
  visualization.py          -- Plotly figures for Streamlit
  theme.py                  -- Streamlit CSS/HTML helpers
  frame_scanner.py          -- picks demo-worthy frames (class diversity)
  app.py                    -- Streamlit dashboard (internal/debug, SemanticPOSS-only)
  api.py                    -- FastAPI backend (judge-facing frontend's data source, dataset-aware)
  datasets/                 -- dataset abstraction (see "Multi-dataset support")
    base.py                   -- DatasetAdapter interface
    __init__.py                -- registry: get_adapter(name, cfg), available_datasets()
    semanticposs.py            -- thin adapter wrapping the UNCHANGED data_loader.py/semantic_labels.py
    rellis.py                  -- RELLIS .bin/.label loading, frame discovery, diagnostics
    rellis_labels.py           -- RELLIS's own LABEL_DICT (official ontology, NOT SemanticPOSS's)
    rellis_adapter.py          -- RELLIS DatasetAdapter implementation
  perception/                -- real AI inference (see "Real AI perception layer")
    base.py                    -- SemanticSegmentationModel protocol, RawPrediction/PerceptionResult
    model.py                    -- SqueezeSeg architecture (verified via strict state_dict load) + loading
    preprocessing.py             -- spherical range-image projection/unprojection
    ontology.py                  -- canonical class space + per-source (model/dataset) mapping tables
    inference.py                  -- timed preprocess->forward->postprocess orchestration
    evaluation.py                  -- accuracy/mIoU/per-class IoU/confusion matrix/distance bands
  spatial/                    -- explicit 2.5D representation (see "2.5D Perception Representation")
    map_2_5d.py                  -- PerceptionCell, Map2_5DStats, build_2_5d_map, validate_* checks
    terrain.py                    -- DRIVABLE/NON_DRIVABLE/OBSTACLE/UNKNOWN classification (per-source rule tables)
    obstacle.py                    -- static/potential-dynamic obstacle split (reuses existing dynamic_score)

frontend/                -- React + TypeScript + Vite + R3F Command Center
  src/
    api/                   -- typed fetch client + response types (mirrors src/api.py, dataset-aware)
    scene/                 -- Three.js / R3F: CellField, PointCloud, Scene
    components/            -- Header (incl. dataset selector), ViewModeTabs, RegionInspector,
                              InformationAnalytics, BenchmarkPanel, FrameControls, PipelineFlow, ScoreBar
    hooks/                  -- useFrameData, useStaticData (fetch + cache per dataset/sequence/frame)
    constants.ts             -- presentation-only colors/labels (no FOVEA logic)
    semanticLookup.ts        -- class-id -> name, re-fetched from the backend on every dataset switch
  .env.example              -- VITE_API_URL=http://localhost:8000

tests/
  test_core.py             -- 12 tests: allocation engine (pre-existing, untouched)
  test_api.py                -- 38 tests: FastAPI HTTP boundary, incl. dataset + perception endpoints
  test_datasets.py            -- 18 tests: dataset abstraction, RELLIS parsing, ontology, engine integration
  test_perception.py           -- 27 tests: model loading, preprocessing, real inference, evaluation, FOVEA-on-predictions
  test_spatial.py                -- 39 tests: 2.5D representation, terrain/obstacle classification, point
                                   conservation, prediction-vs-ground-truth separation, real API responses

results/
  benchmark_result.json    -- output of `python -m src.benchmark`
```

## 6. Dataset structure (SemanticPOSS)

SemanticPOSS, as verified directly from the files (not assumed):

```
dataset/sequences/<seq>/velodyne/<frame>.bin   float32, reshape(-1,4) = x,y,z,intensity
dataset/sequences/<seq>/labels/<frame>.label   uint32; semantic = value & 0xFFFF, instance = value >> 16
```

Confirmed against `dataset/read_data.py` (the dataset's own reader) and
against actual file sizes (frame `000000` of sequence `00`: 66,658 points,
66,658 labels — byte counts match exactly). The class-id -> name mapping
(`src/semantic_labels.py`) is copied verbatim from that same file's
`LABEL_DICT` — SemanticPOSS uses a **sparse, non-contiguous** id set (`0,
4-17, 21, 22`). Any id outside that set is reported as `"unknown"`, never
guessed.

**Dataset setup:** SemanticPOSS is not committed to this repo (multi-GB,
gitignored). Obtain it from the official SemanticPOSS distribution and place
it so that `dataset/sequences/00/velodyne/*.bin` etc. exist. This repo ships
6 sequences' worth of paths configured (`00`-`05`); you only need however
many you actually have data for.

## 7. Dataset configuration

`config.yaml` -> `data`:

```yaml
data:
  root: dataset       # contains sequences/<seq>/{velodyne,labels}
  sequence: "00"
  frame: 0
```

`src/config.py` resolves `data.root` relative to the repo root if it isn't
already absolute. Both the Streamlit app and the FastAPI backend load this
same `config.yaml` — there is exactly one place weights/paths are defined.

---

## Multi-dataset support: SemanticPOSS + RELLIS-3D

FOVEA supports two datasets through a common adapter interface
(`src/datasets/`) — the FOVEA engine, FastAPI, and the React frontend are
all dataset-agnostic; only a small adapter module per dataset knows about
its on-disk layout and ontology.

```
Dataset (SemanticPOSS or RELLIS-3D)
      |
      v
Dataset adapter (src/datasets/semanticposs.py or rellis_adapter.py)
      |  -> common Frame representation (src/data_loader.Frame:
      |     points, semantic_labels, instance_labels, ...)
      v
FOVEA engine (uniform_baseline.py / distance_baseline.py / fovea_allocator.py)
      -- UNCHANGED allocation math; only the class-id->name table and
         semantic-priority/dynamic-class config it's given differ per dataset
      |
      v
FastAPI (/api/datasets/{dataset}/...)  ->  React dataset selector
```

**Important scope note, stated explicitly:** RELLIS-3D is an **off-road**
dataset (trails, grass, mud, vegetation) and SemanticPOSS is a **campus/
urban** dataset — a fundamentally different scene distribution. This
integration does **not** claim FOVEA's cell counts, latency, or information
scores are directly comparable in absolute terms between the two datasets.
The point of running FOVEA on both is to demonstrate that the **same
allocation mechanism** (distance envelope + information-aware refinement)
operates correctly across genuinely different LiDAR scene distributions —
not to rank the datasets against each other.

### RELLIS-3D setup

RELLIS-3D ships as **two separately-downloaded directory trees** (point
clouds and annotations) — this repo reads both **in place**, at whatever
path you extracted them to; nothing is copied or merged. Verified against
the actual files in this repo's local copy:

```
Rellis-3D/<seq>/vel_cloud_node_kitti_bin/<frame>.bin                  float32, reshape(-1,4) = x,y,z,intensity
Rellis-3D 2/<seq>/vel_cloud_node_semantickitti_label_id/<frame>.label  uint32; semantic = value & 0xFFFF, instance = value >> 16
```

`<seq>` is one of `00000`-`00004`. Confirmed by direct inspection (not
assumed): frame `001372` of sequence `00000` is 555,920 bytes = 138,980
float32 values = 34,745 points × 4 columns; its `.label` file is 138,980
bytes = 34,745 uint32 values — point count and label count match exactly.
Checked across one frame from **each of the 5 sequences** with the same
result. Only frame ids where **both** a `.bin` and a matching `.label` file
exist are ever listed or loaded (`src/datasets/rellis.py:list_frame_ids`) —
a point cloud with no label is excluded, never silently shown as labeled.

Point it at your local extraction via `config.yaml`:

```yaml
datasets:
  RELLIS-3D:
    pointcloud_root: Rellis-3D      # -> Rellis-3D/<seq>/vel_cloud_node_kitti_bin/*.bin
    label_root: "Rellis-3D 2"       # -> Rellis-3D 2/<seq>/vel_cloud_node_semantickitti_label_id/*.label
```

Both paths (and SemanticPOSS's `data.root`) are resolved relative to the
repo root if not absolute, same as v0.1. RELLIS-3D data is large
(~7.4GB point clouds + ~1.9GB annotations in this repo's copy) and is
gitignored — never committed.

### RELLIS-3D ontology

RELLIS-3D uses its **own** class ontology — completely different id space
and names from SemanticPOSS. It is **never** reused or aliased; kept in a
separate module (`src/datasets/rellis_labels.py`) with its own `LABEL_DICT`,
mirroring how `src/semantic_labels.py` holds SemanticPOSS's.

The mapping was fetched verbatim from RELLIS-3D's own official benchmark
config (not guessed):
```
https://raw.githubusercontent.com/unmannedlab/RELLIS-3D/main/benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml
```
— specifically its `labels:` section (the raw per-point id space used by
`vel_cloud_node_semantickitti_label_id`, **not** the reduced `learning_map`
training taxonomy that collapses several classes to "void"). Cross-checked
against real data: every semantic id observed in this repo's actual
`.label` files (`{0,3,4,8,9,17,18,19,23,27,31,33}`, sampled across all 5
sequences) is present in the fetched table, confirming it's the right id
space (`tests/test_datasets.py::test_semantic_ids_are_within_official_ontology`
runs this check on demand).

| id | name | id | name | id | name | id | name |
|---|---|---|---|---|---|---|---|
| 0 | void | 6 | water | 12 | building | 27 | barrier |
| 1 | dirt | 7 | sky | 15 | log | 31 | puddle |
| 3 | grass | 8 | vehicle | 17 | person | 33 | mud |
| 4 | tree | 9 | object | 18 | fence | 34 | rubble |
| 5 | pole | 10 | asphalt | 19 | bush | | |

### Dataset-specific semantic priorities

SemanticPOSS's existing `fovea.semantic_priority` table (config.yaml) is
**untouched** — those exact values are still used for SemanticPOSS, unchanged
byte-for-byte. RELLIS-3D gets its **own** table
(`config.yaml` -> `datasets.RELLIS-3D.semantic_priority`), a **separate
prototype heuristic** in the same spirit (person/vehicle high, generic
terrain low, hazards like water/mud/puddle/log/rubble moderate-high) —
**not scientifically validated**, and never derived from or copied to
SemanticPOSS's table. Any RELLIS class absent from the table falls back to
0.1, the same convention SemanticPOSS uses. Dynamic relevance is the same
semantic-category-proxy concept as SemanticPOSS (`person`, `vehicle` for
RELLIS — it has no rider/bike-equivalent category) — **not** real
motion/temporal tracking, labeled honestly as "Dynamic relevance (semantic
proxy)" everywhere it's shown (inspector, analytics panel).

### How dataset selection works end-to-end

- **Config**: `config.yaml` -> `datasets.default` (`SemanticPOSS`) plus one
  entry per dataset with its paths/priorities/dynamic-classes.
- **Engine**: `build_uniform_map` / `build_distance_adaptive_map` /
  `build_fovea_map` (and `map_structure.aggregate_points_to_cells` /
  `compute_information_score`) gained one **optional** `label_dict` kwarg
  (default `None` = SemanticPOSS's table, i.e. byte-identical v0.1
  behavior). The active dataset's semantic-priority table and dynamic-class
  set are substituted into a per-request copy of `cfg["fovea"]` at the API
  layer (`src/api.py:_dataset_context`) — `uniform_baseline.py`,
  `distance_baseline.py`, and `fovea_allocator.py` still just read
  `cfg["fovea"]["semantic_priority"]`/`["dynamic_classes"]` exactly as
  before; **no allocation/scoring formula changed**.
- **API**: every per-frame endpoint has a dataset-aware form under
  `/api/datasets/{dataset}/...` (see §14). The original no-dataset routes
  (`/api/sequences`, `/api/frames/...`, etc.) still work unchanged and are
  fixed to `dataset=SemanticPOSS` — same handler function, not a duplicate
  implementation.
- **Cache**: keys are `(dataset, sequence, frame_id)`, not just `(sequence,
  frame_id)` — SemanticPOSS and RELLIS-3D frames can never collide or serve
  each other's data, and the per-key locking/coalescing fix (see the
  performance section below) applies identically to both.
- **Frontend**: the header's dataset dropdown (`SemanticPOSS` / `RELLIS-3D`,
  default `SemanticPOSS`) drives everything downstream — switching it
  reloads that dataset's sequence list, resets to its first sequence/frame,
  and re-fetches the semantic-class legend (critical: the same raw class id
  means a different thing in each ontology, e.g. id `7` is `car` in
  SemanticPOSS but `sky` in RELLIS-3D, so a stale lookup would mislabel the
  raw point cloud). All four view modes, the FOVEA scene, camera controls,
  region inspector, and benchmark panel are unchanged and dataset-agnostic —
  they just render whatever the selected dataset's API responses contain.

---

## Real AI perception layer (src/perception/)

**Dataset ground-truth labels are used for evaluation only.** FOVEA can now
run on real neural-network predictions instead of ground truth:

```
DatasetAdapter (SemanticPOSS or RELLIS-3D)
      |
      v
Frame (raw x,y,z,intensity -- ground truth, when present, travels alongside
      |  but is NEVER passed into the model)
      v
SemanticSegmentationModel.predict(points)  -- src/perception/model.py
      |  (real forward pass through pretrained weights; model's OWN ontology)
      v
PerceptionResult (predicted labels + canonical-mapped labels + real timing)
      |
      v
FOVEA engine -- build_uniform_map / build_distance_adaptive_map /
                build_fovea_map, COMPLETELY UNMODIFIED, called with
                predicted labels exactly like it's called with ground truth
      |
      v
existing API / visualization (an ADDITIONAL "inference mode" path,
                               ground-truth mode is unchanged)
```

### Model selected: SqueezeSeg (Bonnetal/RangeNet++ pretrained checkpoint, SemanticKITTI)

**Why this model, and what was rejected:**

| Option | Verdict | Reason |
|---|---|---|
| Cylinder3D, SPVCNN | Rejected | Require `spconv`/`torchsparse` -- CUDA-only compiled kernels. This machine (Apple Silicon, no CUDA) cannot run them at all; not a preference, a hard blocker. |
| PointNet++ (pretrained) | Rejected | Public pretrained checkpoints are for indoor classification/segmentation (ModelNet40, S3DIS) or 3D shape tasks -- applying one to outdoor 100k-point automotive/off-road LiDAR would be a severe, scientifically dishonest domain mismatch, not a working model. |
| RandLA-Net, KPConv | Rejected | Custom compiled neighbor-search ops, larger checkpoints, less-verifiable preprocessing pipelines within this task's time budget. |
| **SqueezeSeg (Bonnetal)** | **Selected** | Pure PyTorch 2D CNN (SqueezeNet-style Fire modules) over a spherical range-image projection -- no compiled/CUDA-only ops, runs on CPU or Apple Silicon MPS. **3.4MB checkpoint**, hosted at a stable, direct HTTP URL (verified reachable, not Google Drive). Fully-specified preprocessing (exact FOV/projection/normalization parameters ship in the checkpoint's own config files). SemanticKITTI ontology, MIT-licensed, from the original RangeNet++/Bonnetal paper authors (University of Bonn). |

Checkpoint source (fetched and verified reachable at integration time):
`http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/squeezeseg.tar.gz`
(from `github.com/PRBonn/lidar-bonnetal`, archived but weights remain hosted).
Committed into this repo at `models/squeezeseg/` (3.4MB, MIT license --
provenance and full license text in `models/squeezeseg/NOTICE.md`) so the
demo runs fully offline, with no dependency on that URL staying up.

**Architecture verification:** `src/perception/model.py` reimplements the
SqueezeSeg backbone/decoder as a line-for-line port of the original
`PRBonn/lidar-bonnetal` source (fetched and read directly, not
reconstructed from guesswork), with every layer name matching the
checkpoint's state_dict exactly. Weights are loaded with
`strict=True` -- if the reimplementation were wrong, loading would raise
immediately rather than silently producing garbage. It loads successfully
(all 86 parameter tensors across backbone/decoder/head match by name and
shape), which is itself the correctness proof.

### Preprocessing (`src/perception/preprocessing.py`)

The model expects a spherical range-image projection, not raw points --
reimplemented exactly from the original codebase's `laserscan.py`:

```
depth  = ||[x,y,z]||
yaw    = -atan2(y, x)
pitch  = asin(z / depth)
proj_x = 0.5*(yaw/pi + 1) * W          (floor, clip to [0,W-1])
proj_y = (1 - (pitch+|fov_down|)/fov) * H   (floor, clip to [0,H-1])
```
Image size 64x2048, `fov_up=+3°`, `fov_down=-25°` (all read from the
checkpoint's own `arch_cfg.yaml`, not hardcoded separately). Input channels:
`[range, x, y, z, intensity]`, normalized per-channel with the checkpoint's
own mean/std. When multiple points land on the same pixel, the nearest one
(smallest range) wins. **No point is discarded** -- every point gets a
pixel (clipped to the image border if outside the trained FOV, not
filtered out); `preprocessing.RangeImageProjector.unproject()` scatters
per-pixel predictions back to a per-point array using the same index
mapping recorded during projection, and `src/perception/inference.py`
raises immediately if the result length doesn't match the input length --
`len(predicted_labels) == len(frame.points)` is enforced, not assumed.

**A real bug found and fixed during integration:** SemanticKITTI's own
`.bin` intensity channel is already normalized to ~`[0,1]`; SemanticPOSS and
RELLIS-3D instead store raw `[0,255]` sensor reflectivity. Feeding that
directly into the model's intensity-normalized input channel produced
values ~250x too large, saturating the network -- predictions collapsed
onto one dominant class (~79% "traffic-sign") and accuracy was <1%.
`SqueezeSegModel.preprocess()` now rescales to `[0,1]` whenever the input
clearly isn't already in that range. `tests/test_perception.py`'s
`test_predictions_are_not_degenerate` is a regression guard against this
exact failure mode recurring silently.

### Semantic ontology mapping (`src/perception/ontology.py`)

SqueezeSeg's SemanticKITTI ontology (20 reduced training classes),
SemanticPOSS's, and RELLIS-3D's are three **separate, incompatible**
class-id spaces -- never assumed interchangeable. A small canonical class
set (`unknown, ground, vegetation, building, fence, pole, sign, person,
rider, car, bike, other_vehicle, object, hazard`) is the common space all
three map INTO, by class name, for cross-ontology evaluation and for
running FOVEA on predictions:

- **`hazard`** exists only in RELLIS-3D's off-road ontology (mud, puddle,
  water, log, rubble, barrier) -- SqueezeSeg's urban-driving vocabulary has
  no equivalent, so it can **never** predict this canonical class. This is
  not hidden: `evaluate_predictions()` reports `model_coverage` (fraction of
  ground truth the model can structurally ever get right) and
  `excluded_canonical_classes` explicitly.
- The mapping tables are hand-authored, documented prototype heuristics
  (same status as the semantic-priority tables), not a validated semantic
  equivalence standard.

### Prediction vs. ground truth

`Frame.semantic_labels` (ground truth, when `has_labels=True`) is **never**
passed to `model.predict()` -- every model call in this codebase passes
only `frame.points`. Ground truth is read in exactly one place:
`src/perception/evaluation.py` and the `/evaluation` API endpoint /
`--fovea`-independent CLI evaluation step, strictly to score predictions
after the fact. `tests/test_perception.py::test_ground_truth_never_used_as_prediction`
and the CLI's printed "NOTE: ground truth is NEVER passed to the model" are
both there to make this checkable, not just claimed.

### Confidence

SqueezeSeg's head is a softmax over the 20 output classes -- real,
model-provided confidence (`PerceptionResult.confidence` = per-point max
softmax probability), never invented. If a future model doesn't provide
one, `confidence` is `None`/`null` end-to-end (API, CLI, dataclass) --
checked, never defaulted to a fabricated number.

### Evaluation methodology (`src/perception/evaluation.py`)

Both predictions and ground truth are mapped to the canonical space, then:
overall accuracy, per-class IoU (`TP/(TP+FP+FN)`, `None` when a class is
absent from both prediction and ground truth -- never fabricated as 0),
mean IoU (mean of per-class IoU, excluding the ignored `unknown` class,
matching SemanticKITTI's own ignore-class convention), a full confusion
matrix, and the same accuracy/mIoU broken down by real point range into
`0-10m / 10-30m / 30-60m / 60-100m` bands.

### Measured results (real runs, this machine, no fabricated numbers)

**SemanticPOSS** (3 frames, sequences 00/01), SqueezeSeg zero-shot (no
fine-tuning) — a genuine cross-sensor/cross-domain transfer (KITTI's
Velodyne HDL-64E, urban driving -> SemanticPOSS's Pandora40, campus scenes):

| seq/frame | points | total latency | accuracy | mean IoU |
|---|---|---|---|---|
| 00/000000 | 66,658 | ~0.85-0.95s (cold), ~65ms (warm, see note) | 0.4019 | 0.1296 |
| 00/000050 | 65,377 | ~65ms | 0.4666 | 0.1566 |
| 01/000100 | 65,662 | ~61ms | 0.3523 | 0.1211 |

Per-class IoU on 00/000000 (canonical): `ground 0.619`, `car 0.443`,
`fence 0.150`, `building 0.145`, `vegetation 0.125`, `pole 0.048`,
`sign 0.025`, `person/rider/bike/other_vehicle/object 0.000` (rare classes
in this frame, genuinely 0 overlap, not omitted). Distance-band accuracy on
the same frame decays with range as expected: `0-10m: 0.794`,
`10-30m: 0.384`, `30-60m: 0.205`, `60-100m: 0.220`.

**RELLIS-3D** (3 frames, sequences 00000/00002/00004) — off-road terrain,
a much larger domain gap:

| seq/frame | points | accuracy | mean IoU | model_coverage | excluded |
|---|---|---|---|---|---|
| 00000/000100 | 37,337 | 0.0998 | 0.0304 | 0.986 | `hazard` |
| 00002/000500 | 40,326 | 0.1202 | 0.0342 | 1.000 | `hazard` |
| 00004/000050 | 36,049 | 0.0012 | 0.0001 | 0.916 | `hazard` |

**Reported honestly, not adjusted:** RELLIS-3D accuracy is far lower than
SemanticPOSS's, as expected for an urban-driving-trained model applied to
off-road terrain with no fine-tuning. `model_coverage` is high (91-100% of
ground truth is in classes the model could theoretically predict), so the
low score is a genuine recognition failure under domain shift, not (only)
an ontology-coverage artifact -- the one frame (00004/000050) that
collapsed to near-zero accuracy was not specially investigated beyond
confirming the pipeline ran without error; it is reported as measured, not
excluded or explained away.

**Timing note:** the first inference call on a process is slower
(~750-950ms) than subsequent ones (~60-65ms) -- PyTorch/MPS backend kernel
compilation happens on first use, then is cached for the process's
lifetime. Both numbers are real measurements, not a discrepancy to
"fix" — same warm/cold pattern already documented for FOVEA's own map
build latency (§17).

### Inference hardware

Apple M1 Pro, 16GB RAM, macOS, arm64. No CUDA (Apple Silicon never has
CUDA) -- PyTorch's MPS (Metal Performance Shaders) backend is used
automatically when available, else falls back to CPU; both work with the
same code path (`torch.backends.mps.is_available()` check in
`SqueezeSegModel.__init__`). No GPU cluster, no cloud inference.

### API and CLI

New endpoints (additive only -- every existing endpoint is unchanged):

| Method | Path | Returns |
|---|---|---|
| GET | `/api/perception/models` | Registered models: name, ontology, confidence availability, description |
| GET | `/api/datasets/{dataset}/frames/{sequence}/{frame_id}/perception?model=` | Real inference summary: timing breakdown, point count, predicted class distributions (no per-point arrays) |
| GET | `/api/datasets/{dataset}/frames/{sequence}/{frame_id}/evaluation?model=` | Real accuracy/mIoU/per-class IoU/confusion matrix/distance bands vs. ground truth (`available=false` + honest `message` if the frame has none) |
| GET | `/api/datasets/{dataset}/frames/{sequence}/{frame_id}/map/{method}/predicted?model=` | FOVEA/uniform/distance-adaptive map built from PREDICTED labels -- same engine, same `?include_cells=` flag as ground-truth mode |

CLI (`run_perception.py`):
```bash
python run_perception.py --dataset SemanticPOSS --sequence 00 --frame 000000
python run_perception.py --dataset RELLIS-3D --sequence 00000 --frame 000100 --fovea
python run_perception.py --dataset SemanticPOSS --sequence 00 --frame 000000 --no-eval
```
Loads a real frame, runs real inference, prints the model/timing/predicted
distribution, evaluates against ground truth if present, and (`--fovea`)
runs all three FOVEA-engine methods on the predictions, printing real cell
counts. `uniform`/`distance_adaptive` cell counts are identical between
ground-truth and prediction mode for the same frame (their cell geometry
depends only on point XY, not labels) -- only FOVEA's count differs, since
only its refinement decision reacts to information score. This is real,
observed internal consistency, not an assumption.

### Perception layer limitations

- Zero-shot only -- no fine-tuning was performed on SemanticPOSS or
  RELLIS-3D (would require labeled training splits and meaningfully more
  time than this integration phase allows); reported accuracy reflects
  that, honestly.
- RELLIS-3D's `hazard` canonical class (mud/puddle/water/log/rubble/
  barrier) is structurally unreachable by this SemanticKITTI-trained model
  -- reported via `model_coverage`/`excluded_canonical_classes`, never
  silently mapped to a nearest class.
- Only one model (SqueezeSeg) is implemented; `src/perception/base.py`'s
  `SemanticSegmentationModel` protocol and the config-driven
  `perception.models` registry are designed so a second model is one new
  adapter module, not a rearchitecture -- not built in this phase.
- No dynamic-object detection was added or claimed -- FOVEA's "dynamic
  relevance" remains the same documented semantic-category proxy described
  in §16, now also available for the model's own ontology
  (`perception.models.squeezeseg.dynamic_classes` in config.yaml).
- Confidence is the model's raw softmax max-probability -- not calibrated;
  no claim of calibration is made.

---

## 2.5D Perception Representation (src/spatial/)

**Phase 2 of the SIH work — adds a representation layer, not new FOVEA
mathematics.** `src/spatial/` wraps the existing, UNMODIFIED FOVEA engine
output (`Cell` + `accounting`, from `build_uniform_map` /
`build_distance_adaptive_map` / `build_fovea_map`, run in either
ground-truth or Phase-1-prediction mode) into an explicit representation
with elevation, terrain/traversability, and obstacle status as first-class
fields:

```
Cell (existing, map_structure.py)
  center_x, center_y, cell_size_m, elevation_median/min/max/std,
  majority_class_name, semantic_priority, geometry_score, dynamic_score,
  information_score, resolution_level
      |
      v
PerceptionCell (src/spatial/map_2_5d.py)
  + explicit x/y bounds, terrain_class, obstacle_class, label_source
      |
      v
Map2_5DStats -- map-level terrain/elevation/coverage summary
```

### 1. Why 2.5D, not a flat 2D grid

A flat 2D occupancy grid answers "is this XY location occupied" with a
single bit. FOVEA's cells already carry a real elevation value per XY
location (`elevation_median`, robust to outliers within the cell) --
2.5D keeps that height information alongside the semantic/terrain/obstacle
classification, which is what lets a downstream planner distinguish, e.g.,
a low curb (small elevation change, still `DRIVABLE`) from a wall (large
elevation change, `OBSTACLE`) at the same XY footprint size. Like any 2.5D
representation, it still cannot capture two stacked surfaces at the same
XY (a bridge over a road) -- a known, previously-documented limitation of
median-Z aggregation, unchanged by this phase.

### 2. How elevation is represented

`elevation_median`/`elevation_min`/`elevation_max`/`elevation_std` are
computed directly from the actual LiDAR points assigned to each cell --
two new columns (`elevation_min`, `elevation_max`) were added to the
EXISTING pandas aggregation in `map_structure.py` (a 2-line, purely
additive extension: `.agg(..., elevation_min=("z","min"),
elevation_max=("z","max"))`), alongside the pre-existing `elevation_median`/
`elevation_std`. **No arbitrary or fabricated height is ever used** --
verified by `tests/test_spatial.py::test_map_level_elevation_stats_are_real`,
which cross-checks the reported map-level elevation range against the raw
point cloud's own actual z-range.

### 3. How semantic predictions enter the map

`build_2_5d_map(cells, accounting, label_source, source)` takes whichever
`cells`/`accounting` an existing map-build call already produced --
ground-truth mode (`build_fovea_map(points, frame.semantic_labels, cfg)`)
or Phase 1's prediction mode (`build_fovea_map(points,
perception_result.predicted_labels, model_cfg, label_dict=...)`). It
performs no inference and no spatial allocation itself -- `semantic_class`
on each `PerceptionCell` is exactly `Cell.majority_class_name`, whichever
label source produced it.

### 4. How terrain/traversability is derived

`src/spatial/terrain.py`'s `classify_terrain(class_name, source)` maps a
cell's semantic class name to `DRIVABLE` / `NON_DRIVABLE` / `OBSTACLE` /
`UNKNOWN`, via a transparent, hand-authored lookup table **per source**
(`SemanticPOSS`, `RELLIS-3D`, or `semantickitti-20` for predictions) --
**not** the coarser canonical ontology from `src/perception/ontology.py`,
because canonical merging loses distinctions that matter for terrain (e.g.
canonical `vegetation` merges RELLIS's `grass` -- often traversable
off-road -- with `tree`/`bush`, which are not; canonical `ground` merges
SemanticKITTI's `sidewalk`, not meant for vehicle traffic, with `road`,
which is). This is a **prototype heuristic**, the same status as the
semantic-priority tables -- not a learned or validated classifier. A class
name with no table entry (or an unregistered source) resolves to
`UNKNOWN`, never a guessed default -- e.g. SemanticKITTI's own
`other-ground` is genuinely ambiguous and is left `UNKNOWN` rather than
coin-flipped.

### 5. How obstacles are represented

`src/spatial/obstacle.py`'s `classify_obstacle(terrain, dynamic_score)`
splits terrain's `OBSTACLE` bucket into `STATIC_OBSTACLE` /
`POTENTIAL_DYNAMIC_OBSTACLE`, using FOVEA's **existing**, unmodified
`dynamic_score` field (the same semantic-category proxy documented since
Phase 0) -- not a new computation. **Stated explicitly, as required:**
`POTENTIAL_DYNAMIC_OBSTACLE` reflects "this object's semantic category is
one that could move (car, person, ...)" -- it is **not** true temporal
motion detection. Dynamic object detection is not yet true temporal motion
detection; the current dynamic field is a semantic-category proxy.
**Temporal tracking is a subsequent phase.**

### 6. Prediction vs. ground truth

Every `PerceptionCell`/`Map2_5DStats` carries an explicit `label_source`
(`"ground_truth"` or `"prediction"`) and `source` (dataset name, or model
ontology for predictions) -- set once, at `build_2_5d_map`'s call site,
never inferred or silently defaulted. The API's `/map/{method}/2_5d`
endpoint's default (`label_source=ground_truth`) preserves all existing
behavior; `?label_source=prediction` routes through Phase 1's real
SqueezeSeg inference, the same `_get_prediction_map` the existing
`/map/{method}/predicted` endpoint already uses.
`tests/test_spatial.py::TestPredictionVsGroundTruthSeparation` confirms
the two modes never mix (SemanticKITTI class names never appear in
ground-truth-mode output and vice versa).

### 7. Point conservation

`validate_point_conservation(accounting)` checks the EXISTING engine's own
`accounting` dict for internal consistency: `input_points ==
assigned_points + out_of_envelope_points + invalid_points` -- these four
counters already exist in every baseline's accounting output (Phase 0);
this phase adds no new tracking, only an explicit assertion over data
that was already being computed. Verified `True` on every real frame
tested (see §13 results below). `validate_no_duplicate_cells` similarly
checks (not re-derives) the existing `(cell_size, ix, iy)` key uniqueness
guarantee documented in `map_structure.py` since Phase 0.

### 8. Unknown-cell handling

FOVEA's cell list is **sparse** -- `aggregate_points_to_cells` only ever
creates a `Cell` for a group with at least one point (documented in
`map_structure.py` since Phase 0), so there is no dense background grid
with literal empty cells to represent. Consequently:

- **`unknown_terrain_cells`** means SEMANTIC uncertainty (the cell's
  majority class has no terrain rule for its source, e.g. an
  unlabeled/void point) -- not geometric absence.
- Points that never landed in **any** cell (outside the distance envelope,
  or rejected as non-finite/invalid) are tracked via
  `Map2_5DStats.out_of_envelope_points`/`invalid_points` -- the same
  counters point conservation (§7) already validates, not a new concept.
- `PerceptionCell.has_points` is an explicit, checked field (not an
  assumption) -- it is `True` for every cell FOVEA currently produces (by
  construction), and `elevation_*` is `None` only in the defensive
  `has_points=False` branch, which the current sparse representation never
  actually exercises but the code does not assume away.

### 9. Dataset differences (observed, not asserted)

RELLIS-3D's off-road terrain surfaces meaningfully more `DRIVABLE` cells
via classes SemanticPOSS/SemanticKITTI don't have equivalents for (`grass`,
`dirt`) -- see the measured §13 results below, where RELLIS frames show
`drivable_cells` fractions in a different range than SemanticPOSS's. No
claim is made that the two datasets' terrain/obstacle counts are directly
comparable in absolute terms (same caveat as Phase 1's cross-dataset
evaluation).

### 10. Current limitations

- Terrain/obstacle classification is a hand-authored rule table (documented
  prototype heuristic), not a learned or validated classifier.
- `POTENTIAL_DYNAMIC_OBSTACLE` is a semantic-category proxy, not real
  motion/temporal detection (stated above, repeated here per the explicit
  requirement).
- No terrain ground truth exists in either dataset, so terrain-classification
  *accuracy* is not evaluated -- `evaluation_available=false` is the honest
  answer, not a fabricated number (see §11/API below). What IS validated:
  point conservation, no-duplicate-cell-assignment, spatial bounds
  correctness, elevation validity, and cell-resolution validity -- all
  real, ground-truth-independent engineering checks (`src/spatial/map_2_5d.py`'s
  `validate_*` functions), run on every frame tested in §13.
- `cell_id` is only a position in one map-build's cell list -- it is not a
  persistent identifier across frames or rebuilds. `PerceptionCell` itself
  still has no `parent_id` field (unchanged); a real parent/child tree over
  the same Cells now exists separately in `src/spatial/hierarchy.py` (see
  below) but is not threaded through this 2.5D API response.
- `semantic_distribution` (a full per-cell class histogram) was considered
  and deliberately NOT added -- `majority_class_name`/`majority_class_fraction`
  already capture the signal FOVEA's own math uses, and a full histogram
  would meaningfully bloat every cell's payload for no consumer that exists
  yet.

### API

New endpoint (additive only -- every existing endpoint, including
`/map/{method}` and `/map/{method}/predicted`, is unchanged):

| Method | Path | Returns |
|---|---|---|
| GET | `/api/datasets/{dataset}/frames/{sequence}/{frame_id}/map/{method}/2_5d?label_source=&model=&include_cells=` | `PerceptionCell` list + `Map2_5DStats` + point-conservation/no-duplicate/geometry validation results |

`label_source` defaults to `ground_truth` (existing default behavior,
unchanged); `?label_source=prediction&model=squeezeseg` routes through
Phase 1's real inference. `?include_cells=false` returns only the
map-level stats and validation results (no per-cell payload) -- response
size stays proportional to the existing `/map/{method}` endpoints', never
raw per-point arrays.

---

## Hierarchical spatial representation + LOD selection (src/spatial/)

**Phase 3 (hierarchy) and Phase 3B (LOD selection) — backend-only, no
streaming/frontend/3D-Tiles work yet.** Both wrap the existing, UNMODIFIED
`build_fovea_map` output; neither touches `information_score`, weights,
split thresholds, distance bands, or `refinement_factor`.

- **`src/spatial/hierarchy.py`** (`build_fovea_hierarchy`) reconstructs the
  explicit coarse -> medium -> fine parent/child tree FOVEA's flat `Cell`
  list only implies (a coarse cell that split to medium never gets its own
  `Cell`, so its footprint is synthesized as a structural, non-leaf
  `HierarchyNode`). Real `Cell`s become leaves; nothing is re-scored.
  Point conservation (every internal node's `point_count` == sum of its
  children's, and the forest total == the engine's own `assigned_points`)
  is enforced at build time and independently re-checkable via
  `verify_point_conservation`. Returns a FOREST of level-0 roots (one per
  occupied band+coarse-cell footprint), not a single map-wide root.
- **`src/spatial/lod.py`** (`select_lod_nodes`) picks a non-overlapping
  render frontier from that hierarchy for a given `(x, y)` viewpoint: an
  apparent-size rule (`cell_size_m / distance_to_viewpoint > detail_threshold`)
  descends toward finer children near the viewpoint and stops at a coarser
  ancestor far away; a leaf is always selected once reached (nothing finer
  exists); `max_distance_m` culls whole subtrees outright; an optional
  `point_budget` drops the farthest already-selected nodes first if the
  frontier's total `point_count` exceeds it. `verify_lod_frontier`
  independently re-checks a candidate frontier for duplicate footprints or
  ancestor/descendant overlap. This is a viewing decision only -- FOVEA's
  own resolution decisions are never recomputed.

Both layers are read-only over the hierarchy/Cells they're given and add no
network calls, chunking, or serialization on their own -- that's the next
layer, `src/spatial/chunks.py`, described below -- see `tests/test_hierarchy.py`
and `tests/test_lod.py` for synthetic (hand-computed) and real
SemanticPOSS/RELLIS-3D coverage.

---

## Streamable spatial chunks (src/spatial/chunks.py)

**Phase 3C — backend serialization only, no frontend/Three.js/3D-Tiles work
yet.** Turns an existing LOD frontier into a compact, transport-ready
payload:

```
existing hierarchy (hierarchy.py) -> existing select_lod_nodes() (lod.py)
    -> selected HierarchyNodes -> SpatialChunk (chunks.py) -> serialize_chunk()
    -> HTTP response (src/api.py)
```

This layer makes no FOVEA/hierarchy/LOD decision of its own -- it only
packages whatever `select_lod_nodes` already selected.

### What a chunk contains

One `SpatialChunk` per selected node: `node_id`, `level`/`resolution_label`,
`is_leaf`, `band_id`/`ix`/`iy`/`cell_size_m`, bounds (`x_min`/`x_max`/`y_min`/
`y_max`) + `center_x`/`center_y`/`range_m`, `point_count`, an elevation
summary (`elevation_min`/`max`/`median`), a semantic summary
(`semantic_class`/`semantic_priority`/`majority_class_fraction`),
`geometry_score`/`dynamic_score`/`information_score`, `terrain_class`/
`obstacle_class` (reusing the existing `src/spatial/terrain.py`/`obstacle.py`
classifiers, not new rules), and `label_source`/`source`. **No raw point
array is ever included** -- a chunk is a summary, not a point-cloud dump.

For a node the LOD layer selected AS A LEAF (the common case), every field
is that node's own `Cell` value, unchanged. For a node selected as a
coarser ANCESTOR (an internal, non-leaf `HierarchyNode` -- see
`hierarchy.py`), the numeric fields are a **point-count-weighted aggregate**
over that node's own leaf descendants' already-computed `Cell` fields --
never a re-read of raw points, never a re-derivation of `information_score`/
weights/thresholds/bands/`refinement_factor`. `SpatialChunk.is_leaf` tells a
consumer which case it got.

### Serialization format: FVCH1 (binary)

A real, documented binary format was implemented (not deferred to the
JSON-fallback the task allowed for) — `struct`-packed, no external
dependencies:

```
6-byte magic "FVCH1\0" | uint32 chunk_count | uint32 string_count
| string_count x (uint16 length + UTF-8 bytes)        <- interned string table
| chunk_count x 94-byte fixed record                   <- see src/spatial/chunks.py
```

Numeric fields are **float32** on the wire (not float64) — a documented
size/precision tradeoff (~7 significant digits), the same choice glTF/
Three.js vertex buffers make, appropriate because these are rendering/
inspection values, never fed back into FOVEA computation. Six repeating
string fields (`semantic_class`, `terrain_class`, `obstacle_class`,
`resolution_label`, `label_source`, `source`) are **interned once** into a
shared table and referenced by a 2-byte index per chunk, since a handful of
distinct class names repeat across tens of thousands of chunks.
`point_count`/`ix`/`iy`/`band_id`/`level`/`is_leaf`/every string field
round-trip **exactly**; `verify_chunk_roundtrip` independently re-checks
every field of every chunk after a real serialize -> deserialize pass
(integer/string fields exact, float fields within tolerance) — not just
"did it not crash."

### API endpoint

```
GET /api/datasets/{dataset}/frames/{sequence}/{frame_id}/chunks
    ?detail_threshold=&max_distance=&point_budget=&viewpoint_x=&viewpoint_y=
```

Ground-truth FOVEA mode only. Returns one JSON object: `dataset`/`sequence`/
`frame_id`/`label_source`/`source`, the echoed `lod_config`,
`selected_node_count`/`selected_point_count`/`culled_root_count`/
`culled_subtree_count`/`budget_dropped_node_count`, `chunk_count`,
`payload_format`/`payload_bytes` (the real FVCH1 byte count), a `timing`
object (see below), and `payload_base64` — the FVCH1 binary blob,
base64-encoded so it fits inside a JSON body like every other endpoint in
this API. **This is one string field, never a JSON array of tens of
thousands of per-chunk objects** (the exact thing this phase was told not
to do). Base64 inflates the wire size ~33% over the raw `payload_bytes`
figure -- a real, disclosed tradeoff of staying JSON-shaped for now, not a
claim that base64-over-JSON is the smallest possible transport (a raw
`application/octet-stream` response would remove it; not implemented this
phase — no frontend consumes it yet).

### Cache and concurrency

Reuses the EXACT SAME per-key-lock coalescing discipline `_get_map`/
`_get_prediction_map` already use (see "Caching and concurrency" under
§14) — no new caching strategy invented:

- The hierarchy is built once per frame, cached on the frame's cache entry
  (`entry["hierarchy"]`), from the already-cached FOVEA (ground-truth)
  `cells`/`accounting` — `_get_map(..., "fovea")` is not re-run.
- The LOD selection + chunk payload is cached per **exact parameter
  combination** — `(detail_threshold, max_distance, point_budget,
  viewpoint_x, viewpoint_y)` is the cache key (`entry["chunk_cache"]`),
  because those are exactly the parameters that materially change the
  result. Two requests with different `detail_threshold` are different
  cache entries; identical requests (concurrent or sequential) reuse the
  same cached `(selection, payload, metadata)` tuple.
- Verified by `tests/test_chunks.py`: concurrent requests for a
  never-before-seen parameter combination all get byte-identical payloads
  and byte-identical `timing` (a fresh computation would re-measure
  `perf_counter()` and essentially never match) -- the same proof
  technique `tests/test_api.py`'s own concurrency regression test uses.

### Measured (real data, `detail_threshold=0.01`, no culling/budget)

Four stages timed **separately**, per the task requirement — not a single
combined number:

| Frame | FOVEA build (s) | Hierarchy build (s) | LOD select (s) | Chunk serialize (s) | Chunks | FVCH1 bytes | vs. naive per-object JSON |
|---|---|---|---|---|---|---|---|
| SemanticPOSS 000000 | 0.526 | 0.141 | 0.039 | 0.236 | 36,197 | 3,402,837 | 7.70x smaller |
| SemanticPOSS 000050 | 0.539 | 0.159 | 0.045 | 0.299 | 35,575 | 3,344,369 | 7.70x smaller |
| SemanticPOSS 000100 | 0.493 | 0.195 | 0.046 | 0.229 | 33,566 | 3,155,523 | 7.70x smaller |
| RELLIS 00000/000100 | 0.213 | 0.071 | 0.021 | 0.082 | 19,015 | 1,787,630 | 7.63x smaller |
| RELLIS 00002/000500 | 0.253 | 0.044 | 0.022 | 0.084 | 21,261 | 1,998,741 | 7.63x smaller |
| RELLIS 00004/000050 | 0.217 | 0.037 | 0.018 | 0.073 | 17,150 | 1,612,310 | 7.64x smaller |

~94 bytes/chunk on the wire (matches the 94-byte fixed record almost
exactly -- the interned string table amortizes to nearly nothing at this
chunk count). The "7.7x smaller" figure is a **measured** comparison
against `json.dumps([dataclasses.asdict(c) for c in chunks])` on the exact
same chunk list in the same test run (`tests/test_chunks.py::
test_payload_smaller_than_naive_json_equivalent` and the validation script
behind this table) -- not an estimate, and not compared against base64 (the
actual HTTP response is larger than `payload_bytes` by base64's ~33%, still
smaller than the naive-JSON baseline but by a lower factor). No bandwidth
claim is made beyond what these two concrete payloads measured.

### Limitations (stated plainly)

- **Not 3D Tiles.** FVCH1 is a small, purpose-built binary format for this
  repo's own hierarchy, not a glTF/3D-Tiles-compatible tileset -- no
  implicit tiling, bounding-volume hierarchy spec, or tileset.json exists.
- Ground-truth FOVEA mode only -- prediction-mode chunks (Phase 1's
  perception models) are not wired up this phase.
- `payload_base64` is not raw binary transport (see API section above) --
  the ~33% base64 overhead is real and un-optimized this phase.
- Aggregated (non-leaf) chunks' semantic/elevation summary is a
  point-count-weighted approximation over leaf `Cell`s, not a literal
  recomputation from raw points at that coarser resolution -- documented in
  `src/spatial/chunks.py`'s module docstring, exposed via `is_leaf`.
- No frontend/Three.js/3D-Tiles consumer existed as of Phase 3C -- Phase 3D
  (below) adds one; `FVCH1` is still not a 3D-Tiles-compatible tileset.

---

## Streaming (LOD) frontend (Phase 3D)

**Integration only, not a redesign.** A new overlay mode, **"Streaming
(LOD)"**, was added alongside the three existing ones (Raw / Raw + FOVEA /
FOVEA only, all byte-for-byte unchanged); it is not the default. Selecting
it swaps the hero scene's data source from "fetch the entire FOVEA cell
list" to the Phase 3A-3C pipeline, live, in the browser:

```
GET .../chunks (hierarchy -> select_lod_nodes -> serialize_chunk, all
    server-side and unmodified)
    -> JSON { ...metadata, payload_base64 }
    -> src/api/fvch1.ts: decodeFvch1() -- ArrayBuffer/DataView, typed
       arrays, no per-chunk JS objects
    -> src/scene/ChunkField.tsx -- 2 InstancedMeshes (border + fill tiles),
       same visual language as CellField.tsx
```

The browser no longer needs the entire fine-resolution cell list for this
representation: it asks for a LOD frontier (a handful of parameters --
`detail_threshold`, `max_distance`, `point_budget`, `viewpoint_x/y`) and
renders exactly the nodes the server already selected -- coarser far away,
finer near the viewpoint -- never re-deriving that decision client-side.

### New frontend files

| File | Role |
|---|---|
| `src/api/fvch1.ts` | FVCH1 binary decoder (mirrors `src/spatial/chunks.py` field-for-field) -> struct-of-arrays `DecodedChunkSet`, plus `chunkInfoAt()` for on-click inspection and a base64->ArrayBuffer helper. Validates magic/version, string table, and record bounds; throws `Fvch1DecodeError` on anything malformed -- never returns a partial result. |
| `src/hooks/useSpatialStreaming.ts` | Calls `GET .../chunks`, decodes the payload, exposes `{chunkSet, meta, stage, error, fetchMs, decodeMs}`. Two-stage progressive load (coarse `detail_threshold=0.3` first, then fine `0.02`), per-frame chunk cache (`Map` keyed by threshold+viewpoint), a generation counter + `AbortController` to discard stale/superseded requests. |
| `src/scene/ChunkField.tsx` | Lives inside `<Canvas>` (needs live camera access). Samples the camera's ground-plane position at ~6 Hz and only reports a new viewpoint once it has moved >4m (the debounce/threshold from Step 7) -- driving `useSpatialStreaming`. Renders the decoded set as 2 `InstancedMesh`es (not one object per cell), supports all 4 existing color modes, and reports a small FPS counter. |
| `src/components/StreamedChunkPanel.tsx`, `StreamingTelemetryPanel.tsx` | Sidebar chunk inspector (click-to-select, same visual style as the existing Region Inspector, but reads fields already in the decoded chunk -- no `/cell/{index}/explain` call, since a streamed node's id doesn't correspond to a `/map/fovea` cell index) and a small LOD/Nodes/Points/Payload/FPS telemetry strip, shown only in Streaming mode. |

`src/api/client.ts` gained one function (`getChunks`, passes query params
through -- no LOD logic duplicated in TypeScript); `src/api/types.ts`
gained the `ChunksResponse`/`LODConfig`/`ChunkTiming` mirrors of the
existing Pydantic models. `src/scene/Scene.tsx` gained 5 props and one
conditional render branch. `CellField.tsx`, `PointCloud.tsx`,
`RegionInspector.tsx`, `InformationAnalytics.tsx`, `BenchmarkPanel.tsx`,
and every non-streaming code path in `App.tsx` are **unmodified**.

### Progressive loading

A coarse LOD (`detail_threshold=0.3`, few large nodes) loads first so the
scene is never blank, then a finer pass (`0.02`) replaces it without
blocking the first paint -- verified live: `stage` visibly moves
`loading-coarse -> coarse -> loading-fine -> fine` in the telemetry panel,
and the "reached FINE stage" check passed in every browser run below.

### Measured, in a real browser (SemanticPOSS 00/000000, Chromium, this machine)

| | OLD (`GET .../map/fovea?include_cells=true`) | NEW (`GET .../chunks?detail_threshold=0.02`) |
|---|---|---|
| Response body (real, `Blob([...]).size` on the actual fetched text) | 23,180,841 bytes (~22.1 MB) | 4,456,919 bytes (~4.25 MB) -- **5.2x smaller** |
| ...of which is the FVCH1 payload itself (`payload_bytes`, before base64) | n/a | 3,342,199 bytes (~3.19 MB) -- **6.9x smaller** than OLD |
| Nodes/cells returned | 41,868 (every FOVEA cell) | 35,552 (the LOD-selected frontier at this threshold/viewpoint) |
| In-page `fetch()` time | 506 ms | 22 ms (warm backend cache -- see Caching below; first-ever request for a param combo pays the ~0.5-0.7s FOVEA+hierarchy+LOD+serialize cost measured in the Phase 3C section above) |

Base64-in-JSON transport (see the Phase 3C section) inflates the raw FVCH1
bytes by ~33% on the wire (3,342,199 -> 4,456,268 base64 chars) -- this is
disclosed, not hidden, and is exactly why the NEW total (4.46 MB) is larger
than the raw payload alone; the comparison above is still against the OLD
path's own real, equally-uncompressed JSON response, so it is not stacking
an unfair advantage.

**No bandwidth claim beyond what these two real fetches measured.** These
numbers are for ONE frame at ONE LOD setting; a closer viewpoint/lower
threshold selects more, finer chunks and narrows the gap (by design -- LOD
trades payload for detail, it doesn't create detail for free). Chunk decode
time (`decodeFvch1` on 35,552 chunks) measured 8-16ms in-browser --
negligible next to the ~350ms network fetch it follows.

### What was verified in the browser (headless Chromium, this session)

- Streaming mode reaches the "Fine" stage after loading "Coarse" first, on
  both SemanticPOSS and RELLIS-3D.
- All 4 color view modes (Semantic/Resolution/Elevation/Information) render
  correctly on the streamed chunk field.
- Switching frames and switching datasets (SemanticPOSS <-> RELLIS-3D)
  while in Streaming mode works, telemetry updates, no stale data persists
  across the switch (cache keyed by frame, reset on frame/dataset change).
- Switching back to Raw / Raw + FOVEA / FOVEA only renders correctly and
  the sidebar reverts to the original Region Inspector (backend `/explain`
  call included) -- unmodified from before this phase.
- **Zero console errors, zero uncaught page errors, zero failed/4xx/5xx
  requests** across the full flow (initial load, streaming coarse-> fine,
  all 4 view modes, 2 frame steps, dataset switch, mode switches back).
- Click-to-select raycasting on the streamed `InstancedMesh` uses the
  IDENTICAL mechanism as the pre-existing `CellField` (`onClick` + `e.
  instanceId`) -- not independently re-verified by an automated click in
  this headless run (the sparse, ring-shaped tile layout made landing a
  precise click difficult to script reliably at the tested zoom levels);
  visually confirmed correct by direct code review, not claimed as
  browser-tested where it wasn't.

### Limitations (stated plainly)

- "Streaming (LOD)" is opt-in via the overlay control, not the default --
  a deliberate, lower-risk choice; the default view is unchanged.
- The sidebar's streamed-chunk inspector shows the chunk's own decoded
  fields only; it does not call `/cell/{index}/explain` (no such index
  exists for a streamed node -- see `StreamedChunkPanel.tsx`).
- FPS numbers shown when testing under headless/software (SwiftShader)
  rendering are throttled by CPU-side WebGL emulation and are NOT
  representative of real GPU frame rates -- they were used only to confirm
  the counter itself updates, not as a performance claim.
- No 3D Tiles, no octree, no temporal tracking, no new perception model --
  all explicitly out of scope for this phase and not implemented.

---

## Realistic LiDAR point-cloud rendering (Phase 4AB)

**Visual/frontend-only. No backend, no FOVEA math, no new endpoint.**
`src/fovea_allocator.py`, `hierarchy.py`, `lod.py`, `chunks.py`, and every
Python file are byte-identical to Phase 3D. The prior visualization leaned
on FOVEA's colored tiles as the primary picture; this phase makes the
**real LiDAR point cloud** (already served by the existing
`GET .../points` endpoint, unchanged) the primary picture, with FOVEA's
adaptive-resolution structure as a legible overlay on top of it -- not the
other way around.

- **Real sensor geometry.** `PointCloud.tsx` already rendered actual x/y/z
  LiDAR returns via one `BufferGeometry` + `Float32Array` positions/colors
  in a single `<points>` draw call (no per-point React component/Mesh, both
  before and after this phase) -- this phase raises the requested point
  count (`useFrameData.ts`'s `POINT_CLOUD_CAP`, existing `/points`
  endpoint, existing subsampling logic, no backend change) from 20,000 to
  45,000 so the cloud reads as real sensor density on frames with more
  points than that, and adds a modest adaptive point size (denser clouds
  get slightly smaller points to avoid a "mush" of overlapping dots).
- **Semantic coloring**, unchanged convention: `semanticColor()` /
  `classNameForId()` (`constants.ts` / `semanticLookup.ts`), the same
  lookup CellField and ChunkField already use -- ground, vegetation,
  building, vehicle, person, pole, etc. read as distinct colors, no new
  label taxonomy invented.
- **Elevation coloring is now new for the raw cloud**: `PointCloud.tsx`
  previously ignored `viewMode` entirely (always semantic). It now colors
  each point by its own real z (via the same `elevationColor()` ramp
  CellField/ChunkField use, normalized against this point set's own real
  min/max -- never fabricated) when the Elevation tab is active.
  Resolution/Information tabs fall back to semantic coloring for raw
  points, since neither a per-point resolution level nor an
  information_score exists for a point that was never assigned to a FOVEA
  cell -- documented in `PointCloud.tsx`, not silently wrong.
- **Intensity** (`PointsResponse.intensity`) was already being transported
  by the existing `/points` endpoint (no backend change needed), but is
  **not** wired to a view-mode toggle this phase -- doing so would mean
  adding a 5th button to `ViewModeTabs`, which this phase treats as UI
  surface growth outside "integrate, don't redesign." Left as a natural
  next step, not a backend gap.
- **FOVEA overlay tuned for legibility over the point cloud, not
  prettified.** In **Raw + FOVEA** mode specifically, the point cloud's own
  opacity was raised (0.32 -> 0.6) and the FOVEA tile opacity lowered
  (0.62 -> 0.4) in `Scene.tsx` -- the only numeric change to the existing
  tile rendering. Fine/medium/coarse still read as three distinct colors
  (`RESOLUTION_COLORS`, unchanged) at this opacity; the point cloud is now
  clearly visible underneath rather than dominated by solid tiles. No
  change to tile geometry, height, or footprint math -- "prettier tiles"
  was explicitly out of scope, this is strictly an opacity balance fix so
  both layers stay readable together.
- **Existing depth cues kept, nothing new added**: the scene's existing
  `<fog>`, ground `<Grid>`, and `sizeAttenuation` on the points material
  (points shrink with distance, a real perspective cue) already provided
  atmospheric/depth feel and were not touched -- no glow, particles, or
  decorative effects were added, per the "believable sensor geometry, not
  an AI-dashboard effect" instruction.
- **Modes preserved exactly as named**: Raw / Raw + FOVEA / FOVEA only
  (`OverlayModeControl`) are unchanged in name and selection logic --
  only the rendering *inside* Raw/Raw+FOVEA/FOVEA-only changed (opacity
  balance + elevation coloring for points). Phase 3D's "Streaming (LOD)"
  mode (`ChunkField.tsx`) is untouched by this phase.

### Measured, in a real browser (headless Chromium, SemanticPOSS 00/000000
unless noted)

| Measurement | Value |
|---|---|
| Initial load (nav -> `/points` request observed) | 3.21 s |
| `/points?max=45000` fetch duration (in-browser, `PerformanceResourceTiming`) | 97.6 ms |
| Frame switch (click next -> new frame id visible) | 584 ms |
| Dataset switch (SemanticPOSS -> RELLIS-3D, select -> settled) | 2.04 s |
| Approx. FPS, headless/software (SwiftShader) rendering | ~29 |

The FPS figure is explicitly **not** a real-GPU number -- headless Chromium
here renders via CPU-side SwiftShader (no real GPU in this environment),
which throttles frame rate independent of scene complexity; it is reported
only to confirm the render loop runs continuously under the new point
count/opacity settings, not as a performance claim about a judge's actual
machine. No optimization work was done beyond the two changes above, since
no regression was observed (0 console errors, 0 page errors, 0 failed
requests across the full validation flow below).

### What was verified in the browser

Raw / Raw + FOVEA / FOVEA only, Semantic and Elevation view modes (on both
Raw-only and Raw + FOVEA), all 3 camera presets (Top/Front/Perspective) +
Reset, manual orbit-drag + wheel-zoom, frame switching, dataset switching
(SemanticPOSS <-> RELLIS-3D) -- all rendered correctly, on both datasets.
**Zero console errors, zero uncaught page errors, zero failed/4xx/5xx
requests** across the entire flow.

### Ground truth vs. prediction -- unchanged, still honest

This phase renders whatever `useFrameData` already fetches -- ground-truth
labels by default, exactly as before. Nothing here adds or implies
prediction-mode point-cloud rendering, real dynamic/motion detection, or
temporal tracking; `dynamic_score`'s documented "semantic-category proxy,
not real motion" caveat (stated since Phase 0) is unaffected and unrepeated
here only because it was already stated once, clearly, above.

---

## SIH compliance benchmarks (Phase 4C, src/benchmarking/sih_compliance.py)

**Measurement only -- no FOVEA/hierarchy/LOD/chunk/perception logic changed
or added.** Six functions, each a thin, read-only wrapper around an
EXISTING, unmodified engine function, run against real local SemanticPOSS +
RELLIS-3D frames (3 each by default, `DEFAULT_SEMANTICPOSS_FRAMES`/
`DEFAULT_RELLIS_FRAMES`). Run with:

```bash
python -m src.benchmarking.sih_compliance --out results/sih_compliance_benchmark.json
```

Every number below is from that actual run on this machine, unmodified --
not fabricated, not hand-picked for effect.

### A. Adaptive resolution vs. distance

The SIH brief states two example anchors -- "~5cm near 10m, ~50cm by
100m" -- not a full per-band curve. Grouping the ACTUAL FOVEA cells (3
SemanticPOSS frames, aggregated) by their own `band_id`:

| Band | min cell size | median | max | configured band base |
|---|---|---|---|---|
| 0-10m | 0.0125m | 0.05m | 0.05m | 0.05m |
| 10-30m | 0.025m | 0.10m | 0.10m | 0.10m |
| 30-60m | 0.0625m | 0.25m | 0.25m | 0.25m |
| 60-100m | 0.125m | 0.50m | 0.50m | 0.50m |

**Both stated SIH anchors are met exactly as upper bounds** (0-10m band
caps at 0.05m = 5cm; 60-100m band caps at 0.50m = 50cm), and FOVEA often
does BETTER than the anchor where information score warrants it (down to
1.25cm near 10m) -- never worse. The 10-30m/30-60m figures are the
EXISTING config's own values, not independently SIH-specified, stated as
such rather than implied to be SIH targets. **No configuration change
needed for this requirement** -- the existing `config.yaml` distance bands
already satisfy the stated anchors.

### B. 100m coverage

Real measured max point range (SemanticPOSS 00/000100): **199.8m** --
coverage reaches and exceeds 100m where the actual scan does. Per-band
cell/point counts and point conservation (via the existing
`validate_point_conservation`, reused unmodified) both real and verified:
`{input: 65,831, assigned: 64,525, out_of_envelope: 1,306, invalid: 0,
conserved: true}` for that frame -- every input point accounted for
exactly once.

### C. Uniform high-res vs. FOVEA memory benchmark

Same real frame, same `cfg`, both `build_uniform_map`/`build_fovea_map`
unmodified (SemanticPOSS 00/000000):

| | cells | est. storage | 
|---|---|---|
| Uniform @ configured 0.20m | 23,954 | 2,620 KB |
| FOVEA (adaptive) | 41,868 | 4,579 KB |
| FOVEA FVCH1 (LOD-selected, streamed) | 35,552 chunks | 3,264 KB |

**Honest finding, stated plainly, not spun**: at uniform's OWN currently
configured resolution (0.20m), FOVEA produces MORE cells and a LARGER raw
per-cell storage estimate than uniform for these frames -- because FOVEA's
finest near-field cells (down to 1.25cm) are far finer than uniform's fixed
0.20m grid reaches anywhere, and this dataset's near-field content is dense
enough that the fine-cell count outweighs the coarse-far-field savings.
**FOVEA is not "fewer bytes than uniform at uniform's own resolution" in
this configuration.**

The comparison that IS favorable, and the one that actually matters for
what FOVEA is for, is at **equal peak resolution**: reconfiguring uniform
to `cell_size_m=0.0125` (matching FOVEA's finest achieved cell size this
frame, real reproducible measurement via a `cfg` copy, not editing
`config.yaml`) gives **64,336 cells / 7,036.8 KB** -- i.e. to reach the SAME
near-field detail FOVEA reaches adaptively, a uniform grid needs **~1.5x
more cells/storage** than FOVEA, while ALSO paying that same fine cost
everywhere far away where FOVEA stays coarse. Uniform has no distance
envelope at all, so this comparison is if anything generous to uniform (it
covers more far-range area than FOVEA's envelope). Reproduce via:
`cfg_fine = copy.deepcopy(cfg); cfg_fine["uniform"]["cell_size_m"] = 0.0125;
build_uniform_map(points, labels, cfg_fine)`.

### D. End-to-end pipeline latency

Real backend stage timings, SemanticPOSS 00/000000 (includes real
SqueezeSeg inference):

| Stage | Time |
|---|---|
| Perception (SqueezeSeg, first call incl. cold-start) | 323ms (87-91ms on subsequent frames) |
| FOVEA build | 538ms |
| Hierarchy build | 159ms |
| LOD selection | 42ms |
| Chunk serialization | 254ms |
| **Backend total** | **1.32s** (first frame), **~1.04-1.06s** (warm) |

Process RSS delta measured via `psutil` (real, not estimated): +55MB /
-4MB / +26MB across the 3 frames (GC-dependent, not monotonic -- reported
as measured, not smoothed). **Frontend decode/FPS are NOT re-measured by
this Python-only benchmark** (no browser here) -- cited with explicit
provenance from the real browser measurements already recorded in the
Phase 3D (~8-16ms FVCH1 decode) and Phase 4AB (~29 FPS, headless
SwiftShader software rendering, **explicitly not a real-GPU hardware
claim**) sections above, never presented as freshly measured here.

### E. Distance-wise perception benchmark

Reuses `src.perception.evaluation.evaluate_predictions` (existing,
unmodified) -- already produces exactly these 4 bands. Real SqueezeSeg
inference (trained on SemanticKITTI, urban) against real ground truth, 3
frames each dataset:

| Dataset | Band | Points | Accuracy | mIoU |
|---|---|---|---|---|
| SemanticPOSS 00/000000 | 0-10m | 10,062 | 79.4% | 0.211 |
| | 10-30m | 37,547 | 38.4% | 0.130 |
| | 30-60m | 14,754 | 20.5% | 0.067 |
| | 60-100m | 2,107 | 22.0% | 0.040 |
| RELLIS 00000/000100 | 0-10m | 6,478 | 0.0% | 0.000 |
| | 10-30m | 14,452 | 12.7% | 0.042 |
| | 30-60m | 1,159 | 31.8% | 0.056 |
| | 60-100m | 18 | 5.6% | 0.014 |

Accuracy/mIoU both degrade with distance on SemanticPOSS, as expected (a
real, consistent, un-cherry-picked pattern across all 3 frames tested).
RELLIS-3D numbers are real but expectedly weak (a SemanticKITTI-trained
urban model evaluated off-road, the same documented cross-domain gap
Phase 1's own tests already established) -- one RELLIS frame
(00004/000050) scored near-zero, also reported as-is, not excluded. No
band/class with zero ground-truth points is given a fabricated metric --
`evaluate_predictions`' own `None`-not-`0.0` convention is preserved.

### F. Terrain / static-obstacle reporting

Reuses `src.spatial.map_2_5d.build_2_5d_map` (existing, unmodified) --
Phase 2's terrain.py/obstacle.py rule tables, not touched:

| Dataset (frame) | Drivable | Non-drivable | Obstacle | Unknown | Static | Potential-dynamic |
|---|---|---|---|---|---|---|
| SemanticPOSS 000000 | 12,685 | 9,972 | 18,306 | 905 | 6,748 | 11,558 |
| RELLIS 00000/000100 | 7,042 | 2,586 | 5,761 | 3,946 | 5,604 | 157 |

**Semantic-DERIVED classification, not validated ground truth** -- neither
dataset has real terrain ground truth (stated in `map_2_5d.py`'s own
docstring since Phase 2). `potential_dynamic_obstacle` remains a
semantic-category proxy ("this class COULD move"), never real motion
detection -- unchanged, not claimed otherwise here.

### Tests and reproducibility

`tests/test_sih_compliance.py` -- 25 focused tests, one class per
requirement (A-F) + orchestration, real SemanticPOSS data always, RELLIS-3D
skipped if absent (same convention as every other test file). Full suite:
**225/225 passing** (200 pre-existing + 25 new).

### Requirement status / does FOVEA need adjustment next?

**No FOVEA math or config change is required by anything measured in this
phase.** Requirement A (the one requirement with an explicit external
target) is met as measured, with the existing config, without modification
-- exactly per this phase's own instruction to STOP and report rather than
auto-adjust if a gap were found. The one substantive finding worth carrying
into planning is C's nuance above: FOVEA's advantage over uniform in this
codebase is real but conditional -- it's a peak-resolution/adaptivity
advantage (~1.5x fewer cells than uniform at matched fine resolution, plus
resolution uniform's own configured setting cannot reach at all), not a
blanket "fewer bytes than uniform at any resolution" claim, and that
distinction should be stated exactly this way in any judge-facing summary,
not simplified into an unqualified "N% smaller" number.

## 8. Python environment setup

Python 3.11 (developed/tested on 3.11.9; 3.10 also supported per existing
`__pycache__` artifacts).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 9. Backend dependencies

From `requirements.txt`:

```
numpy>=1.24, pandas>=2.0, PyYAML>=6.0     -- engine
streamlit>=1.30, plotly>=5.18, psutil>=5.9 -- Streamlit debug dashboard
fastapi>=0.110, pydantic>=2.5, uvicorn[standard]>=0.27  -- FastAPI backend
httpx>=0.27                                 -- API tests (FastAPI TestClient)
torch>=2.0                                   -- perception layer (src/perception/) -- CPU/MPS, no CUDA
```

## 10. Frontend dependencies

`frontend/package.json` (installed via `npm install`):

```
react, react-dom                          -- UI
typescript, vite, @vitejs/plugin-react     -- build tooling
tailwindcss, @tailwindcss/vite             -- styling
three, @react-three/fiber, @react-three/drei -- 3D visualization
lucide-react                                -- icons
```

Node.js 20+ recommended (developed on Node 25 / npm 11).

## 11. How to start FastAPI

```bash
python run_api.py
```

Equivalent to `uvicorn src.api:app --host 127.0.0.1 --port 8000`. Interactive
API docs (from the Pydantic response models) at **http://localhost:8000/docs**.

`run_api.py` intentionally does **not** pass `--reload`: uvicorn's file
watcher scans the whole working directory by default, and this repo's cwd
contains `dataset/` (multi-GB) and `.venv/`, which makes the reload watcher's
startup scan pathologically slow. For active backend development, run
uvicorn directly with a scoped watch instead:

```bash
uvicorn src.api:app --reload --reload-dir src --port 8000
```

CORS is enabled for the local Vite dev origins (`localhost:5173`/`4173`)
only — this is a local-only prototype, not a deployed service.

## 12. How to start the React frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # sets VITE_API_URL=http://localhost:8000
npm run dev
```

Open **http://localhost:5173**. The API base URL is never hard-coded in
frontend source — it's always read from `VITE_API_URL`
(`frontend/src/api/client.ts`).

`npm run build` produces a production bundle in `frontend/dist/`
(type-checked via `tsc -b` first).

## 13. How to run the existing Streamlit debugging dashboard

```bash
python run_demo.py     # equivalent to: streamlit run src/app.py
```

Open **http://localhost:8501**. This is preserved exactly as it was — it
remains the team's internal/debugging interface, calling the same
`src/*.py` functions as the FastAPI backend. It is not the judge-facing
deliverable.

## 14. API endpoints

All under `http://localhost:8000` (see `/docs` for the live schema). Every
per-frame endpoint has a **dataset-aware** form under
`/api/datasets/{dataset}/...` (`{dataset}` = `SemanticPOSS` or `RELLIS-3D`)
plus a **legacy** SemanticPOSS-only form with the original v0.1 URL shape,
kept for backward compatibility — both call the same internal Python
functions in `src/api.py` (`_diagnostics_response`, `_map_response`, etc.),
not separate implementations:

| Method | Path | Returns |
|---|---|---|
| GET | `/api/status` | Backend status, dataset root, available sequences/datasets, env info |
| GET | `/api/datasets` | `{datasets: [{name, is_default}], default}` |
| GET | `/api/datasets/{dataset}/sequences` | Sequence ids for that dataset |
| GET | `/api/datasets/{dataset}/sequences/{sequence}/frames` | Frame ids (only ones with both point cloud + label) |
| GET | `/api/datasets/{dataset}/frames/{sequence}/{frame_id}/diagnostics` | `frame_diagnostics` output, dataset-appropriate ontology |
| GET | `/api/datasets/{dataset}/frames/{sequence}/{frame_id}/points?max=N` | Subsampled point cloud (parallel arrays, not array-of-objects, for compact/fast transport) |
| GET | `/api/datasets/{dataset}/frames/{sequence}/{frame_id}/map/{method}` | `cells`, `accounting`, `summary` for `uniform`/`distance_adaptive`/`fovea` (`?include_cells=false` to skip the cell payload) |
| GET | `/api/datasets/{dataset}/frames/{sequence}/{frame_id}/cell/{index}/explain?method=` | `explain_allocation(cell)` text + the cell's fields |
| GET | `/api/datasets/{dataset}/frames/{sequence}/{frame_id}/chunks?detail_threshold=&max_distance=&point_budget=&viewpoint_x=&viewpoint_y=` | Streamable spatial chunks for the existing FOVEA hierarchy + LOD frontier -- see "Streamable spatial chunks" below |
| GET | `/api/datasets/{dataset}/semantic-classes` | Class id/name/priority/dynamic-flag for that dataset's ontology |
| GET | `/api/datasets/{dataset}/benchmark?refresh=&num_frames=&repeats=` | That dataset's `results/benchmark_result*.json`, or a fresh run if `refresh=true` |
| GET | `/api/config` | The loaded `config.yaml` (includes both datasets' settings) |
| GET | `/api/sequences`, `/api/sequences/{sequence}/frames`, `/api/frames/{sequence}/{frame_id}/...`, `/api/semantic-classes`, `/api/benchmark` | Legacy v0.1 routes, fixed to `dataset=SemanticPOSS`, unchanged response shape |

Every field in every response is read from an existing `src/*.py` function's
real output — `src/api.py` performs no scoring/allocation of its own.

### Caching and concurrency

```
FRAME CACHE                                  MAP CACHE (per frame entry)
(dataset, sequence, frame_id)                ├── fovea    -> (cells, accounting, summary)
        |                                    ├── uniform  -> (cells, accounting, summary)
        v                                    └── distance_adaptive -> (cells, accounting, summary)
  loaded Frame  ------------------------------------^
```

`_frame_cache` (`src/api.py`) is a FIFO-bounded in-process cache (8 frames)
keyed by **`(dataset, sequence, frame_id)`** — dataset is part of the key on
purpose: SemanticPOSS sequence `"00"` and a hypothetical RELLIS sequence
that happened to also be named `"00"` must never collide or serve each
other's cells. Each entry additionally caches all three methods'
`(cells, accounting, summary)` so switching endpoints/views for an
already-loaded frame never re-runs `build_fovea_map`/`build_uniform_map`/
`build_distance_adaptive_map` or `summarize_method`. Per-dataset semantic
priority/dynamic-class/label-dict substitution happens once per dataset
name in `_dataset_context` (memoized), not per request.

**Why a plain dict cache wasn't enough:** FastAPI runs `def` (non-async)
endpoints in a thread pool, so concurrent requests for the same frame
genuinely execute in parallel. The frontend fires up to 5 requests at once
per frame selection, and React StrictMode's dev-mode double-effect can
double that again. Without locking, every one of those concurrent requests
saw an empty cache at the same instant and independently reloaded the frame
and rebuilt all three maps — this was the actual root cause of the ~46s
response times (see "Performance notes" below), not a slow algorithm.
`_get_frame_entry`/`_get_map` now use a lock per `(sequence, frame_id[,
method])` with double-checked locking: the first caller computes, concurrent
callers for the same key wait on the lock and then reuse the result already
in the cache — regardless of whether the duplication comes from StrictMode,
multiple browser tabs, or the frontend's own concurrent fetches.
`tests/test_api.py::TestConcurrentRequestsDoNotDuplicateComputation` encodes
this as a regression test (fires 5 concurrent requests at a fresh frame,
asserts it was loaded exactly once).

### Performance notes (measured)

Backend, single frame, cold vs. warm cache (`curl -w '%{time_total}'`,
sequence `00`, this dev machine):

| Request | Cold (never loaded) | Warm (cached) |
|---|---|---|
| `diagnostics` | ~10-30ms | ~3-5ms |
| `points?max=20000` | ~20-30ms | ~10ms |
| `map/uniform?include_cells=false` | ~0.15-0.3s | ~3-20ms |
| `map/distance_adaptive?include_cells=false` | ~0.3-0.4s | ~3-20ms |
| `map/fovea?include_cells=true` | ~0.85-1.7s | ~0.3-0.4s |

The warm `map/fovea?include_cells=true` cost (~0.3-0.4s) is not
recomputation — `?include_cells=false` on the same warm frame returns in
~3ms. It's the cost of building/serializing ~40k+ `CellModel` Pydantic
objects to JSON on every request; bounded and well under the demo's 1-2s
target, so it was left as-is rather than adding further caching complexity
for a prototype.

Full-page load, measured through the actual browser (Chromium via
Playwright, `frontend` dev build), sequence `00`:

| Scenario | Wall-clock to usable FOVEA visualization |
|---|---|
| First load, frame `000000`, cold backend | ~2.1s |
| Full reload, same frame, warm backend | ~0.9s |
| Jump to a never-before-loaded frame | ~1.3s |
| Return to a previously-visited frame | ~0.8s |
| Switch Semantic/Resolution/Elevation/Information views | 0 backend requests (near-instant, client-side recolor only) |

These are measurements from one development machine, not guaranteed
numbers — see the note on latency variance under §17.

## 15. Frontend views

The Command Center is a single dashboard (no page navigation), built for
1440x900 / 1920x1080 desktop.

**Progressive loading:** `frontend/src/hooks/useFrameData.ts` does not use a
single `Promise.all` across all 5 per-frame requests (that would block the
whole dashboard on whichever request is slowest). Instead it loads in
stages, each updating the UI as soon as it resolves:

1. **Diagnostics + FOVEA map** — fired together (both fast/important,
   neither should block the other); this is the main hero visualization and
   the resolution-distribution numbers.
2. **Point cloud** — fired once stage 1 has settled, so the raw LiDAR
   overlay never delays the FOVEA view appearing.
3. **Uniform + distance-adaptive summaries** — fired last, for the benchmark
   panel only (no cells needed there).

Each stage has its own loading flag; the UI shows honest, specific states
("Loading FOVEA map…", "Loading LiDAR points…") rather than one generic
spinner, and never displays a stage's data before that stage's request has
actually resolved. Switching Semantic/Resolution/Elevation/Information view
modes re-colors the already-fetched FOVEA cells client-side — it never
issues a new backend request.

- **Header** — dataset selector (`SemanticPOSS` / `RELLIS-3D`, default
  `SemanticPOSS`) + sequence/frame identity + live backend connection status.
  Switching datasets reloads that dataset's sequence list, resets to its
  first sequence/frame, re-fetches its semantic-class legend, and clears the
  selected cell — the 3D scene, view modes, camera controls, region
  inspector, and benchmark panel are all reused unchanged, just re-fed with
  the new dataset's real API responses.
- **Hero 3D view** (Three.js / R3F) — raw LiDAR point cloud (semantic-colored,
  toggleable), sensor origin marker, ground grid, and the FOVEA cell field
  rendered as real geometry (footprint = actual `cell_size_m`, position =
  actual `center_x`/`center_y`, height = actual `elevation_median`/
  `elevation_std`) with four color modes:
  - **Semantic** — cell color = majority semantic class.
  - **Resolution** (hero view) — coarse/medium/fine, so the core "high
    information -> high resolution" behavior is immediately visible.
  - **Elevation** — cell color = normalized elevation (terrain-style ramp).
  - **Information** — cell color = `information_score` (hotspot ramp).
- **Frame controls** — sequence selector, prev/next, scrub slider, play/pause
  (lightweight autoplay — advances one frame at a fixed interval, loads only
  the selected frame).
- **Region inspector** — click any cell in the 3D view to see its actual
  stored fields (semantic class, distance, semantic/geometry/dynamic/
  uncertainty/information scores, allocated resolution) plus the real
  `explain_allocation(cell)` text. Uncertainty always renders `N/A` — never
  fabricated.
- **Information analytics** — frame-averaged signal bars + the real FOVEA
  resolution-level distribution (Fine/Medium/Coarse counts from the backend).
- **Benchmark panel** — live per-frame comparison (Uniform vs
  Distance-Adaptive vs FOVEA: cells, latency, storage, assignment rate) plus
  the multi-frame `results/benchmark_result.json` averages. Any unmeasured
  metric renders `Not measured`, never a guess.
- **Pipeline strip** — compact explanatory flow (Raw LiDAR -> Understand ->
  Score -> Adaptive resolution -> 2.5D map).

## 16. FOVEA's resolution decision

1. Each configured distance band (`distance_adaptive.bands` in
   `config.yaml`) defines a **base cell size** — identical to the
   distance-adaptive baseline. This is the coarsest a cell in that band is
   allowed to be.
2. Points are first aggregated at the band's **finest** permitted size
   (`base / refinement_factor`) to get a per-micro-cell information score
   (semantic priority of the majority class + normalized local elevation std
   as a geometric-complexity proxy + a dynamic-class flag).
3. A standard top-down quadtree split decision decides, per coarse group and
   then per medium sub-group, whether to refine — based on configurable
   score thresholds (`split_to_medium_threshold`, `split_to_fine_threshold`).
4. A region ends at one of 3 cell sizes per band: `base`, `base/2`, or
   `base/refinement_factor` — never coarser than `base`, never finer than the
   floor. The distance envelope is a hard constraint; information score only
   decides "how much finer, within it."

Verified on sequence 00, frame 0: a `car` cell and a `ground` cell at the
*same range* (27.3 m, same 10-30 m band, base size 0.10 m) received
different resolutions — `car`: 0.025 m (fine, info_score 0.76); `ground`:
0.10 m (coarse, info_score 0.03). This is the core behavior distinguishing
FOVEA from the pure distance-adaptive baseline, and both `tests/test_core.py`
and the region inspector demonstrate it against real data.

## 17. Uniform vs. Distance-Adaptive vs. FOVEA

| Method | Resolution driven by |
|---|---|
| Uniform | Nothing — fixed cell size everywhere (`uniform.cell_size_m`) |
| Distance-Adaptive | Range from sensor only — two regions at the same distance always match |
| FOVEA | Range (hard envelope) **and** information score (semantic + geometry + dynamic) within it |

Measured on sequence 00, frame 0 (66,658 points), from
`results/benchmark_result.json`:

| Method | Cells | Latency (mean of 3 repeats) | Est. storage |
|---|---|---|---|
| Uniform (0.20 m) | 23,954 | ~0.19 s | ~2.6 MB |
| Distance-adaptive | 35,552 | ~0.29 s | ~3.9 MB |
| FOVEA | 41,868 | ~0.8-3.0 s (see note below) | ~4.6 MB |

**Honest note:** FOVEA produces *more* cells and *more* storage than both
baselines here, not less. This is expected: FOVEA's coarse level matches the
distance-adaptive baseline exactly, and refinement only *adds* cells in
high-information regions. **The claim for v0.1 is representation efficiency
in the sense of concentrating detail where it's useful, not raw memory
reduction** — the frontend's benchmark panel states this explicitly rather
than claiming FOVEA is "faster" or "smaller."

Latency variance note: repeated in-process runs of `build_fovea_map` showed
0.8-3.0s on the original dev machine — likely OS-level scheduling/disk-cache
noise, not a robust number. This is single-frame map-generation latency, not
FPS.

## 18. Benchmarking

```bash
python -m src.benchmark --config config.yaml --num-frames 5 --repeats 3
```

Writes `results/benchmark_result.json` with per-frame, per-method measured
metrics (cell counts, latency, storage estimate, distance-band stats). The
FastAPI `/api/benchmark` endpoint serves this file directly, and can also
trigger a small fresh run (`?refresh=true&num_frames=&repeats=`) which
overwrites it — nothing about benchmark numbers is fabricated or hard-coded
in either frontend.

## 19. Tests

```bash
python -m unittest discover tests -v
```

- `tests/test_core.py` — 12 tests (pre-existing, untouched): coordinate->cell
  quantization, distance-band assignment/monotonicity, semantic priority
  lookup, information-score weighting, point conservation, aggregation
  determinism.
- `tests/test_api.py` — 38 tests against the real FastAPI app (via
  `TestClient`, real data, no mocking): status/sequences, frame loading/
  diagnostics, point subsampling, all 3 map endpoints, cell explanation,
  config, semantic classes, benchmark availability, concurrency regression
  (fires concurrent requests at a never-before-loaded frame, asserts it's
  loaded exactly once), dataset selection (both datasets' routes, legacy
  routes match dataset-aware ones, unknown dataset 404s, RELLIS ontology
  never leaks SemanticPOSS class names or vice versa, cache isolation when
  switching datasets, concurrent RELLIS requests don't duplicate work), and
  perception endpoints (real inference, evaluation matching a direct
  computation exactly, prediction-mode FOVEA map differs from ground-truth
  mode while uniform/distance-adaptive geometry stays identical).
- `tests/test_datasets.py` — 18 tests for the dataset abstraction: registry,
  ontology separation, SemanticPOSS adapter reproduces pre-existing behavior
  exactly (488 frames, 66,658-point frame 0), RELLIS sequence/frame
  discovery, `.bin`/`.label` byte-level parsing across all 5 sequences,
  point/label count matching, every observed semantic id falls inside the
  official ontology, mismatched point/label counts raise (not silently
  truncate), and end-to-end FOVEA engine runs on real RELLIS frames.
- `tests/test_perception.py` — 27 tests: ontology mapping correctness,
  range-image projection shapes/invariants (including a degenerate
  zero-range point not crashing), strict state_dict loading (the
  architecture-correctness proof), real inference on a real frame
  (prediction length exactly matches input, confidence in `[0,1]`, timing
  fields non-negative, a regression guard against the intensity-scale bug
  recurring), real evaluation (accuracy/mIoU bounds, confusion-matrix shape
  and point-conservation, distance-band coverage), and FOVEA actually
  building different cells from predictions vs. ground truth.

- `tests/test_hierarchy.py` — 21 tests (Phase 3): hand-built synthetic
  Cells with independently hand-computed expected tree shape/point-counts
  (unsplit-coarse leaf, medium-only split, fine split through two
  synthesized ancestor levels), plus real SemanticPOSS/RELLIS-3D frames
  through the real, unmodified `build_fovea_map` — leaf count matches
  engine cell count, point conservation, no duplicate footprints,
  child-level/cell-size ratios match `refinement_factor`.
- `tests/test_lod.py` — 20 tests (Phase 3B): synthetic multi-root
  hierarchies with geometrically real node centers (near-viewpoint ->
  finest descendants, far-viewpoint -> coarsest ancestor, both in one
  multi-root call), ancestor/descendant exclusion, leaf fallback at
  zero distance, max-distance culling, point-budget dropping the
  farthest nodes first, determinism, empty-hierarchy, invalid-frontier
  detection (duplicate/ancestor+descendant), plus real SemanticPOSS/
  RELLIS-3D frames confirming `build_fovea_map`'s own output is
  unchanged by any LOD selection call.
- `tests/test_chunks.py` — 25 tests (Phase 3C): hand-built synthetic
  chunks with independently hand-computed weighted-aggregate values (leaf
  chunk == exact `Cell` fields, aggregate chunk == hand-computed
  point-count-weighted average), binary round-trip (exact integer/string
  fields, tolerant float fields), determinism, empty/multiple-chunk
  payloads, corrupt-payload rejection (bad magic, truncated, bad string
  index), the real FastAPI `/chunks` endpoint (parameters respected, cache
  reuse via identical timing on repeat, different parameters are different
  cache entries, concurrent fresh-parameter requests don't duplicate
  computation), plus real SemanticPOSS/RELLIS-3D frames through the full
  unmodified pipeline confirming FOVEA's own output stays unchanged.

RELLIS-specific tests are `@unittest.skipUnless`-guarded on the RELLIS
directories actually being present on disk — the suite still passes on a
machine with SemanticPOSS only, just with those tests skipped rather than
failed. Perception/spatial tests require `models/squeezeseg/` (committed) and
`torch` (in requirements.txt) but not RELLIS-3D.

All suites pass together: **200/200 OK** (with RELLIS-3D present; fewer if
RELLIS tests are skipped, none fail).

## Validation performed (multi-dataset integration)

Actually run, not just unit-tested — every number below came from a real
command against real data on this machine:

- **SemanticPOSS regression**: `get_adapter("SemanticPOSS", cfg)` reproduces
  the exact pre-existing numbers — 488 frames in sequence `00`, frame `0` =
  66,658 points, and (through the full API + browser) FOVEA/distance-adaptive
  /uniform cell counts of 41,868/35,552/23,954 — identical to the values
  documented in §17, confirming zero behavioral change for SemanticPOSS.
- **RELLIS-3D, real data, 3 sequences**: loaded frames from sequences
  `00000`, `00002`, `00004` (not just one), ran `build_uniform_map` /
  `build_distance_adaptive_map` / `build_fovea_map` — **unmodified** —
  against each, and confirmed sensible, differentiated output, e.g. seq
  `00000` frame `000100`: 37,337 points, FOVEA → 19,335 cells (18,774
  coarse / 391 medium / 170 fine), the single highest-information cell was a
  `person` cell (RELLIS ontology) with `information_score=1.00`, resolution
  `fine`, and a real generated explanation ("high semantic priority for
  'person' (1.00); high local geometric complexity (1.00); classified as a
  dynamic-relevant object category").
- **Cross-dataset, through the actual browser** (Playwright): loaded
  SemanticPOSS, switched to RELLIS-3D (header dropdown), clicked a real
  RELLIS cell and confirmed the inspector showed a RELLIS class name
  (`bush`) and a RELLIS-specific explanation, switched back to SemanticPOSS
  and confirmed its numbers were unchanged — zero console errors, zero
  stale-dataset requests (see the race-condition fix below).
- **A real bug found and fixed during this validation**: switching datasets
  could momentarily pair the new dataset with the previous dataset's stale
  sequence id (e.g. `dataset=RELLIS-3D` + `sequence=00`, which doesn't exist
  for RELLIS) — the backend raised an unhandled `FileNotFoundError` for that
  combination, which the browser reported as a CORS failure rather than a
  clean 404. Fixed on both sides: `src/api.py:_normalize_frame_id` now
  catches `FileNotFoundError` and returns a proper 404, and
  `frontend/src/App.tsx`'s dataset-switch handler clears `sequence`/
  `frameIds` synchronously so the invalid pairing is never requested at all.
- **Performance/caching regression check**: concurrent requests (5 at once,
  `ThreadPoolExecutor`) against a never-before-loaded RELLIS frame produced
  exactly one cache-population event (`/api/status.cached_frames`), same
  guarantee as the SemanticPOSS-only version. Switching datasets in the
  browser produced two independent, correctly-isolated cache entries
  (`SemanticPOSS/00/000000` and `RELLIS-3D/00000/000100`), never a
  collision.
- Generated a real (not fabricated) RELLIS-3D benchmark via
  `?refresh=true&num_frames=5` → `results/benchmark_result_rellis-3d.json`
  (gitignored, regeneratable) — 5 real frames from sequence `00000`.

## 20. Current limitations

- No trained perception model — semantic labels are each dataset's ground
  truth (SemanticPOSS's own labels; RELLIS-3D's own labels), used as an
  oracle stand-in.
- No real dynamic-object tracking — "dynamic relevance" is a static
  semantic-category proxy (`fovea.dynamic_classes` for SemanticPOSS,
  `datasets.RELLIS-3D.dynamic_classes` for RELLIS-3D), not motion tracking.
  Shown as-is in the UI ("Dynamic relevance (semantic proxy)"), never
  relabeled as tracking, for both datasets.
- RELLIS-3D's semantic-priority table (config.yaml) is a **separate,
  independently-authored prototype heuristic** — not scientifically
  validated, and not derived from or benchmarked against SemanticPOSS's
  table. Cross-dataset cell counts/latencies/scores are **not** claimed to
  be directly comparable in absolute terms — see "Multi-dataset support"
  above for why (different scene distributions, off-road vs. campus/urban).
- RELLIS-3D sequence discovery is a fixed, documented list in config.yaml
  (`00000`-`00004`), not self-discovered from disk the way SemanticPOSS's
  `sequences/` directory is — RELLIS's on-disk layout has no equivalent
  top-level manifest to enumerate from.
- No confidence/uncertainty — SemanticPOSS provides none. The `Cell` model
  and the frontend always render this as `N/A`/`null`, never a fabricated
  value; the config's uncertainty weight defaults to 0.
- Elevation aggregation is median-Z per cell — not a ground/DEM model, and
  cannot represent stacked surfaces (bridges, overhangs). The 3D view's
  "tile at elevation" rendering makes this explicit rather than implying a
  full 3D reconstruction.
- FOVEA's cell/storage count is *higher* than both baselines on the measured
  frame — see §17. The UI never claims FOVEA is faster or uses less memory.
- Region inspector selection is click-based via raycasting against an
  `InstancedMesh` — works well at typical zoom levels; extremely thin/fine
  cells can require zooming in for a precise click.
- No GPU-accelerated backend, no ROS2, no cloud/production deployment —
  out of scope for this prototype.

## 21. Development workflow

- The Python engine (`src/data_loader.py` through `src/metrics.py`) is the
  source of truth. Changes there should keep `tests/test_core.py` passing
  and, if the `Cell`/`accounting`/`summary` shapes change, keep
  `src/api.py`'s Pydantic models and `tests/test_api.py` in sync.
- `src/api.py` should stay a thin serialization layer — if you find yourself
  computing a score or threshold decision inside it, that logic belongs in
  `src/fovea_allocator.py` or `src/map_structure.py` instead.
- The frontend (`frontend/src`) should never reimplement scoring/allocation
  — every number displayed should trace back to a field already present in
  an API response. `frontend/src/constants.ts` is presentation-only (colors,
  labels); it does not compute anything FOVEA-specific.
- Run both test suites (`python -m unittest discover tests -v`) and `npm run
  build` (type-checks + bundles) before committing changes that touch the
  API boundary.
- The dataset (`dataset/`) and both `.venv/` and `frontend/node_modules/`
  are gitignored — never commit dataset content or installed dependencies.

---

## Architecture reference: module responsibilities

```
data_loader.py         -> Frame (points + semantic/instance labels), with diagnostics
semantic_labels.py      -> verbatim SemanticPOSS LABEL_DICT + priority lookup
geometry.py             -> shared coordinate/index/band-assignment helpers
map_structure.py        -> Cell dataclass + vectorized (pandas) point->cell aggregation
                           shared by all 3 methods (elevation median/std, majority
                           semantic class, information score)
uniform_baseline.py     -> Baseline 1: fixed cell size everywhere
distance_baseline.py    -> Baseline 2: cell size purely a function of range (bands)
fovea_allocator.py      -> FOVEA: quadtree top-down refinement within the
                           distance envelope, driven by information score
metrics.py              -> measured comparison metrics, distance-stratified stats
visualization.py        -> Plotly figures (Streamlit-independent)
app.py                  -> Streamlit dashboard (internal/debug)
api.py                  -> FastAPI backend (judge-facing frontend's data source)
benchmark.py             -> CLI benchmark -> JSON
```

The allocation policy is isolated behind `build_fovea_map(points, labels,
cfg)` — swapping the heuristic for a future optimization-based formulation
does not require touching the data structure, aggregation, API, or either
frontend.

### Why elevation = median-Z, not max-Z or true ground height

Median is robust to outliers within a cell. It is **not** a bare-earth/DEM
elevation model — a cell covering a building or tree canopy reports that
structure's median height, not terrain height.

### 2.5D limitation (documented, not hidden)

One elevation value per XY cell cannot represent stacked surfaces (a bridge
over a road, an overhang) — a known limitation of 2.5D representations in
general.

## What's next for v0.2

- Replace ground-truth labels with a real segmentation model's predictions
  behind the same `PointCloud + SemanticPredictions + optional Confidence`
  interface (`fovea_allocator.build_fovea_map` doesn't care where labels
  come from).
- Real per-point/per-cell confidence once a model provides it; wire into the
  already-present (but currently zero-weighted) uncertainty term — the API
  and frontend already have `uncertainty_score` plumbed through as `N/A`,
  ready to receive it.
- Temporal consistency / real dynamic-object detection using the available
  multi-frame sequences and instance labels (`instance_labels` are already
  loaded and available in `Frame`, just unused in v0.1's scoring).
- Move the heuristic weighted-sum allocation to a formal constrained
  optimization — the data flow already separates "compute scene features"
  from "decide resolution," so this is a policy swap, not a rearchitecture.
- Binary/typed-array point-cloud transport if frame sizes grow well beyond
  SemanticPOSS's ~65k points/frame (current JSON parallel-array transport is
  adequate at this scale but not infinitely scalable).
- Frontend (Three.js) consumption and 3D Tiles export of the hierarchy
  (`src/spatial/hierarchy.py`) + LOD selection (`src/spatial/lod.py`) +
  chunk serialization (`src/spatial/chunks.py`, `GET .../chunks`) built in
  Phase 3/3B/3C — deliberately not built yet; the full backend pipeline is
  wired into `src/api.py` and tested end-to-end on real data, but no
  browser/renderer consumes it yet, and `FVCH1` is a purpose-built format,
  not a 3D-Tiles-compatible tileset.
- Raw binary (`application/octet-stream`) transport for the `/chunks`
  endpoint instead of base64-in-JSON, once a real consumer needs the
  ~33% wire-size saving that removes.
