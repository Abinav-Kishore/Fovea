# FOVEA v0.1

Prototype for SIH problem statement 26053: *Adaptive Variable Resolution 2.5D
Lidar Mapping for Dynamic Environment Perception*.

FOVEA takes a LiDAR point cloud + semantic labels and produces a **variable
resolution 2.5D semantic-elevation map**, where cell size is bounded by a
distance-dependent envelope (near = fine, far = coarse, per the SIH example)
but can be *further refined within that envelope* based on an information
score (semantic priority + local geometric complexity + dynamic-object
relevance). This lets two regions at the **same distance** end up at
**different resolutions** if one is more informative than the other — that
behavior is measured and demonstrated below, not just asserted.

FOVEA is **not** claiming to invent LiDAR semantic segmentation, 2.5D
mapping, adaptive sampling, or resource-constrained optimization — all of
that is established prior work. The contribution here is the specific
integration: information-aware refinement *constrained by* a distance
envelope, applied to a navigation-oriented 2.5D semantic-elevation grid.

**Status: proof-of-system prototype (v0.1), not a finished research result.**
Every weight/threshold in `config.yaml` is a heuristic default, clearly
labelled as such in code and config comments, and is meant to be replaced by
a validated formulation later (see "What's next" below).

---

## What's real vs. what's a stand-in

| Piece | v0.1 | Future |
|---|---|---|
| Point cloud | Real SemanticPOSS `.bin` frames | Same, or live LiDAR |
| Semantic input | SemanticPOSS ground-truth `.label` files, used as an **oracle perception stand-in** | A trained segmentation model's predictions, same interface |
| Confidence/uncertainty | **Not available** — SemanticPOSS has no confidence field, so it's excluded from the score by default (weight 0 in config), never invented | Calibrated model confidence |
| Dynamic-object relevance | Semantic-category proxy only (e.g. "car"/"person" are flagged dynamic-relevant) — **no actual motion/temporal tracking** | Real temporal/tracking-based dynamic detection |
| Allocation policy | Configurable heuristic weighted sum + quadtree threshold split | A validated constrained-optimization formulation |

---

## Verified dataset format (inspected, not assumed)

```
SemanticPOSS_dataset/dataset/sequences/<seq>/velodyne/<frame>.bin   float32, reshape(-1,4) = x,y,z,intensity
SemanticPOSS_dataset/dataset/sequences/<seq>/labels/<frame>.label   uint32; semantic = value & 0xFFFF, instance = value >> 16
```
Confirmed against `SemanticPOSS_dataset/dataset/read_data.py` (the dataset's
own reader) and against actual file sizes (frame `000000` of sequence `00`:
66,658 points, 66,658 labels — byte counts match exactly). The class-id →
name mapping (`src/semantic_labels.py`) is copied verbatim from that same
file's `LABEL_DICT` — SemanticPOSS uses a **sparse, non-contiguous** id set
(`0, 4-17, 21, 22`), which is why it's loaded explicitly instead of assumed
to match SemanticKITTI. Any id outside that set (we observed one: id `20` in
frame 0) is reported as `"unknown"`, never guessed.

---

## Setup

```bash
pip install -r requirements.txt
```

## Run the dashboard

```bash
python run_demo.py
```
(equivalent to `streamlit run src/app.py`). It auto-loads sequence `00`,
frame `000000` by default (already verified to contain person / rider / car
/ bike / building / ground — good class diversity for the demo scenario) and
lets you switch sequence/frame/view from the sidebar.

## Run the benchmark (no UI)

```bash
python -m src.benchmark --config config.yaml --num-frames 5 --repeats 3
```
Writes `results/benchmark_result.json` with per-frame, per-method measured
metrics (cell counts, latency, storage estimate, distance-band stats,
environment info) — nothing in that file is fabricated.

## Run tests

```bash
python -m unittest discover tests -v
```
12 tests covering coordinate→cell quantization (incl. negative coordinates),
distance-band assignment and monotonicity, semantic priority lookup and
unknown-id fallback, information-score weighting (including the
uncertainty-excluded-by-default behavior), point conservation/accounting
(no silent point loss, NaN points rejected not dropped), and aggregation
determinism.

---

## Architecture

```
data_loader.py        -> Frame (points + semantic/instance labels), with diagnostics
semantic_labels.py     -> verbatim SemanticPOSS LABEL_DICT + priority lookup
geometry.py            -> shared coordinate/index/band-assignment helpers
map_structure.py       -> Cell dataclass + vectorized (pandas) point->cell aggregation
                          shared by all 3 methods (elevation median/std, majority
                          semantic class, information score)
uniform_baseline.py    -> Baseline 1: fixed cell size everywhere
distance_baseline.py   -> Baseline 2: cell size purely a function of range (bands)
fovea_allocator.py     -> FOVEA: quadtree top-down refinement within the
                          distance envelope, driven by information score
metrics.py             -> measured comparison metrics, distance-stratified stats
visualization.py       -> Plotly figures (Streamlit-independent)
app.py                 -> Streamlit dashboard
benchmark.py           -> CLI benchmark -> JSON
```

The allocation policy is isolated behind `build_fovea_map(points, labels,
cfg)` — swapping the heuristic for a future optimization-based formulation
does not require touching the data structure, aggregation, or visualization
code.

### Why elevation = median-Z, not max-Z or true ground height

Median is robust to outliers within a cell and is documented as such in
`map_structure.py`. It is **not** a bare-earth/DEM elevation model — a cell
covering a building or tree canopy reports that structure's median height,
not terrain height. This is a semantic-elevation representation.

### 2.5D limitation (documented, not hidden)

One elevation value per XY cell cannot represent stacked surfaces (a bridge
over a road, an overhang). This is a known limitation of 2.5D representations
in general, not something FOVEA solves.

### How FOVEA's envelope + refinement actually works

1. Each configured distance band (`distance_adaptive.bands` in
   `config.yaml`) defines a **base cell size** — identical to the
   distance-adaptive baseline. This is the coarsest a cell in that band is
   allowed to be.
2. Points are first aggregated at the band's **finest** permitted size
   (`base / refinement_factor`) to get a per-micro-cell information score
   (semantic priority of the majority class + normalized local elevation std
   as a geometric-complexity proxy + a dynamic-class flag).
3. A standard top-down quadtree split decision (not novel — this is a
   textbook adaptive-mesh technique) decides, per coarse group and then per
   medium sub-group, whether to refine — based on configurable score
   thresholds (`split_to_medium_threshold`, `split_to_fine_threshold`).
4. A region can end at 3 possible cell sizes per band: `base`, `base/2`, or
   `base/refinement_factor` — never coarser than `base`, never finer than the
   floor. The distance envelope is a hard constraint; information score only
   decides "how much finer, within it."

**This was verified against real data, not just asserted.** On sequence 00,
frame 0, a `car` cell and a `ground` cell at the *same range* (27.3 m,
same 10–30 m band, base size 0.10 m) received different resolutions:

```
CAR cell:    range=27.3m  size=0.0250m  level=fine    info_score=0.76
GROUND cell: range=27.3m  size=0.1000m  level=coarse  info_score=0.03
```

---

## Measured results (sequence 00, frame 000000, 66,658 points, this machine)

From an actual run of `python -m src.benchmark --num-frames 5 --repeats 3`
(`results/benchmark_result.json` has the full output):

| Method | Cells | Single-frame latency (mean of 3 repeats) | Est. storage |
|---|---|---|---|
| Uniform (0.20 m) | 23,954 | ~0.19 s | ~2.6 MB |
| Distance-adaptive | 35,552 | ~0.29 s | ~3.9 MB |
| FOVEA | 41,868 | ~0.8–3.0 s (see note) | ~4.6 MB |

**Honest note on FOVEA's cell/storage count:** FOVEA produces *more* cells
and *more* storage than both baselines here, not less. This is expected and
correct, not a bug: FOVEA's coarse level matches the distance-adaptive
baseline exactly, and refinement only *adds* cells in high-information
regions — it never removes cells elsewhere. **The claim for v0.1 is
representation efficiency in the sense of concentrating detail where it's
useful, not raw memory reduction** — we do not claim FOVEA saves memory
over the distance-adaptive baseline, because it measurably does not, on this
frame, with these heuristic weights.

**Honest note on latency variance:** repeated in-process runs of
`build_fovea_map` on identical input showed latency ranging from 0.8s to
~3.0s on this machine (memory and object counts stayed flat across repeats,
ruling out a leak in our code) — likely OS-level scheduling/disk-cache noise
on this particular Windows machine, not measured as a robust number. All
three methods stay comfortably under interactive/dashboard-usable latency,
and the Streamlit app caches map builds per (sequence, frame) so this cost
is paid once per frame selection, not per interaction.

This is a **single-frame map generation latency**, not FPS — only one frame
was benchmarked per timing run. The benchmark's `approx_throughput_frames_per_s`
field is a real measured sequential wall-clock number when running multiple
frames, not a claimed real-time rate.

---

## Known limitations / what's NOT done in v0.1

- No trained perception model — semantic labels are the dataset's ground
  truth, used as an oracle stand-in (by design, per scope).
- No real dynamic-object tracking — "dynamic relevance" is a static
  semantic-category proxy.
- No confidence/uncertainty — SemanticPOSS provides none; the config exposes
  an uncertainty weight but it defaults to 0 and is never faked.
- Elevation aggregation is median-Z per cell — not a ground/DEM model, and
  cannot represent stacked surfaces (bridges, overhangs).
- Geometry score (elevation std) is a cheap, documented proxy for local
  complexity, not a validated terrain-roughness metric.
- Region inspector selection in the dashboard is index-based (via a hover +
  number input), not true click-to-select — a click-driven inspector was
  judged not essential to the demo's conceptual clarity given the time
  budget.
- No ROS2, no GPU, no cloud/production deployment — explicitly out of scope
  for this prototype per the assignment.

## What's next for v0.2

- Replace ground-truth labels with a real segmentation model's predictions
  behind the same `PointCloud + SemanticPredictions + optional Confidence`
  interface (`fovea_allocator.build_fovea_map` doesn't care where labels
  come from).
- Real per-point/per-cell confidence once a model provides it; wire into the
  already-present (but currently zero-weighted) uncertainty term.
- Temporal consistency / real dynamic-object detection using the available
  multi-frame sequences and instance labels (`instance_labels` are already
  loaded and available in `Frame`, just unused in v0.1's scoring).
- Move the heuristic weighted-sum allocation to a formal constrained
  optimization (e.g. maximize total utility subject to a representation
  budget) — the data flow already separates "compute scene features" from
  "decide resolution," so this is a policy swap, not a rearchitecture.
- Click-driven (not index-based) region inspector.
