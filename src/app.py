"""FOVEA v0.1 Streamlit dashboard.

Run with:  streamlit run src/app.py
Backend algorithms (loader, baselines, allocator, metrics) are plain
Python/NumPy/pandas modules with no Streamlit dependency -- this file only
wires them to widgets (and to src/theme.py for presentation), so a future
React/WebGL frontend can call the same backend functions directly.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data_loader import Frame, list_frame_ids, load_frame, frame_diagnostics
from src.distance_baseline import build_distance_adaptive_map
from src.fovea_allocator import build_fovea_map, explain_allocation
from src.map_structure import Cell
from src.metrics import summarize_method
from src.theme import METHOD_ACCENTS, get_css, hero_diagram_svg, method_card_html, section_header
from src.uniform_baseline import build_uniform_map
from src.visualization import cell_scatter_for_inspector, cells_to_rectangles_figure, raw_semantic_scatter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fovea.app")

st.set_page_config(page_title="FOVEA v0.1", layout="wide", page_icon="🛰️")
st.markdown(get_css(), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_config():
    return load_config("config.yaml")


@st.cache_data(show_spinner="Loading frame...")
def _load_frame_cached(data_root: str, sequence: str, frame_id: str):
    return load_frame(data_root, sequence, frame_id)


@st.cache_data(show_spinner="Building maps (uniform / distance-adaptive / FOVEA)...")
def _build_all_maps(_points_hash, points, labels, cfg):
    u_cells, u_acc = build_uniform_map(points, labels, cfg)
    d_cells, d_acc = build_distance_adaptive_map(points, labels, cfg)
    f_cells, f_acc = build_fovea_map(points, labels, cfg)
    return (u_cells, u_acc), (d_cells, d_acc), (f_cells, f_acc)


def find_contrast_example(cells: list[Cell], range_tolerance_m: float = 3.0):
    """Find a real (not staged) pair of FOVEA cells at ~the same range where
    one is high-priority and got refined finer, and one is low-priority and
    stayed coarse -- the core behavior distinguishing FOVEA from pure
    distance-adaptive mapping. Returns (hot_cell, cold_cell) or None."""
    hot_candidates = [c for c in cells if c.dynamic_score >= 1.0 or c.semantic_priority >= 0.6]
    cold_candidates = [c for c in cells if c.semantic_priority <= 0.15]
    best = None
    best_gap = None
    for h in sorted(hot_candidates, key=lambda c: -c.information_score)[:60]:
        nearby_cold = [c for c in cold_candidates if abs(c.range_m - h.range_m) <= range_tolerance_m]
        for cold in nearby_cold:
            if cold.cell_size_m <= h.cell_size_m:
                continue  # want a genuine finer-vs-coarser contrast
            gap = cold.cell_size_m / h.cell_size_m
            if best_gap is None or gap > best_gap:
                best, best_gap = (h, cold), gap
    return best


def main():
    st.markdown(
        f"""
        <div class="fv-hero-title">FOVEA</div>
        <div class="fv-hero-sub">
            Variable-resolution 2.5D semantic-elevation mapping for autonomous ground vehicles &mdash;
            SIH 26053 prototype. Distance sets a resolution <b>envelope</b> (near = fine, far = coarse);
            what a region actually <i>contains</i> decides how much detail it gets within that envelope.
        </div>
        <div class="fv-badge-row">
            <div class="fv-badge" style="animation-delay:0.05s">SIH 26053</div>
            <div class="fv-badge" style="animation-delay:0.10s">v0.1 &middot; proof-of-system</div>
            <div class="fv-badge" style="animation-delay:0.15s">SemanticPOSS ground-truth = oracle perception</div>
            <div class="fv-badge" style="animation-delay:0.20s">heuristic weights, not scientifically validated</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(hero_diagram_svg(), unsafe_allow_html=True)

    cfg = _load_config()

    with st.sidebar:
        st.markdown("### 🗂️ Dataset / Frame")
        sequence_opts = ["00", "01", "02", "03", "04", "05"]
        sequence = st.selectbox("Sequence", options=sequence_opts,
                                 index=sequence_opts.index(cfg["data"]["sequence"]))
        try:
            frame_ids = list_frame_ids(cfg["data"]["root"], sequence)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        default_frame = f"{cfg['data']['frame']:06d}"
        frame_id = st.selectbox("Frame", options=frame_ids,
                                 index=frame_ids.index(default_frame) if default_frame in frame_ids else 0)

        st.markdown("---")
        st.markdown("### ⚙️ FOVEA weights (config.yaml)")
        w = cfg["fovea"]["weights"]
        for label, key, color in [("Semantic", "semantic", "#22d3ee"), ("Geometry", "geometry", "#a78bfa"),
                                   ("Dynamic", "dynamic", "#f472b6"), ("Uncertainty", "uncertainty", "#fbbf24")]:
            val = w.get(key, 0.0)
            st.markdown(
                f'<div style="font-size:0.78rem; color:#8892a8; margin-bottom:2px;">{label} &middot; {val:.2f}</div>'
                f'<div style="background:rgba(255,255,255,0.06); border-radius:6px; height:7px; margin-bottom:8px;">'
                f'<div style="width:{val*100:.0f}%; background:{color}; height:100%; border-radius:6px;"></div></div>',
                unsafe_allow_html=True,
            )
        st.caption("Uncertainty is 0 by default -- SemanticPOSS has no confidence field.")

    frame = _load_frame_cached(cfg["data"]["root"], sequence, frame_id)
    diag = frame_diagnostics(frame)

    if not frame.has_labels:
        st.warning("⚠️ No label file found for this frame -- running in NO-SEMANTIC-INPUT (geometry-only) mode.")

    points_hash = hash((sequence, frame_id))
    (u_cells, u_acc), (d_cells, d_acc), (f_cells, f_acc) = _build_all_maps(
        points_hash, frame.points, frame.semantic_labels, cfg
    )

    bands = cfg["distance_adaptive"]["bands"]
    u_summary = summarize_method(u_cells, u_acc, bands)
    d_summary = summarize_method(d_cells, d_acc, bands)
    f_summary = summarize_method(f_cells, f_acc, bands)

    st.markdown(
        section_header("📊", "Measured metrics", "Same frame, same machine, same run -- every number below is measured, not asserted."),
        unsafe_allow_html=True,
    )
    cards_html = '<div class="fv-card-row">'
    cards_html += method_card_html(
        "Uniform", "▦", METHOD_ACCENTS["uniform"], u_summary["output_cell_count"],
        u_summary["map_generation_latency_s"], u_summary["assignment_rate"],
        u_summary["estimated_map_storage_kb"], delay=0.0,
    )
    cards_html += method_card_html(
        "Distance-Adaptive", "◎", METHOD_ACCENTS["distance_adaptive"], d_summary["output_cell_count"],
        d_summary["map_generation_latency_s"], d_summary["assignment_rate"],
        d_summary["estimated_map_storage_kb"], delay=0.1,
    )
    cards_html += method_card_html(
        "FOVEA", "✦", METHOD_ACCENTS["fovea"], f_summary["output_cell_count"],
        f_summary["map_generation_latency_s"], f_summary["assignment_rate"],
        f_summary["estimated_map_storage_kb"], level_counts=f_summary["cells_by_resolution_level"],
        is_hero=True, delay=0.2,
    )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    contrast = find_contrast_example(f_cells)
    if contrast:
        hot, cold = contrast
        st.markdown(
            f"""<div class="fv-explain-box">
            🎯 <b>Live example from this exact frame:</b> a <b>{hot.majority_class_name}</b> cell and a
            <b>{cold.majority_class_name}</b> cell sit at nearly the same range
            (<code>{hot.range_m:.1f}m</code> vs <code>{cold.range_m:.1f}m</code>, same distance band) &mdash;
            yet FOVEA gives the {hot.majority_class_name} cell a <b>{cold.cell_size_m/hot.cell_size_m:.0f}&times; finer</b>
            cell (<code>{hot.cell_size_m:.4f}m</code> vs <code>{cold.cell_size_m:.4f}m</code>) because its
            information score is higher (<code>{hot.information_score:.2f}</code> vs <code>{cold.information_score:.2f}</code>).
            Distance-adaptive alone would have given both the identical <code>{cold.cell_size_m:.4f}m</code> cell.
            </div>""",
            unsafe_allow_html=True,
        )

    tabs = st.tabs(["🔎 Raw Semantic", "▦ Uniform", "◎ Distance-Adaptive", "✦ FOVEA",
                     "🌈 Resolution Levels", "🔬 Region Inspector", "📋 Diagnostics"])

    with tabs[0]:
        st.plotly_chart(raw_semantic_scatter(frame.points, frame.semantic_labels,
                                              cfg["visualization"]["point_subsample_max"]),
                         use_container_width=True)
    with tabs[1]:
        st.plotly_chart(cells_to_rectangles_figure(u_cells, "Uniform 2.5D map", color_by="semantic"),
                         use_container_width=True)
        st.markdown('<div class="fv-note">Fixed cell size everywhere, regardless of distance or content.</div>',
                    unsafe_allow_html=True)
    with tabs[2]:
        st.plotly_chart(cells_to_rectangles_figure(d_cells, "Distance-adaptive 2.5D map", color_by="semantic"),
                         use_container_width=True)
        st.markdown('<div class="fv-note">Cell size depends only on range from the sensor -- two regions at the '
                    'same distance always get the same resolution here, regardless of what they contain.</div>',
                    unsafe_allow_html=True)
    with tabs[3]:
        st.plotly_chart(cells_to_rectangles_figure(f_cells, "FOVEA 2.5D map", color_by="semantic"),
                         use_container_width=True)
        st.markdown('<div class="fv-note">Same distance envelope as the baseline above, but high-information '
                    'regions (people, vehicles, riders, geometric complexity) are refined finer within it.</div>',
                    unsafe_allow_html=True)
    with tabs[4]:
        st.plotly_chart(cells_to_rectangles_figure(f_cells, "FOVEA resolution levels", color_by="resolution"),
                         use_container_width=True)
        st.markdown('<div class="fv-note">Red = coarse (band base size) &middot; amber = medium (base/2) &middot; '
                    'green = fine (base/refinement_factor). This is real backend cell geometry, not a color '
                    'overlay on a fixed grid.</div>', unsafe_allow_html=True)
    with tabs[5]:
        st.markdown(section_header("🔬", "Region inspector", "Every value below is read directly from the selected FOVEA cell's stored fields."),
                    unsafe_allow_html=True)
        st.plotly_chart(cell_scatter_for_inspector(f_cells), use_container_width=True)

        max_idx = len(f_cells) - 1
        idx = st.number_input("Cell index (hover the scatter above to find one)", min_value=0, max_value=max_idx, value=0, step=1)
        c = f_cells[idx]
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("**📍 Location & resolution**")
                st.json({
                    "range_m": round(c.range_m, 3), "center_x": round(c.center_x, 3),
                    "center_y": round(c.center_y, 3), "cell_size_m": c.cell_size_m,
                    "resolution_level": c.resolution_level, "band_id": c.band_id,
                    "point_count": c.point_count,
                })
        with c2:
            with st.container(border=True):
                st.markdown("**🧬 Semantic / geometry / allocation factors**")
                st.json({
                    "majority_class": c.majority_class_name,
                    "majority_class_fraction": round(c.majority_class_fraction, 3),
                    "elevation_median_m": round(c.elevation_median, 3),
                    "elevation_std_m": round(c.elevation_std, 3),
                    "semantic_priority": round(c.semantic_priority, 3),
                    "geometry_score": round(c.geometry_score, 3),
                    "dynamic_score": c.dynamic_score,
                    "information_score": round(c.information_score, 3),
                })
        st.markdown(f'<div class="fv-explain-box">💡 {explain_allocation(c)}</div>', unsafe_allow_html=True)
    with tabs[6]:
        st.markdown(section_header("📋", "Frame diagnostics", "Loaded directly from the .bin/.label files -- no assumptions."),
                    unsafe_allow_html=True)
        st.json(diag)
        with st.expander("Distance-stratified stats (FOVEA)"):
            st.json(f_summary["distance_band_stats"])


if __name__ == "__main__":
    main()
