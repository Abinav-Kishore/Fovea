"""Plotly figure builders for the Streamlit dashboard. Kept independent of
Streamlit itself (only depends on plotly + numpy) so a future React/WebGL
frontend could reuse these figures or the underlying data directly."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.map_structure import Cell
from src.semantic_labels import LABEL_DICT
from src.theme import LEVEL_COLORS, LEVEL_LABELS

# Brighter, saturated palette tuned for a dark background (the original
# dataset SEM_COLOR values in read_data.py are muted and hard to read here).
_SEMANTIC_COLORS = {
    "unlabeled": "#64748b", "1 person": "#ff5c7a", "2+ person": "#ff5c7a",
    "rider": "#f472b6", "car": "#38bdf8", "trunk": "#b08968", "plants": "#4ade80",
    "traffic sign 1": "#f87171", "traffic sign 2": "#f87171", "traffic sign 3": "#f87171",
    "pole": "#fde047", "trashcan": "#a3e635", "building": "#fbbf24",
    "cone/stone": "#67e8f9", "fence": "#fb923c", "bike": "#22d3ee",
    "ground": "#475569", "unknown": "#e2e8f0",
}

_FONT = dict(family="Inter, sans-serif", color="#c7ccdb", size=12)
_PAPER = "rgba(0,0,0,0)"
_PLOT_BG = "rgba(255,255,255,0.02)"
_GRID = "rgba(255,255,255,0.06)"


def _color_for(name: str) -> str:
    return _SEMANTIC_COLORS.get(name, "#e2e8f0")


def _apply_theme(fig: go.Figure, title: str, height: int = 600) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk, sans-serif", size=17, color="#e6e9f2")),
        xaxis_title="X (m)", yaxis_title="Y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1, gridcolor=_GRID, zerolinecolor=_GRID),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
        height=height,
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT_BG,
        font=_FONT,
        legend=dict(itemsizing="constant", bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(t=48, l=10, r=10, b=10),
        transition=dict(duration=350, easing="cubic-in-out"),
        hoverlabel=dict(bgcolor="#131a2b", font=dict(family="JetBrains Mono, monospace", size=11)),
    )
    return fig


def raw_semantic_scatter(points: np.ndarray, semantic_labels: np.ndarray, max_points: int = 40000) -> go.Figure:
    n = points.shape[0]
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        points, semantic_labels = points[idx], semantic_labels[idx]

    fig = go.Figure()
    from src.semantic_labels import class_name
    names = np.array([class_name(int(l)) for l in semantic_labels])
    for name in sorted(set(names)):
        mask = names == name
        fig.add_trace(go.Scattergl(
            x=points[mask, 0], y=points[mask, 1], mode="markers",
            marker=dict(size=2.4, color=_color_for(name), opacity=0.85),
            name=name,
        ))
    return _apply_theme(fig, "Raw semantic point cloud (top-down)")


def cells_to_rectangles_figure(cells: list[Cell], title: str, color_by: str = "semantic") -> go.Figure:
    """Draw each cell as an actual rectangle sized to its own cell_size_m --
    this is what makes the variable-resolution structure visually honest
    (not a uniform grid colored differently)."""
    fig = go.Figure()
    if not cells:
        return _apply_theme(fig, f"{title} (no cells)")

    if color_by == "resolution":
        groups = {}
        for c in cells:
            groups.setdefault(c.resolution_level, []).append(c)
        for lvl, group in sorted(groups.items()):
            color = LEVEL_COLORS.get(lvl, "#888")
            label = LEVEL_LABELS.get(lvl, str(lvl))
            _add_rect_trace(fig, group, color, f"{label} (~{group[0].cell_size_m:.3f}m)")
    else:
        groups = {}
        for c in cells:
            groups.setdefault(c.majority_class_name, []).append(c)
        for name, group in sorted(groups.items()):
            _add_rect_trace(fig, group, _color_for(name), name)

    return _apply_theme(fig, title)


def _add_rect_trace(fig: go.Figure, cells: list[Cell], color: str, name: str, max_cells: int = 15000):
    """Render cells as filled rectangles via one Scatter trace per group using
    'fill toself' polygons batched with None separators (fast in plotly)."""
    if len(cells) > max_cells:
        rng = np.random.default_rng(0)
        cells = [cells[i] for i in rng.choice(len(cells), size=max_cells, replace=False)]
    xs, ys = [], []
    for c in cells:
        h = c.cell_size_m / 2
        cx, cy = c.center_x, c.center_y
        xs += [cx - h, cx + h, cx + h, cx - h, cx - h, None]
        ys += [cy - h, cy - h, cy + h, cy + h, cy - h, None]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself",
        line=dict(width=0.5, color=color), fillcolor=color,
        name=name, opacity=0.88,
        hoverinfo="skip",
    ))


def resolution_level_overlay(cells: list[Cell]) -> go.Figure:
    return cells_to_rectangles_figure(cells, "Resolution-level overlay (fine / medium / coarse)", color_by="resolution")


def cell_scatter_for_inspector(cells: list[Cell]) -> go.Figure:
    """Scatter of cell centers, colored by resolution level, sized by cell
    size, for click-to-inspect. Uses markers (not rectangles) so Plotly click
    events can identify a specific cell reliably."""
    fig = go.Figure()
    xs = [c.center_x for c in cells]
    ys = [c.center_y for c in cells]
    sizes = [max(3, min(20, c.cell_size_m * 40)) for c in cells]
    colors = [c.resolution_level for c in cells]
    customdata = list(range(len(cells)))
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(
            size=sizes, color=colors,
            colorscale=[[0, LEVEL_COLORS[0]], [0.5, LEVEL_COLORS[1]], [1, LEVEL_COLORS[2]]],
            showscale=True, colorbar=dict(title="level", tickvals=[0, 1, 2],
                                           ticktext=["Coarse", "Medium", "Fine"]),
            line=dict(width=0.4, color="rgba(255,255,255,0.3)"),
        ),
        customdata=customdata,
        hovertext=[f"idx={i}  class={c.majority_class_name}  range={c.range_m:.1f}m  size={c.cell_size_m:.3f}m"
                   for i, c in enumerate(cells)],
        hoverinfo="text",
    ))
    return _apply_theme(fig, "Cell inspector (hover for details, then enter the index below)")
