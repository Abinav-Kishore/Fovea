"""Visual theme for the FOVEA dashboard: CSS injection, an animated concept
diagram (pure SVG/CSS, no JS needed), and HTML builders for metric/legend
cards. Kept separate from app.py so the backend stays presentation-agnostic.
"""
from __future__ import annotations

METHOD_ACCENTS = {
    "uniform": "#94a3b8",
    "distance_adaptive": "#fbbf24",
    "fovea": "#22d3ee",
}

LEVEL_COLORS = {0: "#ef4444", 1: "#fbbf24", 2: "#22c55e"}
LEVEL_LABELS = {0: "Coarse", 1: "Medium", 2: "Fine"}


def get_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --fv-bg-0: #070a12;
  --fv-bg-1: #0d1220;
  --fv-bg-2: #131a2b;
  --fv-card: rgba(255,255,255,0.045);
  --fv-card-border: rgba(255,255,255,0.09);
  --fv-cyan: #22d3ee;
  --fv-violet: #a78bfa;
  --fv-pink: #f472b6;
  --fv-amber: #fbbf24;
  --fv-text: #e6e9f2;
  --fv-text-dim: #8892a8;
}

html, body, [class*="stApp"] {
  font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
  background: radial-gradient(ellipse 80% 60% at 20% -10%, rgba(34,211,238,0.10), transparent),
              radial-gradient(ellipse 70% 50% at 100% 0%, rgba(167,139,250,0.10), transparent),
              linear-gradient(180deg, var(--fv-bg-0) 0%, var(--fv-bg-1) 40%, var(--fv-bg-2) 100%);
  color: var(--fv-text);
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(13,18,32,0.98), rgba(7,10,18,0.98));
  border-right: 1px solid var(--fv-card-border);
}

/* replay a gentle fade/rise on the main content every rerun (tab switch, widget change) */
[data-testid="stAppViewContainer"] .main .block-container {
  animation: fv-fade-up 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes fv-fade-up {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}

.fv-hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 3.2rem;
  line-height: 1.05;
  margin: 0;
  background: linear-gradient(100deg, var(--fv-cyan), var(--fv-violet) 45%, var(--fv-pink) 90%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  animation: fv-gradient-shift 6s ease-in-out infinite alternate;
  letter-spacing: -0.02em;
}
@keyframes fv-gradient-shift {
  from { background-position: 0% 50%; }
  to   { background-position: 100% 50%; }
}

.fv-hero-sub {
  font-size: 1.02rem;
  color: var(--fv-text-dim);
  max-width: 46em;
  margin-top: 0.4rem;
  line-height: 1.55;
}

.fv-badge-row { display:flex; gap:0.5rem; margin-top:0.9rem; flex-wrap:wrap; }
.fv-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  border: 1px solid var(--fv-card-border);
  background: var(--fv-card);
  color: var(--fv-text-dim);
  animation: fv-fade-up 0.6s cubic-bezier(0.16,1,0.3,1) backwards;
}

/* metric / method cards */
.fv-card-row { display:flex; gap:1rem; margin: 0.6rem 0 1.2rem 0; flex-wrap: wrap; }
.fv-card {
  flex: 1 1 260px;
  background: var(--fv-card);
  border: 1px solid var(--fv-card-border);
  border-radius: 18px;
  padding: 1.15rem 1.3rem;
  backdrop-filter: blur(14px);
  position: relative;
  overflow: hidden;
  transition: transform 0.25s cubic-bezier(0.16,1,0.3,1), border-color 0.25s ease, box-shadow 0.25s ease;
  animation: fv-fade-up 0.55s cubic-bezier(0.16,1,0.3,1) backwards;
}
.fv-card:hover {
  transform: translateY(-4px);
  border-color: var(--fv-accent, var(--fv-cyan));
  box-shadow: 0 10px 30px -8px color-mix(in srgb, var(--fv-accent, var(--fv-cyan)) 45%, transparent);
}
.fv-card.hero-method {
  border-color: rgba(34,211,238,0.35);
  animation: fv-fade-up 0.55s cubic-bezier(0.16,1,0.3,1) backwards, fv-pulse-glow 3.2s ease-in-out infinite;
}
@keyframes fv-pulse-glow {
  0%, 100% { box-shadow: 0 0 0px rgba(34,211,238,0); }
  50%      { box-shadow: 0 0 26px -4px rgba(34,211,238,0.45); }
}
.fv-card-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fv-accent, var(--fv-text-dim));
  display:flex; align-items:center; gap:0.4rem;
}
.fv-card-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.35rem;
  font-weight: 600;
  margin: 0.15rem 0 0.7rem 0;
}
.fv-stat-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 0.9rem; }
.fv-stat-label { font-size: 0.68rem; color: var(--fv-text-dim); text-transform:uppercase; letter-spacing:0.04em; }
.fv-stat-value { font-family:'JetBrains Mono', monospace; font-size: 1.02rem; font-weight:500; color: var(--fv-text); }
.fv-level-bar { display:flex; height:9px; border-radius:6px; overflow:hidden; margin-top:0.85rem; background:rgba(255,255,255,0.06); }
.fv-level-bar > div { transition: width 0.6s cubic-bezier(0.16,1,0.3,1); }
.fv-level-legend { display:flex; gap:0.8rem; margin-top:0.45rem; font-size:0.68rem; color: var(--fv-text-dim); }
.fv-level-legend span.dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:0.3rem; vertical-align:middle; }

.fv-section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 600;
  font-size: 1.3rem;
  margin: 1.6rem 0 0.3rem 0;
  display:flex; align-items:center; gap:0.5rem;
}
.fv-section-sub { color: var(--fv-text-dim); font-size:0.88rem; margin-bottom:0.8rem; }

.fv-explain-box {
  background: linear-gradient(120deg, rgba(34,211,238,0.10), rgba(167,139,250,0.08));
  border: 1px solid rgba(34,211,238,0.25);
  border-left: 3px solid var(--fv-cyan);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  font-size: 0.92rem;
  line-height: 1.6;
  animation: fv-fade-up 0.5s cubic-bezier(0.16,1,0.3,1);
}

/* tabs */
button[data-baseweb="tab"] {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
}
div[data-baseweb="tab-highlight"] {
  background: linear-gradient(90deg, var(--fv-cyan), var(--fv-violet)) !important;
}

/* plot container lift-in */
[data-testid="stPlotlyChart"] {
  animation: fv-fade-up 0.6s cubic-bezier(0.16,1,0.3,1) backwards;
  border-radius: 16px;
  overflow: hidden;
}

/* honest-note callout */
.fv-note {
  border: 1px dashed var(--fv-card-border);
  border-radius: 12px;
  padding: 0.8rem 1rem;
  font-size: 0.82rem;
  color: var(--fv-text-dim);
  margin-top: 0.6rem;
}
</style>
"""


def hero_diagram_svg() -> str:
    """Illustrative (not data-driven) concept diagram: distance shrinks cell
    size outward (near=fine, far=coarse), but one far, high-priority region
    is pulsing/highlighted to show FOVEA refining it despite its distance."""
    return """
<div style="display:flex; justify-content:center; margin: 0.4rem 0 0.2rem 0;">
<svg width="100%" height="230" viewBox="0 0 760 230" style="max-width:760px;">
  <style>
    .fv-ring { stroke: rgba(255,255,255,0.10); fill: none; }
    .fv-cell { fill: rgba(34,211,238,0.16); stroke: rgba(34,211,238,0.4); stroke-width:1; }
    .fv-cell-far { fill: rgba(251,191,36,0.14); stroke: rgba(251,191,36,0.35); }
    .fv-hot { fill: rgba(244,63,94,0.55); stroke: #f43f5e; stroke-width: 1.4;
              animation: fv-pulse 1.8s ease-in-out infinite; transform-origin: center; }
    @keyframes fv-pulse {
      0%, 100% { opacity: 0.55; transform: scale(1); }
      50%      { opacity: 1; transform: scale(1.18); }
    }
    .fv-sensor { fill: #22d3ee; animation: fv-blink 2.4s ease-in-out infinite; }
    @keyframes fv-blink { 0%,100%{opacity:1;} 50%{opacity:0.5;} }
    .fv-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; fill: #8892a8; }
    .fv-label-hot { font-family: 'JetBrains Mono', monospace; font-size: 11px; fill: #f43f5e; font-weight:600;}
  </style>

  <circle class="fv-sensor" cx="70" cy="115" r="6"/>
  <text class="fv-label" x="40" y="140">sensor</text>

  <!-- fine near-field grid -->
  <g>
    <rect class="fv-cell" x="120" y="95" width="14" height="14"/>
    <rect class="fv-cell" x="136" y="95" width="14" height="14"/>
    <rect class="fv-cell" x="120" y="111" width="14" height="14"/>
    <rect class="fv-cell" x="136" y="111" width="14" height="14"/>
    <rect class="fv-cell" x="120" y="127" width="14" height="14"/>
    <rect class="fv-cell" x="136" y="127" width="14" height="14"/>
    <rect class="fv-cell" x="120" y="79" width="14" height="14"/>
    <rect class="fv-cell" x="136" y="79" width="14" height="14"/>
  </g>
  <text class="fv-label" x="112" y="165">near &#183; fine (~0.05m)</text>

  <!-- medium mid-field grid -->
  <g>
    <rect class="fv-cell-far" x="300" y="70" width="30" height="30"/>
    <rect class="fv-cell-far" x="332" y="70" width="30" height="30"/>
    <rect class="fv-cell-far" x="300" y="102" width="30" height="30"/>
    <rect class="fv-cell-far" x="332" y="102" width="30" height="30"/>
    <rect class="fv-cell-far" x="300" y="134" width="30" height="30"/>
    <rect class="fv-cell-far" x="332" y="134" width="30" height="30"/>
  </g>
  <text class="fv-label" x="296" y="185">mid &#183; medium (~0.10-0.25m)</text>

  <!-- coarse far-field grid, with one hot high-priority cell refined -->
  <g>
    <rect class="fv-cell-far" x="520" y="50" width="58" height="58" opacity="0.7"/>
    <rect class="fv-cell-far" x="580" y="50" width="58" height="58" opacity="0.7"/>
    <rect class="fv-cell-far" x="520" y="110" width="58" height="58" opacity="0.7"/>
    <!-- the refined "important object" cell, subdivided finer despite being far -->
    <g transform="translate(580,110)">
      <rect class="fv-hot" x="0" y="0" width="27" height="27"/>
      <rect class="fv-hot" x="29" y="0" width="27" height="27"/>
      <rect class="fv-hot" x="0" y="29" width="27" height="27"/>
      <rect class="fv-hot" x="29" y="29" width="27" height="27"/>
    </g>
  </g>
  <text class="fv-label" x="510" y="185">far &#183; coarse (~0.5m)</text>
  <text class="fv-label-hot" x="590" y="46">&#9679; pedestrian &#8594; refined anyway</text>

  <line x1="20" y1="115" x2="700" y2="115" stroke="rgba(255,255,255,0.06)" stroke-dasharray="4 6"/>
</svg>
</div>
"""


def method_card_html(
    name: str,
    icon: str,
    accent: str,
    cell_count: int,
    latency_s: float,
    assignment_rate: float,
    storage_kb: float,
    level_counts: dict | None = None,
    is_hero: bool = False,
    delay: float = 0.0,
) -> str:
    level_bar = ""
    level_legend = ""
    if level_counts:
        total = sum(level_counts.values()) or 1
        segs = []
        legend = []
        for lvl in sorted(level_counts.keys()):
            pct = 100.0 * level_counts[lvl] / total
            color = LEVEL_COLORS.get(lvl, "#888")
            segs.append(f'<div style="width:{pct:.2f}%; background:{color};"></div>')
            legend.append(
                f'<span><span class="dot" style="background:{color};"></span>'
                f'{LEVEL_LABELS.get(lvl, lvl)} {level_counts[lvl]:,}</span>'
            )
        level_bar = f'<div class="fv-level-bar">{"".join(segs)}</div>'
        level_legend = f'<div class="fv-level-legend">{"".join(legend)}</div>'

    hero_class = "hero-method" if is_hero else ""
    return f"""
<div class="fv-card {hero_class}" style="--fv-accent:{accent}; animation-delay:{delay}s;">
  <div class="fv-card-eyebrow">{icon} {'HERO METHOD' if is_hero else 'BASELINE'}</div>
  <div class="fv-card-title">{name}</div>
  <div class="fv-stat-grid">
    <div><div class="fv-stat-label">Cells</div><div class="fv-stat-value">{cell_count:,}</div></div>
    <div><div class="fv-stat-label">Latency</div><div class="fv-stat-value">{latency_s*1000:.0f} ms</div></div>
    <div><div class="fv-stat-label">Assigned</div><div class="fv-stat-value">{assignment_rate*100:.1f}%</div></div>
    <div><div class="fv-stat-label">Storage</div><div class="fv-stat-value">{storage_kb:.0f} KB</div></div>
  </div>
  {level_bar}
  {level_legend}
</div>
"""


def section_header(icon: str, title: str, subtitle: str = "") -> str:
    sub = f'<div class="fv-section-sub">{subtitle}</div>' if subtitle else ""
    return f'<div class="fv-section-title">{icon} {title}</div>{sub}'
