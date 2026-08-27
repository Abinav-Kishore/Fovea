/**
 * Presentation-only constants (colors, labels). No FOVEA logic lives here --
 * every value colored/labeled by these maps comes from the backend
 * (majority_class_name, resolution_level, information_score, etc.).
 */

// Mirrors the palette used by the existing Streamlit dashboard
// (src/visualization.py _SEMANTIC_COLORS) so both frontends read consistently.
export const SEMANTIC_COLORS: Record<string, string> = {
  unlabeled: "#64748b",
  "1 person": "#ff5c7a",
  "2+ person": "#ff5c7a",
  rider: "#f472b6",
  car: "#38bdf8",
  trunk: "#b08968",
  plants: "#4ade80",
  "traffic sign 1": "#f87171",
  "traffic sign 2": "#f87171",
  "traffic sign 3": "#f87171",
  pole: "#fde047",
  trashcan: "#a3e635",
  building: "#fbbf24",
  "cone/stone": "#67e8f9",
  fence: "#fb923c",
  bike: "#22d3ee",
  ground: "#475569",
  unknown: "#e2e8f0",
};

export function semanticColor(name: string): string {
  return SEMANTIC_COLORS[name] ?? "#e2e8f0";
}

// Resolution level 0/1/2 -> Coarse/Medium/Fine, matching src/theme.py LEVEL_LABELS.
export const RESOLUTION_COLORS: Record<number, string> = {
  0: "#f97355", // coarse -> muted orange/red
  1: "#fbbf24", // medium -> amber
  2: "#34d399", // fine -> green
};
export const RESOLUTION_LABELS: Record<number, string> = {
  0: "Coarse",
  1: "Medium",
  2: "Fine",
};
export const RESOLUTION_DESCRIPTIONS: Record<number, string> = {
  0: "Low spatial detail",
  1: "Intermediate detail",
  2: "High spatial detail",
};

type RampStop = [number, [number, number, number]];

function rampColor(t: number, stops: RampStop[]): string {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (clamped >= t0 && clamped <= t1) {
      const f = (clamped - t0) / (t1 - t0 || 1);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
      return `rgb(${r},${g},${b})`;
    }
  }
  const last = stops[stops.length - 1][1];
  return `rgb(${last[0]},${last[1]},${last[2]})`;
}

// Information-score ramp: dark (background-matching, low info) -> violet -> hot
// magenta/amber -> bright white (high info) -- a "thermal hotspot" ramp so the
// handful of high-information regions pop hard against a mostly-dark scene.
const INFORMATION_STOPS: RampStop[] = [
  [0.0, [10, 14, 26]],
  [0.3, [59, 40, 110]],
  [0.55, [147, 51, 160]],
  [0.8, [244, 63, 94]],
  [1.0, [253, 224, 71]],
];
export function informationColor(t: number): string {
  return rampColor(t, INFORMATION_STOPS);
}

// Elevation ramp: low (near-ground, cool blue) -> mid (teal/green) -> high
// (warm tan/white), evoking a terrain colormap rather than a heat map.
const ELEVATION_STOPS: RampStop[] = [
  [0.0, [20, 40, 70]],
  [0.35, [30, 120, 130]],
  [0.65, [110, 170, 90]],
  [1.0, [235, 220, 180]],
];
export function elevationColor(t: number): string {
  return rampColor(t, ELEVATION_STOPS);
}

function rampCssGradient(stops: RampStop[]): string {
  const parts = stops.map(([t, [r, g, b]]) => `rgb(${r},${g},${b}) ${(t * 100).toFixed(0)}%`);
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

// CSS gradients built from the exact same stops the 3D scene uses, so a
// legend bar always matches what's actually rendered -- one source of truth.
export const INFORMATION_GRADIENT_CSS = rampCssGradient(INFORMATION_STOPS);
export const ELEVATION_GRADIENT_CSS = rampCssGradient(ELEVATION_STOPS);

export const VIEW_MODES = ["semantic", "resolution", "elevation", "information"] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

export const VIEW_MODE_LABELS: Record<ViewMode, string> = {
  semantic: "Semantic",
  resolution: "Resolution",
  elevation: "Elevation",
  information: "Information",
};

export const METHOD_LABELS: Record<string, string> = {
  uniform: "Uniform",
  distance_adaptive: "Distance-Adaptive",
  fovea: "FOVEA",
};
