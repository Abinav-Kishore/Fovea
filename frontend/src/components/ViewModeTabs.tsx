import { Contrast, Layers, Mountain, Radar, Sparkles, Waves } from "lucide-react";
import {
  ELEVATION_GRADIENT_CSS,
  INFORMATION_GRADIENT_CSS,
  INTENSITY_GRADIENT_CSS,
  RESOLUTION_COLORS,
  RESOLUTION_DESCRIPTIONS,
  RESOLUTION_LABELS,
  VIEW_MODES,
  VIEW_MODE_LABELS,
  type ViewMode,
} from "../constants";
import type { OverlayMode } from "../scene/Scene";

interface ViewModeTabsProps {
  mode: ViewMode;
  onChange: (mode: ViewMode) => void;
}

const ICONS: Record<ViewMode, React.ComponentType<{ size?: number }>> = {
  semantic: Layers,
  resolution: Waves,
  elevation: Mountain,
  information: Sparkles,
  intensity: Contrast,
};

export function ViewModeTabs({ mode, onChange }: ViewModeTabsProps) {
  return (
    <div className="flex items-center gap-1 fv-panel px-1.5 py-1.5 pointer-events-auto">
      {VIEW_MODES.map((m) => {
        const Icon = ICONS[m];
        const active = m === mode;
        return (
          <button
            key={m}
            onClick={() => onChange(m)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              active
                ? "bg-[var(--fv-cyan)]/15 text-[var(--fv-cyan)] border border-[var(--fv-cyan)]/40"
                : "text-[var(--fv-text-dim)] border border-transparent hover:text-[var(--fv-text)] hover:bg-white/5"
            }`}
          >
            <Icon size={13} />
            {VIEW_MODE_LABELS[m]}
          </button>
        );
      })}
    </div>
  );
}

const OVERLAY_OPTIONS: { key: OverlayMode; label: string }[] = [
  { key: "raw", label: "Raw" },
  { key: "overlay", label: "Raw + FOVEA" },
  { key: "fovea", label: "FOVEA only" },
  // Phase 3D: renders the LOD-selected chunk frontier (hierarchy -> LOD ->
  // /chunks -> FVCH1 decode) instead of the full FOVEA cell list -- see
  // src/scene/ChunkField.tsx. The three modes above are unchanged.
  { key: "streaming", label: "Streaming (LOD)" },
];

/** RAW / RAW+FOVEA / FOVEA-only -- lets the judge see the raw point cloud and
 * the FOVEA representation share the same physical scene. */
export function OverlayModeControl({ mode, onChange }: { mode: OverlayMode; onChange: (m: OverlayMode) => void }) {
  return (
    <div className="flex items-center gap-1 fv-panel px-1.5 py-1.5 pointer-events-auto">
      <Radar size={13} className="text-[var(--fv-text-faint)] ml-1 mr-0.5" />
      {OVERLAY_OPTIONS.map((opt) => {
        const active = opt.key === mode;
        return (
          <button
            key={opt.key}
            onClick={() => onChange(opt.key)}
            className={`px-2.5 py-1.5 rounded-lg text-[11px] font-medium transition-colors ${
              active
                ? "bg-[var(--fv-cyan)]/15 text-[var(--fv-cyan)] border border-[var(--fv-cyan)]/40"
                : "text-[var(--fv-text-dim)] border border-transparent hover:text-[var(--fv-text)] hover:bg-white/5"
            }`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

export function ResolutionLegend() {
  return (
    <div className="fv-panel px-3.5 py-2.5 flex flex-col gap-1.5 pointer-events-none">
      <div className="text-[9px] uppercase tracking-wider text-[var(--fv-text-faint)]">FOVEA resolution</div>
      {[2, 1, 0].map((lvl) => (
        <div key={lvl} className="flex items-center gap-2 text-xs">
          <span
            className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
            style={{ background: RESOLUTION_COLORS[lvl] }}
          />
          <span className="text-[var(--fv-text)] font-medium w-14">{RESOLUTION_LABELS[lvl]}</span>
          <span className="text-[var(--fv-text-faint)] text-[10px]">{RESOLUTION_DESCRIPTIONS[lvl]}</span>
        </div>
      ))}
    </div>
  );
}

/** Only lists semantic classes actually present in the current frame's FOVEA
 * cells -- not the full static SemanticPOSS label set. */
export function SemanticLegend({ classes }: { classes: { name: string; color: string }[] }) {
  if (classes.length === 0) return null;
  return (
    <div className="fv-panel px-3.5 py-2.5 flex flex-col gap-1.5 pointer-events-none max-w-[220px]">
      <div className="text-[9px] uppercase tracking-wider text-[var(--fv-text-faint)]">
        Classes present in this frame
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1">
        {classes.map((c) => (
          <div key={c.name} className="flex items-center gap-1.5 text-[11px]">
            <span className="inline-block w-2 h-2 rounded-full shrink-0" style={{ background: c.color }} />
            <span className="text-[var(--fv-text-dim)]">{c.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Shared low->high gradient legend for the Information and Elevation views --
 * the gradient CSS is built from the exact same ramp the 3D scene uses. */
export function GradientLegend({
  title,
  gradientCss,
  lowLabel,
  highLabel,
}: {
  title: string;
  gradientCss: string;
  lowLabel: string;
  highLabel: string;
}) {
  return (
    <div className="fv-panel px-3.5 py-2.5 flex flex-col gap-1.5 pointer-events-none w-64">
      <div className="text-[9px] uppercase tracking-wider text-[var(--fv-text-faint)]">{title}</div>
      <div className="h-2 rounded-full" style={{ background: gradientCss }} />
      <div className="flex justify-between text-[10px] text-[var(--fv-text-dim)]">
        <span>{lowLabel}</span>
        <span>{highLabel}</span>
      </div>
    </div>
  );
}

export function InformationLegend() {
  return (
    <GradientLegend
      title="Information score"
      gradientCss={INFORMATION_GRADIENT_CSS}
      lowLabel="LOW INFORMATION"
      highLabel="HIGH INFORMATION"
    />
  );
}

export function ElevationLegend() {
  return (
    <GradientLegend
      // Phase 4AB: this legend now covers both the FOVEA cell field (median
      // z per cell) and the raw point cloud (each point's own real z) --
      // wording kept generic on purpose rather than claiming a per-cell
      // statistic that doesn't apply to a raw point.
      title="Elevation (height, z)"
      gradientCss={ELEVATION_GRADIENT_CSS}
      lowLabel="LOW"
      highLabel="HIGH"
    />
  );
}

export function IntensityLegend() {
  return (
    <GradientLegend
      title="Intensity (raw LiDAR return, points only)"
      gradientCss={INTENSITY_GRADIENT_CSS}
      lowLabel="LOW"
      highLabel="HIGH"
    />
  );
}

const VIEW_MODE_STORY: Record<ViewMode, string> = {
  semantic: "Cell color reflects each region's majority semantic class.",
  resolution: "High-information regions receive finer spatial representation.",
  elevation: "Cell height/color reflects the region's measured elevation.",
  information: "Cell color reflects the FOVEA information score that drove the resolution decision.",
  intensity: "Point color reflects the raw LiDAR return intensity -- FOVEA cells have no per-point intensity, so this view is most meaningful in Raw/Raw+FOVEA mode.",
};

/** Small explanatory strip near the hero visualization -- text only, never
 * claims anything beyond what the algorithm actually implements. */
export function ContextualLabel({ mode }: { mode: ViewMode }) {
  return (
    <div className="pointer-events-none">
      <div className="text-[11px] font-medium text-[var(--fv-text)]">Information-aware adaptive resolution</div>
      <div className="text-[10px] text-[var(--fv-text-faint)]">{VIEW_MODE_STORY[mode]}</div>
    </div>
  );
}
