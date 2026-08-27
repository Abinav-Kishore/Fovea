import { useMemo } from "react";
import { BarChart3 } from "lucide-react";
import type { Cell, MapSummary } from "../api/types";
import { RESOLUTION_COLORS, RESOLUTION_LABELS } from "../constants";
import { ScoreBar } from "./ScoreBar";

interface InformationAnalyticsProps {
  cells: Cell[];
  summary: MapSummary | null;
  selectedResolutionLevel?: number | null;
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function InformationAnalytics({ cells, summary, selectedResolutionLevel = null }: InformationAnalyticsProps) {
  const avg = useMemo(
    () => ({
      distance: mean(cells.map((c) => c.range_m)),
      semantic: mean(cells.map((c) => c.semantic_priority)),
      geometry: mean(cells.map((c) => c.geometry_score)),
      dynamic: mean(cells.map((c) => c.dynamic_score)),
      information: mean(cells.map((c) => c.information_score)),
    }),
    [cells]
  );

  const levelCounts = summary?.cells_by_resolution_level ?? {};
  const totalCells = Object.values(levelCounts).reduce((a, b) => a + b, 0);

  return (
    <div className="fv-panel p-4 flex flex-col gap-4">
      <div className="flex items-center gap-2 text-[var(--fv-cyan)]">
        <BarChart3 size={14} />
        <h2 className="font-display text-sm font-semibold tracking-wide">INFORMATION ANALYTICS</h2>
      </div>

      <div>
        <div className="text-[10px] uppercase tracking-wider text-[var(--fv-text-faint)] mb-2">
          Frame-average signals (all FOVEA cells)
        </div>
        <div className="flex flex-col gap-2">
          <ScoreBar
            label="Distance"
            value={avg.distance !== null ? Math.min(1, avg.distance / 100) : null}
            suffix={avg.distance !== null ? "" : ""}
            color="var(--fv-blue)"
          />
          <ScoreBar label="Semantic priority" value={avg.semantic} color="var(--fv-cyan)" />
          <ScoreBar label="Geometry score" value={avg.geometry} color="var(--fv-blue)" />
          <ScoreBar label="Dynamic relevance" sublabel="(semantic proxy)" value={avg.dynamic} color="var(--fv-amber)" />
          <ScoreBar label="Uncertainty" value={null} color="var(--fv-text-faint)" />
          <ScoreBar label="Information score" value={avg.information} color="var(--fv-green)" />
        </div>
        {avg.distance !== null && (
          <div className="text-[10px] text-[var(--fv-text-faint)] mt-1.5 font-mono">
            mean range: {avg.distance.toFixed(1)} m (bar normalized to 100m for display)
          </div>
        )}
      </div>

      <div className="h-px bg-white/8" />

      <div>
        <div className="text-[10px] uppercase tracking-wider text-[var(--fv-text-faint)] mb-2">
          Resolution distribution (FOVEA)
        </div>
        <div className="flex flex-col gap-2">
          {[2, 1, 0].map((lvl) => {
            const count = levelCounts[String(lvl)] ?? 0;
            const pct = totalCells > 0 ? (count / totalCells) * 100 : 0;
            const isSelected = selectedResolutionLevel === lvl;
            return (
              <div
                key={lvl}
                className={`flex items-center gap-2.5 rounded-md -mx-1 px-1 py-0.5 transition-colors ${
                  isSelected ? "bg-[var(--fv-cyan)]/10 ring-1 ring-[var(--fv-cyan)]/40" : ""
                }`}
              >
                <span className="text-[11px] text-[var(--fv-text-dim)] w-16 shrink-0">
                  {RESOLUTION_LABELS[lvl].toUpperCase()}
                </span>
                <div className="flex-1 h-3 rounded bg-white/6 overflow-hidden">
                  <div
                    className="h-full transition-all duration-500"
                    style={{ width: `${pct}%`, background: RESOLUTION_COLORS[lvl] }}
                  />
                </div>
                <span className="font-mono text-[11px] w-16 text-right text-[var(--fv-text)]">
                  {count.toLocaleString()}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
