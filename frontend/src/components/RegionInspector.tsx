import { useEffect, useState } from "react";
import { Crosshair, MousePointerClick } from "lucide-react";
import { api } from "../api/client";
import type { Cell, DatasetName } from "../api/types";
import { RESOLUTION_COLORS } from "../constants";
import { ScoreBar } from "./ScoreBar";

interface RegionInspectorProps {
  dataset: DatasetName;
  sequence: string;
  frameId: string | null;
  cell: Cell | null;
}

export function RegionInspector({ dataset, sequence, frameId, cell }: RegionInspectorProps) {
  const [explanation, setExplanation] = useState<string | null>(null);

  useEffect(() => {
    setExplanation(null);
    if (!frameId || !cell) return;
    const controller = new AbortController();
    api
      .explainCell(dataset, sequence, frameId, cell.index, "fovea", controller.signal)
      .then((r) => setExplanation(r.explanation))
      .catch(() => undefined);
    return () => controller.abort();
  }, [dataset, sequence, frameId, cell]);

  return (
    <div className="fv-panel p-4 flex flex-col gap-3">
      <div className="flex items-center gap-2 text-[var(--fv-cyan)]">
        <Crosshair size={14} />
        <h2 className="font-display text-sm font-semibold tracking-wide">FOVEA DECISION</h2>
      </div>

      {!cell ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-2 py-8 text-[var(--fv-text-faint)] text-xs">
          <MousePointerClick size={22} />
          <span>Click a cell in the resolution view to inspect it.</span>
        </div>
      ) : (
        <div className="fv-fade-in flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs text-[var(--fv-text-dim)]">Region #{cell.index}</span>
            <span
              className="font-mono text-[10px] px-2 py-0.5 rounded-full border"
              style={{
                color: RESOLUTION_COLORS[cell.resolution_level],
                borderColor: RESOLUTION_COLORS[cell.resolution_level] + "55",
                background: RESOLUTION_COLORS[cell.resolution_level] + "18",
              }}
            >
              {cell.resolution_label.toUpperCase()} · {cell.cell_size_m.toFixed(4)}m
            </span>
          </div>

          <div className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
            <Field label="Semantic class" value={cell.majority_class_name} />
            <Field label="Distance" value={`${cell.range_m.toFixed(1)} m`} />
            <Field label="Resolution level" value={cell.resolution_label} />
            <Field label="Cell size" value={`${cell.cell_size_m.toFixed(4)} m`} />
            <Field label="Point count" value={cell.point_count.toLocaleString()} />
            <Field label="Elevation (median)" value={`${cell.elevation_median.toFixed(2)} m`} />
          </div>

          <div className="h-px bg-white/8 my-1" />

          <div className="flex flex-col gap-2">
            <ScoreBar label="Semantic priority" value={cell.semantic_priority} color="var(--fv-cyan)" />
            <ScoreBar label="Geometry score" value={cell.geometry_score} color="var(--fv-blue)" />
            <ScoreBar
              label="Dynamic relevance"
              sublabel="(semantic proxy)"
              value={cell.dynamic_score}
              color="var(--fv-amber)"
            />
            <ScoreBar label="Uncertainty" value={cell.uncertainty_score} color="var(--fv-text-faint)" />
            <div className="h-px bg-white/6 my-0.5" />
            <ScoreBar label="Information score" value={cell.information_score} color="var(--fv-green)" />
          </div>

          <div className="h-px bg-white/8 my-1" />

          <div>
            <div className="text-[10px] uppercase tracking-wider text-[var(--fv-text-faint)] mb-1.5">
              Why this region was refined
            </div>
            <p className="text-xs leading-relaxed text-[var(--fv-text-dim)] font-mono">
              {explanation ?? "Loading explanation…"}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] text-[var(--fv-text-faint)] uppercase tracking-wide">{label}</div>
      <div className="text-[var(--fv-text)] font-mono">{value}</div>
    </div>
  );
}
