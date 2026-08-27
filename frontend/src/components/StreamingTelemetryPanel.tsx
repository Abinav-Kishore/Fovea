import { Gauge } from "lucide-react";
import type { StreamingTelemetry } from "../scene/ChunkField";

const STAGE_LABELS: Record<string, string> = {
  idle: "Idle",
  "loading-coarse": "Loading coarse…",
  coarse: "Coarse",
  "loading-fine": "Refining…",
  fine: "Fine",
  error: "Error",
};

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

/** Small, existing-style telemetry strip for overlayMode === "streaming" --
 * LOD stage / selected node count / selected point count / FVCH1 payload
 * size / live FPS. Every number is read straight from
 * useSpatialStreaming's state (src/hooks/useSpatialStreaming.ts) and
 * ChunkField's own FPS counter -- nothing here is estimated. */
export function StreamingTelemetryPanel({ telemetry }: { telemetry: StreamingTelemetry | null }) {
  if (!telemetry) return null;
  const stageLabel = STAGE_LABELS[telemetry.stage] ?? telemetry.stage;
  const stageColor = telemetry.stage === "error" ? "var(--fv-red)" : telemetry.stage === "fine" ? "var(--fv-green)" : "var(--fv-amber)";

  return (
    <div className="fv-panel px-3.5 py-2.5 flex flex-col gap-1.5 pointer-events-none w-56">
      <div className="flex items-center gap-1.5 text-[9px] uppercase tracking-wider text-[var(--fv-text-faint)]">
        <Gauge size={11} />
        Streaming telemetry
      </div>
      <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[11px] font-mono">
        <span className="text-[var(--fv-text-faint)]">LOD</span>
        <span style={{ color: stageColor }}>{stageLabel}</span>
        <span className="text-[var(--fv-text-faint)]">Nodes</span>
        <span className="text-[var(--fv-text)]">{telemetry.nodeCount.toLocaleString()}</span>
        <span className="text-[var(--fv-text-faint)]">Points</span>
        <span className="text-[var(--fv-text)]">{telemetry.pointCount.toLocaleString()}</span>
        <span className="text-[var(--fv-text-faint)]">Payload</span>
        <span className="text-[var(--fv-text)]">{formatBytes(telemetry.payloadBytes)}</span>
        <span className="text-[var(--fv-text-faint)]">FPS</span>
        <span className="text-[var(--fv-text)]">{telemetry.fps !== null ? telemetry.fps.toFixed(0) : "—"}</span>
      </div>
      {(telemetry.fetchMs !== null || telemetry.decodeMs !== null) && (
        <div className="text-[9px] text-[var(--fv-text-faint)] font-mono">
          fetch {telemetry.fetchMs?.toFixed(0) ?? "—"}ms · decode {telemetry.decodeMs?.toFixed(1) ?? "—"}ms
        </div>
      )}
    </div>
  );
}
