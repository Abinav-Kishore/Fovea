import { Crosshair, MousePointerClick } from "lucide-react";
import type { ChunkInfo } from "../api/fvch1";
import { RESOLUTION_COLORS } from "../constants";
import { ScoreBar } from "./ScoreBar";

interface StreamedChunkPanelProps {
  chunk: ChunkInfo | null;
}

/** Sidebar counterpart to RegionInspector for overlayMode === "streaming".
 * Deliberately does NOT call the backend's /cell/{index}/explain endpoint --
 * a streamed chunk's node_id doesn't correspond to an index into the
 * /map/fovea cells array (a chunk can be a single leaf Cell OR a
 * point-count-weighted aggregate over several, see src/spatial/chunks.py),
 * so there is no cell index to explain. Every field shown here is already
 * IN the decoded chunk -- nothing additional is fetched. */
export function StreamedChunkPanel({ chunk }: StreamedChunkPanelProps) {
  return (
    <div className="fv-panel p-4 flex flex-col gap-3">
      <div className="flex items-center gap-2 text-[var(--fv-cyan)]">
        <Crosshair size={14} />
        <h2 className="font-display text-sm font-semibold tracking-wide">STREAMED CHUNK</h2>
      </div>

      {!chunk ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-2 py-8 text-[var(--fv-text-faint)] text-xs">
          <MousePointerClick size={22} />
          <span>Click a tile in the streaming view to inspect it.</span>
        </div>
      ) : (
        <div className="fv-fade-in flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs text-[var(--fv-text-dim)]">{chunk.nodeId}</span>
            <span
              className="font-mono text-[10px] px-2 py-0.5 rounded-full border"
              style={{
                color: RESOLUTION_COLORS[chunk.level],
                borderColor: RESOLUTION_COLORS[chunk.level] + "55",
                background: RESOLUTION_COLORS[chunk.level] + "18",
              }}
            >
              {chunk.resolutionLabel.toUpperCase()} · {chunk.cellSizeM.toFixed(4)}m
            </span>
          </div>

          <div className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
            <Field label="Semantic class" value={chunk.semanticClass} />
            <Field label="Distance" value={`${chunk.rangeM.toFixed(1)} m`} />
            <Field label="Terrain" value={chunk.terrainClass} />
            <Field label="Obstacle" value={chunk.obstacleClass} />
            <Field label="Point count" value={chunk.pointCount.toLocaleString()} />
            <Field label="Elevation (median)" value={`${chunk.elevationMedian.toFixed(2)} m`} />
          </div>

          <div className="h-px bg-white/8 my-1" />

          <div className="flex flex-col gap-2">
            <ScoreBar label="Semantic priority" value={chunk.semanticPriority} color="var(--fv-cyan)" />
            <ScoreBar label="Geometry score" value={chunk.geometryScore} color="var(--fv-blue)" />
            <ScoreBar label="Dynamic relevance" sublabel="(semantic proxy)" value={chunk.dynamicScore} color="var(--fv-amber)" />
            <div className="h-px bg-white/6 my-0.5" />
            <ScoreBar label="Information score" value={chunk.informationScore} color="var(--fv-green)" />
          </div>

          <div className="h-px bg-white/8 my-1" />

          <div className="text-[10px] leading-relaxed text-[var(--fv-text-dim)] font-mono">
            {chunk.isLeaf ? (
              <>Exact single-cell measurement (this LOD node is a leaf of the FOVEA hierarchy).</>
            ) : (
              <>
                Aggregate over multiple finer cells the LOD layer folded into one tile at this
                distance -- values above are a point-count-weighted average, majority class fraction{" "}
                {(chunk.majorityClassFraction * 100).toFixed(0)}%. See src/spatial/chunks.py.
              </>
            )}
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
