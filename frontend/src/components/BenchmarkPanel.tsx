import { useMemo } from "react";
import { Scale } from "lucide-react";
import type { BenchmarkResponse, MapSummary } from "../api/types";
import { METHOD_LABELS } from "../constants";

interface BenchmarkPanelProps {
  uniform: MapSummary | null;
  distanceAdaptive: MapSummary | null;
  fovea: MapSummary | null;
  benchmark: BenchmarkResponse | null;
  // Stage 4 (uniform/distance-adaptive) loads after the FOVEA map -- while
  // it's in flight, show an honest "Loading…" rather than implying these
  // baselines were never measured.
  secondaryLoading?: boolean;
}

type Row = { key: string; label: string; summary: MapSummary | null; loading?: boolean };

function fmtMs(s: number | undefined | null, loading?: boolean): string {
  if (loading) return "Loading…";
  if (s === undefined || s === null || Number.isNaN(s)) return "Not measured";
  return `${(s * 1000).toFixed(0)} ms`;
}
function fmtKb(kb: number | undefined | null, loading?: boolean): string {
  if (loading) return "Loading…";
  if (kb === undefined || kb === null) return "Not measured";
  return kb > 1024 ? `${(kb / 1024).toFixed(2)} MB` : `${kb.toFixed(0)} KB`;
}

export function BenchmarkPanel({ uniform, distanceAdaptive, fovea, benchmark, secondaryLoading }: BenchmarkPanelProps) {
  const rows: Row[] = [
    { key: "uniform", label: METHOD_LABELS.uniform, summary: uniform, loading: secondaryLoading && !uniform },
    {
      key: "distance_adaptive",
      label: METHOD_LABELS.distance_adaptive,
      summary: distanceAdaptive,
      loading: secondaryLoading && !distanceAdaptive,
    },
    { key: "fovea", label: METHOD_LABELS.fovea, summary: fovea },
  ];

  const multiFrame = useMemo(() => {
    const data = benchmark?.data;
    if (!data || data.per_frame_results.length === 0) return null;
    const methods: Array<"uniform" | "distance_adaptive" | "fovea"> = ["uniform", "distance_adaptive", "fovea"];
    const n = data.per_frame_results.length;
    const perMethod: Record<string, { meanLatencyS: number; meanCells: number }> = {};
    for (const m of methods) {
      let latencySum = 0;
      let cellSum = 0;
      for (const frame of data.per_frame_results) {
        latencySum += frame.methods[m]?.map_generation_latency_s ?? 0;
        cellSum += frame.methods[m]?.output_cell_count ?? 0;
      }
      perMethod[m] = { meanLatencyS: latencySum / n, meanCells: cellSum / n };
    }
    return { n, sequence: data.sequence, timestamp: data.timestamp, perMethod };
  }, [benchmark]);

  const foveaCells = fovea?.output_cell_count;
  const daCells = distanceAdaptive?.output_cell_count;
  const uCells = uniform?.output_cell_count;

  return (
    <div style={{ gridArea: "benchmark" }} className="fv-panel m-3 mt-0 p-4 flex gap-6 overflow-hidden">
      <div className="flex flex-col gap-2 min-w-[720px]">
        <div className="flex items-center gap-2 text-[var(--fv-cyan)]">
          <Scale size={14} />
          <h2 className="font-display text-sm font-semibold tracking-wide">
            BENCHMARK &middot; UNIFORM vs DISTANCE-ADAPTIVE vs FOVEA (this frame, measured live)
          </h2>
        </div>

        <table className="text-xs w-full border-collapse">
          <thead>
            <tr className="text-[var(--fv-text-faint)] text-[10px] uppercase tracking-wide">
              <th className="text-left font-medium py-1">Method</th>
              <th className="text-right font-medium py-1">Cells</th>
              <th className="text-right font-medium py-1">Latency</th>
              <th className="text-right font-medium py-1">Storage (est.)</th>
              <th className="text-right font-medium py-1">Assigned</th>
              <th className="text-right font-medium py-1">Cell sizes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className="border-t border-white/6">
                <td className="py-1.5 text-[var(--fv-text)] font-medium">{r.label}</td>
                <td className="py-1.5 text-right font-mono">
                  {r.summary ? r.summary.output_cell_count.toLocaleString() : r.loading ? "Loading…" : "Not measured"}
                </td>
                <td className="py-1.5 text-right font-mono">{fmtMs(r.summary?.map_generation_latency_s, r.loading)}</td>
                <td className="py-1.5 text-right font-mono">{fmtKb(r.summary?.estimated_map_storage_kb, r.loading)}</td>
                <td className="py-1.5 text-right font-mono">
                  {r.summary ? `${(r.summary.assignment_rate * 100).toFixed(1)}%` : r.loading ? "Loading…" : "Not measured"}
                </td>
                <td className="py-1.5 text-right font-mono">
                  {r.summary ? r.summary.distinct_cell_sizes_m.length : r.loading ? "Loading…" : "Not measured"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {foveaCells !== undefined && daCells !== undefined && uCells !== undefined && (
          <p className="text-[10px] text-[var(--fv-text-faint)] font-mono mt-1">
            Measured on this frame: FOVEA {foveaCells.toLocaleString()} cells vs distance-adaptive{" "}
            {daCells.toLocaleString()} vs uniform {uCells.toLocaleString()}. FOVEA refines within the same
            distance envelope as distance-adaptive, so it never produces fewer cells -- this is representation
            concentration, not a memory-reduction claim.
          </p>
        )}
      </div>

      <div className="w-px bg-white/8" />

      <div className="flex flex-col gap-2 flex-1 min-w-0">
        <div className="text-[10px] uppercase tracking-wider text-[var(--fv-text-faint)]">
          Multi-frame benchmark
          {benchmark?.available && benchmark.source && ` (${benchmark.source})`}
        </div>
        {!benchmark?.available || !multiFrame ? (
          <p className="text-xs text-[var(--fv-text-faint)] font-mono">
            {benchmark?.message ?? "Not measured -- run `python -m src.benchmark` to generate this file."}
          </p>
        ) : (
          <>
            <p className="text-[11px] text-[var(--fv-text-dim)] font-mono">
              seq {multiFrame.sequence} &middot; {multiFrame.n} frames &middot;{" "}
              {new Date(multiFrame.timestamp).toLocaleString()}
            </p>
            <div className="flex flex-col gap-1.5">
              {(["uniform", "distance_adaptive", "fovea"] as const).map((m) => (
                <div key={m} className="flex items-center justify-between text-xs">
                  <span className="text-[var(--fv-text-dim)]">{METHOD_LABELS[m]}</span>
                  <span className="font-mono text-[var(--fv-text)]">
                    {fmtMs(multiFrame.perMethod[m].meanLatencyS)} avg &middot;{" "}
                    {Math.round(multiFrame.perMethod[m].meanCells).toLocaleString()} cells avg
                  </span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
