import { Radio } from "lucide-react";
import { DATASET_NAMES, type DatasetName } from "../api/types";

interface HeaderProps {
  dataset: DatasetName;
  onDatasetChange: (dataset: DatasetName) => void;
  availableDatasets: DatasetName[];
  sequence: string;
  frameId: string | null;
  backendConnected: boolean | null;
}

export function Header({
  dataset,
  onDatasetChange,
  availableDatasets,
  sequence,
  frameId,
  backendConnected,
}: HeaderProps) {
  return (
    <header
      style={{ gridArea: "header" }}
      className="flex items-center justify-between px-6 border-b border-white/8 bg-[var(--fv-bg-1)]"
    >
      <div className="flex items-baseline gap-3">
        <h1 className="font-display text-2xl font-bold tracking-tight text-white">FOVEA</h1>
        <span className="text-xs text-[var(--fv-text-dim)] hidden md:inline">
          Foveated Variable-Resolution 2.5D LiDAR Perception
        </span>
      </div>

      <div className="flex items-center gap-5 font-mono text-xs text-[var(--fv-text-dim)]">
        <div className="flex items-center gap-1.5">
          <span className="text-[var(--fv-text-faint)]">DATASET</span>
          <select
            value={dataset}
            onChange={(e) => onDatasetChange(e.target.value as DatasetName)}
            className="bg-transparent text-[var(--fv-text)] border border-white/10 rounded-md px-1.5 py-0.5 outline-none focus:border-[var(--fv-cyan)]/50 cursor-pointer"
          >
            {(availableDatasets.length ? availableDatasets : DATASET_NAMES).map((d) => (
              <option key={d} value={d} className="bg-[var(--fv-bg-1)]">
                {d}
              </option>
            ))}
          </select>
        </div>
        <div>
          <span className="text-[var(--fv-text-faint)]">SEQ </span>
          <span className="text-[var(--fv-text)]">{sequence}</span>
        </div>
        <div>
          <span className="text-[var(--fv-text-faint)]">FRAME </span>
          <span className="text-[var(--fv-text)]">{frameId ?? "—"}</span>
        </div>
        <div className="flex items-center gap-1.5 pl-2 border-l border-white/10">
          {backendConnected === null ? (
            <Radio size={13} className="text-[var(--fv-text-faint)]" />
          ) : (
            <span
              className={`fv-status-dot inline-block w-2 h-2 rounded-full ${
                backendConnected ? "bg-[var(--fv-cyan)]" : "bg-[var(--fv-red)]"
              }`}
            />
          )}
          <span className={backendConnected ? "text-[var(--fv-cyan)]" : "text-[var(--fv-red)]"}>
            {backendConnected === null ? "Connecting…" : backendConnected ? "Backend: Connected" : "Backend: Offline"}
          </span>
        </div>
      </div>
    </header>
  );
}
