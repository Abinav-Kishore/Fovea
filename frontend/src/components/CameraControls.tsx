import { RotateCcw } from "lucide-react";
import type { CameraPreset } from "../scene/Scene";

interface CameraControlsProps {
  preset: CameraPreset;
  onPreset: (preset: CameraPreset) => void;
  onReset: () => void;
}

const PRESETS: { key: CameraPreset; label: string }[] = [
  { key: "perspective", label: "Perspective" },
  { key: "top", label: "Top" },
  { key: "front", label: "Front" },
];

export function CameraControls({ preset, onPreset, onReset }: CameraControlsProps) {
  return (
    <div className="flex items-center gap-1 fv-panel px-1.5 py-1.5 pointer-events-auto">
      {PRESETS.map((p) => (
        <button
          key={p.key}
          onClick={() => onPreset(p.key)}
          className={`px-2.5 py-1.5 rounded-lg text-[11px] font-medium transition-colors ${
            preset === p.key
              ? "bg-[var(--fv-cyan)]/15 text-[var(--fv-cyan)] border border-[var(--fv-cyan)]/40"
              : "text-[var(--fv-text-dim)] border border-transparent hover:text-[var(--fv-text)] hover:bg-white/5"
          }`}
        >
          {p.label}
        </button>
      ))}
      <button
        onClick={onReset}
        title="Reset View"
        className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[11px] font-medium text-[var(--fv-text-dim)] hover:text-[var(--fv-text)] hover:bg-white/5 border-l border-white/8 ml-1 pl-2.5"
      >
        <RotateCcw size={12} />
        Reset
      </button>
    </div>
  );
}
