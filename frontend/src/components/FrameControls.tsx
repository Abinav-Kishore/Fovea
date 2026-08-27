import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";

interface FrameControlsProps {
  sequence: string;
  sequences: string[];
  onSequenceChange: (seq: string) => void;
  frameIds: string[];
  frameIndex: number;
  onFrameIndexChange: (index: number) => void;
  playing: boolean;
  onTogglePlaying: () => void;
}

export function FrameControls({
  sequence,
  sequences,
  onSequenceChange,
  frameIds,
  frameIndex,
  onFrameIndexChange,
  playing,
  onTogglePlaying,
}: FrameControlsProps) {
  const max = Math.max(0, frameIds.length - 1);

  return (
    <div className="fv-panel px-4 py-2.5 flex items-center gap-4 pointer-events-auto">
      <select
        value={sequence}
        onChange={(e) => onSequenceChange(e.target.value)}
        className="bg-transparent text-xs font-mono text-[var(--fv-text)] border border-white/10 rounded-md px-2 py-1 outline-none focus:border-[var(--fv-cyan)]/50"
      >
        {sequences.map((s) => (
          <option key={s} value={s} className="bg-[var(--fv-bg-1)]">
            seq {s}
          </option>
        ))}
      </select>

      <button
        onClick={() => onFrameIndexChange(Math.max(0, frameIndex - 1))}
        disabled={frameIndex <= 0}
        className="text-[var(--fv-text-dim)] hover:text-[var(--fv-cyan)] disabled:opacity-30 disabled:hover:text-[var(--fv-text-dim)]"
      >
        <ChevronLeft size={16} />
      </button>

      <button
        onClick={onTogglePlaying}
        className="text-[var(--fv-cyan)] hover:opacity-80"
        title={playing ? "Pause" : "Play"}
      >
        {playing ? <Pause size={16} /> : <Play size={16} />}
      </button>

      <button
        onClick={() => onFrameIndexChange(Math.min(max, frameIndex + 1))}
        disabled={frameIndex >= max}
        className="text-[var(--fv-text-dim)] hover:text-[var(--fv-cyan)] disabled:opacity-30 disabled:hover:text-[var(--fv-text-dim)]"
      >
        <ChevronRight size={16} />
      </button>

      <input
        type="range"
        min={0}
        max={max}
        value={frameIndex}
        onChange={(e) => onFrameIndexChange(Number(e.target.value))}
        className="w-40 accent-[var(--fv-cyan)]"
      />
      <span className="font-mono text-xs text-[var(--fv-text-dim)] w-28 text-right">
        {frameIds[frameIndex] ?? "—"} / {frameIds.length}
      </span>
    </div>
  );
}
