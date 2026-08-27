interface ScoreBarProps {
  label: string;
  sublabel?: string; // small honest qualifier, e.g. "(semantic proxy)" for dynamic relevance
  value: number | null; // null renders as N/A, never fabricated
  color?: string;
  suffix?: string;
}

export function ScoreBar({ label, sublabel, value, color = "var(--fv-cyan)", suffix = "" }: ScoreBarProps) {
  const isNA = value === null || Number.isNaN(value);
  const pct = isNA ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-2.5">
      <span className="text-[11px] text-[var(--fv-text-dim)] w-28 shrink-0">
        {label}
        {sublabel && <span className="text-[var(--fv-text-faint)]"> {sublabel}</span>}
      </span>
      <div className="flex-1 h-1.5 rounded-full bg-white/6 overflow-hidden">
        {!isNA && (
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${pct}%`, background: color }}
          />
        )}
      </div>
      <span className="font-mono text-[11px] w-14 text-right text-[var(--fv-text)]">
        {isNA ? "N/A" : `${value.toFixed(2)}${suffix}`}
      </span>
    </div>
  );
}
