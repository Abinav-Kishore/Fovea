import { ArrowRight, Boxes, Gauge, Radar, ScanEye, Waypoints } from "lucide-react";

const STEPS = [
  { icon: Radar, label: "Raw LiDAR" },
  { icon: ScanEye, label: "Understand" },
  { icon: Gauge, label: "Score" },
  { icon: Waypoints, label: "Adaptive resolution" },
  { icon: Boxes, label: "2.5D map" },
];

export function PipelineFlow() {
  return (
    <div className="fv-panel p-3 flex items-center justify-between">
      {STEPS.map((step, i) => (
        <div key={step.label} className="flex items-center gap-2">
          <div className="flex flex-col items-center gap-1">
            <div className="w-7 h-7 rounded-lg bg-[var(--fv-cyan)]/10 border border-[var(--fv-cyan)]/25 flex items-center justify-center text-[var(--fv-cyan)]">
              <step.icon size={13} />
            </div>
            <span className="text-[9px] text-[var(--fv-text-dim)] whitespace-nowrap">{step.label}</span>
          </div>
          {i < STEPS.length - 1 && <ArrowRight size={11} className="text-[var(--fv-text-faint)] mb-3.5" />}
        </div>
      ))}
    </div>
  );
}
