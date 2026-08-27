import { useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import type { ChunkInfo } from "./api/fvch1";
import type { Cell, DatasetName } from "./api/types";
import { BenchmarkPanel } from "./components/BenchmarkPanel";
import { CameraControls } from "./components/CameraControls";
import { FrameControls } from "./components/FrameControls";
import { Header } from "./components/Header";
import { InformationAnalytics } from "./components/InformationAnalytics";
import { PipelineFlow } from "./components/PipelineFlow";
import { RegionInspector } from "./components/RegionInspector";
import { StreamedChunkPanel } from "./components/StreamedChunkPanel";
import { StreamingTelemetryPanel } from "./components/StreamingTelemetryPanel";
import {
  ContextualLabel,
  ElevationLegend,
  InformationLegend,
  IntensityLegend,
  OverlayModeControl,
  ResolutionLegend,
  SemanticLegend,
  ViewModeTabs,
} from "./components/ViewModeTabs";
import { semanticColor, type ViewMode } from "./constants";
import { useFrameData } from "./hooks/useFrameData";
import { useStaticData } from "./hooks/useStaticData";
import type { StreamingTelemetry } from "./scene/ChunkField";
import { Scene, type CameraPreset, type OverlayMode } from "./scene/Scene";
import { initSemanticLookup } from "./semanticLookup";

const AUTOPLAY_INTERVAL_MS = 650;
const DEFAULT_FRAMING_RADIUS = 30;

export default function App() {
  const [dataset, setDataset] = useState<DatasetName>("SemanticPOSS");
  const { status, benchmark, backendConnected } = useStaticData(dataset);

  const [sequences, setSequences] = useState<string[]>([]);
  const [sequence, setSequence] = useState("");
  const [frameIds, setFrameIds] = useState<string[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>("resolution");
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("overlay");
  const [playing, setPlaying] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [cameraToken, setCameraToken] = useState<{ preset: CameraPreset; nonce: number }>({
    preset: "perspective",
    nonce: 0,
  });

  // Phase 3D: state for the "Streaming (LOD)" overlay mode -- selected
  // chunk (sidebar inspector) and live telemetry (LOD stage/nodes/points/
  // payload/FPS). Both come from Scene -> ChunkField via callback props,
  // same lifting pattern selectedIndex/onSelect already uses.
  const [selectedChunk, setSelectedChunk] = useState<ChunkInfo | null>(null);
  const [streamingTelemetry, setStreamingTelemetry] = useState<StreamingTelemetry | null>(null);

  // Semantic class legend/lookup, sourced from the backend's ontology for the
  // CURRENT dataset -- re-fetched on every dataset switch. This matters: a
  // raw class id means something different in each ontology (e.g. id 7 is
  // "car" in SemanticPOSS but "sky" in RELLIS-3D), so a stale lookup from the
  // previous dataset would mislabel every point in the raw cloud.
  useEffect(() => {
    const controller = new AbortController();
    api
      .getSemanticClasses(dataset, controller.signal)
      .then((r) => initSemanticLookup(r.classes))
      .catch(() => undefined);
    return () => controller.abort();
  }, [dataset]);

  // Sequence list for the CURRENT dataset -- reloaded on every dataset
  // switch, resetting to that dataset's first sequence. This is what makes
  // switching datasets "reload the sequence list, reset to the first valid
  // sequence/frame" rather than carrying over a sequence id that may not
  // even exist in the new dataset (e.g. SemanticPOSS "00" vs RELLIS "00000").
  useEffect(() => {
    const controller = new AbortController();
    api
      .getSequences(dataset, controller.signal)
      .then((r) => {
        setSequences(r.sequences);
        setSequence(r.sequences[0] ?? "");
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [dataset]);

  // Frame list for the current (dataset, sequence) pair.
  useEffect(() => {
    if (!sequence) return;
    const controller = new AbortController();
    api
      .getFrames(dataset, sequence, controller.signal)
      .then((r) => {
        setFrameIds(r.frame_ids);
        setFrameIndex(0);
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [dataset, sequence]);

  const frameId = frameIds[frameIndex] ?? null;
  const {
    diagnostics,
    diagnosticsLoading,
    foveaMap,
    foveaLoading,
    points,
    pointsLoading,
    pointsError,
    uniformSummary,
    distanceSummary,
    secondaryLoading,
    secondaryError,
    error,
  } = useFrameData(dataset, sequence, frameId);

  // Reset the selected cell whenever the frame changes underneath it.
  useEffect(() => setSelectedIndex(null), [frameId]);
  useEffect(() => {
    setSelectedChunk(null);
    setStreamingTelemetry(null);
  }, [frameId, dataset]);

  // Lightweight autoplay: advance one frame at a fixed interval while playing.
  useEffect(() => {
    if (!playing || frameIds.length === 0) return;
    const id = setInterval(() => {
      setFrameIndex((i) => (i + 1) % frameIds.length);
    }, AUTOPLAY_INTERVAL_MS);
    return () => clearInterval(id);
  }, [playing, frameIds.length]);

  const cells: Cell[] = foveaMap?.cells ?? [];
  const selectedCell = selectedIndex !== null ? cells[selectedIndex] ?? null : null;
  const elevationOffset = diagnostics?.z_range?.[0] ?? 0;

  // Camera framing radius, derived from this frame's actual measured range
  // (frame_diagnostics.range_stats_m.mean, a real backend field) -- so the
  // default view auto-frames each frame's real spatial extent instead of a
  // fixed magic distance that could leave the scene tiny or overwhelming.
  const framingRadius = useMemo(() => {
    if (!diagnostics) return DEFAULT_FRAMING_RADIUS;
    const mean = diagnostics.range_stats_m.mean;
    return Math.min(60, Math.max(15, mean * 1.8));
  }, [diagnostics]);

  // Only the semantic classes actually present in this frame's FOVEA cells --
  // never the full static label set -- for a compact, meaningful legend.
  const presentSemanticClasses = useMemo(() => {
    const names = new Set(cells.map((c) => c.majority_class_name));
    return Array.from(names)
      .sort()
      .map((name) => ({ name, color: semanticColor(name) }));
  }, [cells]);

  const handleDatasetChange = (d: DatasetName) => {
    if (d === dataset) return;
    setPlaying(false);
    setSelectedIndex(null);
    // Clear sequence/frameIds SYNCHRONOUSLY, in the same render as the
    // dataset switch. Without this, useFrameData's effect (deps: [dataset,
    // sequence, frameId]) fires on the very next render with the NEW dataset
    // but the OLD dataset's sequence id still in state (e.g. dataset=
    // "RELLIS-3D" + sequence="00", which doesn't exist for RELLIS) -- a real
    // stale-pair request this exact bug produced during testing. Clearing
    // frameIds makes frameId resolve to null, so useFrameData skips fetching
    // entirely until the new dataset's own sequence list has loaded.
    setSequence("");
    setFrameIds([]);
    setFrameIndex(0);
    setDataset(d);
  };

  const handleCameraPreset = (preset: CameraPreset) => setCameraToken((t) => ({ preset, nonce: t.nonce + 1 }));
  const handleCameraReset = () => setCameraToken((t) => ({ ...t, nonce: t.nonce + 1 }));

  return (
    <div
      className="h-screen w-screen grid overflow-hidden"
      style={{
        gridTemplateRows: "56px 1fr auto",
        gridTemplateColumns: "1fr 360px",
        gridTemplateAreas: `"header header" "hero sidebar" "benchmark benchmark"`,
      }}
    >
      <Header
        dataset={dataset}
        onDatasetChange={handleDatasetChange}
        availableDatasets={status?.available_datasets ?? []}
        sequence={sequence}
        frameId={frameId}
        backendConnected={backendConnected}
      />

      <main style={{ gridArea: "hero" }} className="relative">
        <Scene
          cells={cells}
          points={points}
          viewMode={viewMode}
          elevationOffset={elevationOffset}
          overlayMode={overlayMode}
          selectedIndex={selectedIndex}
          onSelect={setSelectedIndex}
          framingRadius={framingRadius}
          cameraToken={cameraToken}
          dataset={dataset}
          sequence={sequence}
          frameId={frameId}
          onChunkSelect={setSelectedChunk}
          onStreamingTelemetry={setStreamingTelemetry}
        />

        <div className="absolute top-3 left-3 right-3 flex items-start justify-between pointer-events-none gap-3">
          <div className="flex flex-col gap-2">
            <div className="flex items-start gap-2">
              <ViewModeTabs mode={viewMode} onChange={setViewMode} />
              <OverlayModeControl mode={overlayMode} onChange={setOverlayMode} />
            </div>
            <div className="pointer-events-auto pl-1">
              <ContextualLabel mode={viewMode} />
            </div>
            {overlayMode === "streaming" && (
              <div className="pointer-events-auto">
                <StreamingTelemetryPanel telemetry={streamingTelemetry} />
              </div>
            )}
          </div>
          <div className="pointer-events-auto w-64 shrink-0">
            <PipelineFlow />
          </div>
        </div>

        <div className="absolute bottom-3 left-3 flex items-end gap-2">
          {viewMode === "resolution" && <ResolutionLegend />}
          {viewMode === "semantic" && <SemanticLegend classes={presentSemanticClasses} />}
          {viewMode === "information" && <InformationLegend />}
          {viewMode === "elevation" && <ElevationLegend />}
          {viewMode === "intensity" && <IntensityLegend />}
        </div>

        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-2">
          <FrameControls
            sequence={sequence}
            sequences={sequences}
            onSequenceChange={(s) => {
              setPlaying(false);
              setSequence(s);
            }}
            frameIds={frameIds}
            frameIndex={frameIndex}
            onFrameIndexChange={setFrameIndex}
            playing={playing}
            onTogglePlaying={() => setPlaying((p) => !p)}
          />
        </div>

        <div className="absolute bottom-3 right-3">
          <CameraControls preset={cameraToken.preset} onPreset={handleCameraPreset} onReset={handleCameraReset} />
        </div>

        {(diagnosticsLoading || foveaLoading || pointsLoading || pointsError || error) && (
          <div className="absolute top-3 right-3 mt-14 flex flex-col items-end gap-1.5">
            {error && (
              <div className="fv-panel px-3 py-2 text-xs text-[var(--fv-red)] max-w-xs">{error}</div>
            )}
            {!error && (diagnosticsLoading || foveaLoading) && (
              <div className="fv-panel px-3 py-2 text-xs text-[var(--fv-text-dim)]">
                Loading FOVEA map…
              </div>
            )}
            {!error && !foveaLoading && pointsLoading && (
              <div className="fv-panel px-3 py-2 text-xs text-[var(--fv-text-dim)]">
                Loading LiDAR points…
              </div>
            )}
            {!error && pointsError && (
              <div className="fv-panel px-3 py-2 text-xs text-[var(--fv-red)] max-w-xs">
                Point cloud unavailable: {pointsError}
              </div>
            )}
          </div>
        )}
        {!error && secondaryError && (
          <div className="absolute bottom-16 right-3 fv-panel px-3 py-2 text-xs text-[var(--fv-red)] max-w-xs">
            Baseline comparison unavailable: {secondaryError}
          </div>
        )}
      </main>

      <aside style={{ gridArea: "sidebar" }} className="fv-scrollbar overflow-y-auto p-3 flex flex-col gap-3">
        {overlayMode === "streaming" ? (
          <StreamedChunkPanel chunk={selectedChunk} />
        ) : (
          <RegionInspector dataset={dataset} sequence={sequence} frameId={frameId} cell={selectedCell} />
        )}
        <InformationAnalytics
          cells={cells}
          summary={foveaMap?.summary ?? null}
          selectedResolutionLevel={selectedCell?.resolution_level ?? null}
        />
      </aside>

      <BenchmarkPanel
        uniform={uniformSummary}
        distanceAdaptive={distanceSummary}
        fovea={foveaMap?.summary ?? null}
        benchmark={benchmark}
        secondaryLoading={secondaryLoading}
      />
    </div>
  );
}
