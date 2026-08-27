import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import type { ThreeEvent } from "@react-three/fiber";
import type { DatasetName } from "../api/types";
import { chunkInfoAt, type ChunkInfo, type DecodedChunkSet } from "../api/fvch1";
import { useSpatialStreaming, type StreamingStage } from "../hooks/useSpatialStreaming";
import { RESOLUTION_COLORS, elevationColor, informationColor, semanticColor, type ViewMode } from "../constants";

export interface StreamingTelemetry {
  stage: StreamingStage;
  nodeCount: number;
  pointCount: number;
  payloadBytes: number;
  fetchMs: number | null;
  decodeMs: number | null;
  fps: number | null;
}

interface ChunkFieldProps {
  dataset: DatasetName;
  sequence: string;
  frameId: string | null;
  viewMode: ViewMode;
  elevationOffset: number;
  opacity?: number;
  onSelect?: (info: ChunkInfo | null) => void;
  onTelemetry?: (t: StreamingTelemetry) => void;
}

// Same true-to-scale "framed tile" visual language as CellField.tsx --
// border + inset fill plate, top face at the region's real elevation.
const MIN_FILL_HEIGHT = 0.06;
const BORDER_HEIGHT = 0.035;
const BORDER_FOOTPRINT_FACTOR = 0.94;
const FILL_FOOTPRINT_FACTOR = 0.74;
const BORDER_COLOR = "#0a0e1a";
const BORDER_COLOR_SELECTED = "#22d3ee";

// Camera-viewpoint change threshold/throttle (Step 7): only report a new
// viewpoint (and thus trigger a new /chunks request) once the camera has
// moved more than this many meters on the ground plane, sampled at most
// ~6x/second -- not every render frame. A prototype-appropriate fixed
// threshold, not a continuous high-frequency stream of requests.
const VIEWPOINT_MOVE_THRESHOLD_M = 4;
const VIEWPOINT_SAMPLE_INTERVAL_S = 0.16;

const _matrix = new THREE.Matrix4();
const _position = new THREE.Vector3();
const _scale = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _color = new THREE.Color();

function displayElevation(set: DecodedChunkSet, i: number, elevationOffset: number): number {
  return Math.max(0, set.elevationMedian[i] - elevationOffset);
}

/** Camera-driven streaming + InstancedMesh rendering of the LOD-selected
 * chunk frontier. Lives inside <Canvas> (needs live camera access via
 * useThree/useFrame) -- Scene.tsx renders this in place of CellField when
 * overlayMode === "streaming". This is the ONLY component that calls
 * useSpatialStreaming; CellField/PointCloud are completely untouched. */
export function ChunkField({
  dataset,
  sequence,
  frameId,
  viewMode,
  elevationOffset,
  opacity = 1,
  onSelect,
  onTelemetry,
}: ChunkFieldProps) {
  const { camera } = useThree();
  const [viewpoint, setViewpoint] = useState({ x: 0, y: 0 });
  const lastSampleRef = useRef(0);
  const lastReportedRef = useRef({ x: 0, y: 0 });
  const reportedOnceRef = useRef(false);

  // Sample the camera's ground-plane position at a fixed low rate and only
  // report a change once it exceeds the move threshold -- the debounce/
  // throttle called for in Step 7. World X/Z map directly to LiDAR x/y (see
  // CellField's own position.set(center_x, ..., center_y) convention).
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (!reportedOnceRef.current) {
      reportedOnceRef.current = true;
      lastSampleRef.current = t;
      lastReportedRef.current = { x: camera.position.x, y: camera.position.z };
      setViewpoint({ x: camera.position.x, y: camera.position.z });
      return;
    }
    if (t - lastSampleRef.current < VIEWPOINT_SAMPLE_INTERVAL_S) return;
    lastSampleRef.current = t;
    const dx = camera.position.x - lastReportedRef.current.x;
    const dy = camera.position.z - lastReportedRef.current.y;
    if (Math.hypot(dx, dy) < VIEWPOINT_MOVE_THRESHOLD_M) return;
    lastReportedRef.current = { x: camera.position.x, y: camera.position.z };
    setViewpoint({ x: camera.position.x, y: camera.position.z });
  });

  const streaming = useSpatialStreaming(dataset, sequence, frameId, viewpoint, true);
  const { chunkSet, meta, stage, fetchMs, decodeMs } = streaming;

  // FPS -- a small rolling counter, reported (throttled) via onTelemetry
  // rather than causing its own re-render every frame.
  const fpsFramesRef = useRef(0);
  const fpsWindowStartRef = useRef(0);
  const fpsRef = useRef<number | null>(null);
  useFrame((state) => {
    fpsFramesRef.current += 1;
    const elapsed = state.clock.elapsedTime - fpsWindowStartRef.current;
    if (elapsed >= 0.5) {
      fpsRef.current = fpsFramesRef.current / elapsed;
      fpsFramesRef.current = 0;
      fpsWindowStartRef.current = state.clock.elapsedTime;
    }
  });

  useEffect(() => {
    const id = window.setInterval(() => {
      onTelemetry?.({
        stage,
        nodeCount: meta?.chunk_count ?? chunkSet?.count ?? 0,
        pointCount: meta?.selected_point_count ?? 0,
        payloadBytes: meta?.payload_bytes ?? 0,
        fetchMs,
        decodeMs,
        fps: fpsRef.current,
      });
    }, 500);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage, meta, chunkSet, fetchMs, decodeMs]);

  const fillRef = useRef<THREE.InstancedMesh>(null);
  const borderRef = useRef<THREE.InstancedMesh>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  // A new chunk set invalidates any previous selection -- an old instanceId
  // no longer necessarily refers to the same node.
  useEffect(() => {
    setSelectedIndex(null);
    onSelect?.(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chunkSet]);

  const maxElevationDisplay = useMemo(() => {
    if (!chunkSet || chunkSet.count === 0) return 0.001;
    let max = 0.001;
    for (let i = 0; i < chunkSet.count; i++) {
      const h = displayElevation(chunkSet, i, elevationOffset);
      if (h > max) max = h;
    }
    return max;
  }, [chunkSet, elevationOffset]);

  useLayoutEffect(() => {
    const fill = fillRef.current;
    const border = borderRef.current;
    if (!fill || !border || !chunkSet || chunkSet.count === 0) return;

    for (let i = 0; i < chunkSet.count; i++) {
      const elevationDisplay = displayElevation(chunkSet, i, elevationOffset);
      const spread = chunkSet.elevationMax[i] - chunkSet.elevationMin[i];
      const fillThickness = Math.min(0.6, Math.max(MIN_FILL_HEIGHT, spread));
      const cellSize = chunkSet.cellSizeM[i];
      const borderFootprint = Math.max(0.02, cellSize * BORDER_FOOTPRINT_FACTOR);
      const fillFootprint = Math.max(0.015, cellSize * FILL_FOOTPRINT_FACTOR);
      const cx = chunkSet.centerX[i];
      const cy = chunkSet.centerY[i];

      _position.set(cx, elevationDisplay - BORDER_HEIGHT / 2, cy);
      _scale.set(borderFootprint, BORDER_HEIGHT, borderFootprint);
      _matrix.compose(_position, _quat, _scale);
      border.setMatrixAt(i, _matrix);
      _color.set(i === selectedIndex ? BORDER_COLOR_SELECTED : BORDER_COLOR);
      border.setColorAt(i, _color);

      _position.set(cx, elevationDisplay + 0.01 - fillThickness / 2, cy);
      _scale.set(fillFootprint, fillThickness, fillFootprint);
      _matrix.compose(_position, _quat, _scale);
      fill.setMatrixAt(i, _matrix);

      let hex: string;
      if (viewMode === "semantic") {
        hex = semanticColor(chunkSet.stringTable[chunkSet.semanticClassIdx[i]]);
      } else if (viewMode === "resolution") {
        hex = RESOLUTION_COLORS[chunkSet.level[i]] ?? "#888888";
      } else if (viewMode === "elevation") {
        hex = elevationColor(elevationDisplay / maxElevationDisplay);
      } else {
        hex = informationColor(chunkSet.informationScore[i]);
      }
      _color.set(hex);
      fill.setColorAt(i, _color);
    }
    fill.instanceMatrix.needsUpdate = true;
    border.instanceMatrix.needsUpdate = true;
    if (fill.instanceColor) fill.instanceColor.needsUpdate = true;
    if (border.instanceColor) border.instanceColor.needsUpdate = true;
    fill.computeBoundingSphere();
    border.computeBoundingSphere();
  }, [chunkSet, viewMode, elevationOffset, maxElevationDisplay, selectedIndex]);

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    if (e.instanceId === undefined || !chunkSet) return;
    setSelectedIndex(e.instanceId);
    onSelect?.(chunkInfoAt(chunkSet, e.instanceId));
  };

  if (!chunkSet || chunkSet.count === 0) return null;

  return (
    <group>
      <instancedMesh
        key={`border-${chunkSet.count}-${stage}`}
        ref={borderRef}
        args={[undefined, undefined, chunkSet.count]}
        frustumCulled={false}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial toneMapped={false} roughness={0.9} metalness={0} transparent={opacity < 1} opacity={opacity} />
      </instancedMesh>
      <instancedMesh
        key={`fill-${chunkSet.count}-${stage}`}
        ref={fillRef}
        args={[undefined, undefined, chunkSet.count]}
        onClick={handleClick}
        frustumCulled={false}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial toneMapped={false} roughness={0.5} metalness={0.05} transparent={opacity < 1} opacity={opacity} />
      </instancedMesh>
    </group>
  );
}
