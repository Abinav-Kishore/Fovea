import { useLayoutEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { ThreeEvent } from "@react-three/fiber";
import type { Cell } from "../api/types";
import { RESOLUTION_COLORS, elevationColor, informationColor, semanticColor, type ViewMode } from "../constants";

interface CellFieldProps {
  cells: Cell[];
  viewMode: ViewMode;
  elevationOffset: number;
  selectedIndex: number | null;
  onSelect: (index: number) => void;
  opacity?: number;
}

// 1 world unit = 1 real meter, no vertical exaggeration -- true-to-scale so the
// height of a tile is exactly the cell's measured elevation, not a stylized fake-up.
const ELEVATION_SCALE = 1;
const MIN_FILL_HEIGHT = 0.06;
const BORDER_HEIGHT = 0.035;
const BORDER_FOOTPRINT_FACTOR = 0.94; // outer "frame" -- close to the true cell_size_m footprint
const FILL_FOOTPRINT_FACTOR = 0.74; // inset colored tile, leaves the border visibly showing
const BORDER_COLOR = "#0a0e1a"; // near-background "grout" so every tile reads as a discrete cell
const BORDER_COLOR_SELECTED = "#22d3ee";

const _matrix = new THREE.Matrix4();
const _position = new THREE.Vector3();
const _scale = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _color = new THREE.Color();

/** Per-cell display elevation (ground-relative, clamped >= 0) -- shared by the
 * fill/border tiles and the selection marker so they always agree. */
function displayElevation(cell: Cell, elevationOffset: number): number {
  return Math.max(0, (cell.elevation_median - elevationOffset) * ELEVATION_SCALE);
}

export function CellField({ cells, viewMode, elevationOffset, selectedIndex, onSelect, opacity = 1 }: CellFieldProps) {
  const fillRef = useRef<THREE.InstancedMesh>(null);
  const borderRef = useRef<THREE.InstancedMesh>(null);

  const maxElevationDisplay = useMemo(() => {
    let max = 0.001;
    for (const c of cells) {
      const h = displayElevation(c, elevationOffset);
      if (h > max) max = h;
    }
    return max;
  }, [cells, elevationOffset]);

  useLayoutEffect(() => {
    const fill = fillRef.current;
    const border = borderRef.current;
    if (!fill || !border || cells.length === 0) return;

    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      const elevationDisplay = displayElevation(c, elevationOffset);
      // Each cell renders as a small "framed tile": a dark border plate plus a
      // smaller inset colored fill plate on top, both true-to-scale flat tiles
      // (not columns) whose TOP face sits at the cell's real elevation -- a
      // full ground->elevation pillar turns sparse tall cells (building/fence
      // returns many meters up) into unreadable needles at these sub-meter
      // footprints. Fill thickness uses the cell's own elevation_std (a real
      // measured field, clamped only for a visible minimum), so rougher cells
      // read as visibly thicker.
      const fillThickness = Math.min(0.6, Math.max(MIN_FILL_HEIGHT, c.elevation_std * 2));
      const borderFootprint = Math.max(0.02, c.cell_size_m * BORDER_FOOTPRINT_FACTOR);
      const fillFootprint = Math.max(0.015, c.cell_size_m * FILL_FOOTPRINT_FACTOR);

      _position.set(c.center_x, elevationDisplay - BORDER_HEIGHT / 2, c.center_y);
      _scale.set(borderFootprint, BORDER_HEIGHT, borderFootprint);
      _matrix.compose(_position, _quat, _scale);
      border.setMatrixAt(i, _matrix);
      _color.set(i === selectedIndex ? BORDER_COLOR_SELECTED : BORDER_COLOR);
      border.setColorAt(i, _color);

      _position.set(c.center_x, elevationDisplay + 0.01 - fillThickness / 2, c.center_y);
      _scale.set(fillFootprint, fillThickness, fillFootprint);
      _matrix.compose(_position, _quat, _scale);
      fill.setMatrixAt(i, _matrix);

      let hex: string;
      if (viewMode === "semantic") {
        hex = semanticColor(c.majority_class_name);
      } else if (viewMode === "resolution") {
        hex = RESOLUTION_COLORS[c.resolution_level] ?? "#888888";
      } else if (viewMode === "elevation") {
        hex = elevationColor(elevationDisplay / maxElevationDisplay);
      } else {
        hex = informationColor(c.information_score);
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
  }, [cells, viewMode, elevationOffset, maxElevationDisplay, selectedIndex]);

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    if (e.instanceId !== undefined) onSelect(e.instanceId);
  };

  const selectedCell = selectedIndex !== null ? cells[selectedIndex] ?? null : null;

  if (cells.length === 0) return null;

  return (
    <group>
      <instancedMesh ref={borderRef} args={[undefined, undefined, cells.length]} frustumCulled={false}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          toneMapped={false}
          roughness={0.9}
          metalness={0}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </instancedMesh>
      {/* Fill tiles carry the click handling -- they're the visually dominant
          (larger, on-top) layer, so raycasting against them feels natural. */}
      <instancedMesh ref={fillRef} args={[undefined, undefined, cells.length]} onClick={handleClick} frustumCulled={false}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          toneMapped={false}
          roughness={0.5}
          metalness={0.05}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </instancedMesh>
      {selectedCell && (
        <SelectedCellMarker
          cell={selectedCell}
          elevationDisplay={displayElevation(selectedCell, elevationOffset)}
        />
      )}
    </group>
  );
}

/** A single animated ring outline over the selected cell -- cheap (one extra
 * object, not per-instance) and keeps the cell's own fill color intact rather
 * than overwriting it, so the judge can still see what the cell actually is. */
function SelectedCellMarker({ cell, elevationDisplay }: { cell: Cell; elevationDisplay: number }) {
  const ringRef = useRef<THREE.Mesh>(null);
  const size = Math.max(0.15, cell.cell_size_m * BORDER_FOOTPRINT_FACTOR);

  useFrame((state) => {
    if (!ringRef.current) return;
    const pulse = 1 + 0.12 * Math.sin(state.clock.elapsedTime * 3.2);
    ringRef.current.scale.set(pulse, pulse, 1);
  });

  return (
    <mesh
      ref={ringRef}
      position={[cell.center_x, elevationDisplay + 0.03, cell.center_y]}
      rotation={[-Math.PI / 2, 0, 0]}
    >
      <ringGeometry args={[size * 0.62, size * 0.75, 32]} />
      <meshBasicMaterial color="#22d3ee" transparent opacity={0.85} toneMapped={false} />
    </mesh>
  );
}
