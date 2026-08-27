import { Suspense, useEffect, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, OrbitControls, PerspectiveCamera } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import type { Cell, PointsResponse } from "../api/types";
import type { ViewMode } from "../constants";
import { CellField } from "./CellField";
import { PointCloud } from "./PointCloud";

export type OverlayMode = "raw" | "overlay" | "fovea";
export type CameraPreset = "perspective" | "top" | "front";

interface SceneProps {
  cells: Cell[];
  points: PointsResponse | null;
  viewMode: ViewMode;
  elevationOffset: number;
  overlayMode: OverlayMode;
  selectedIndex: number | null;
  onSelect: (index: number) => void;
  framingRadius: number;
  /** Bump this to re-apply the current preset even if it didn't change (Reset View). */
  cameraToken: { preset: CameraPreset; nonce: number };
}

function SensorOrigin() {
  return (
    <group>
      <mesh position={[0, 0.35, 0]}>
        <sphereGeometry args={[0.28, 16, 16]} />
        <meshStandardMaterial color="#22d3ee" emissive="#22d3ee" emissiveIntensity={0.8} toneMapped={false} />
      </mesh>
      <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.4, 0.5, 32]} />
        <meshBasicMaterial color="#22d3ee" transparent opacity={0.5} />
      </mesh>
    </group>
  );
}

function presetTarget(preset: CameraPreset, radius: number): { position: THREE.Vector3; target: THREE.Vector3 } {
  switch (preset) {
    case "top":
      // Straight bird's-eye view -- the clearest way to read the resolution grid.
      return { position: new THREE.Vector3(0.001, radius * 1.7, 0.001), target: new THREE.Vector3(0, 0, 0) };
    case "front":
      return {
        position: new THREE.Vector3(0, radius * 0.42, radius * 1.35),
        target: new THREE.Vector3(0, radius * 0.12, 0),
      };
    case "perspective":
    default:
      return {
        position: new THREE.Vector3(radius * 0.87, radius * 0.8, radius * 1.0),
        target: new THREE.Vector3(0, radius * 0.05, 0),
      };
  }
}

/** Smoothly flies the camera to a preset position/target whenever `cameraToken`
 * changes (view preset click, or a Reset-View nonce bump). Framed from the
 * frame's actual measured range (framingRadius), not a fixed magic distance,
 * so different frames/sequences with different spatial extents auto-frame
 * sensibly instead of appearing tiny in a huge empty space. */
function CameraRig({
  cameraToken,
  framingRadius,
  controlsRef,
}: {
  cameraToken: { preset: CameraPreset; nonce: number };
  framingRadius: number;
  controlsRef: React.RefObject<OrbitControlsImpl | null>;
}) {
  const { camera } = useThree();
  const goal = useRef({ position: new THREE.Vector3(), target: new THREE.Vector3() });
  const animating = useRef(false);

  useEffect(() => {
    const { position, target } = presetTarget(cameraToken.preset, framingRadius);
    goal.current.position.copy(position);
    goal.current.target.copy(target);
    animating.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraToken.preset, cameraToken.nonce, framingRadius]);

  useFrame(() => {
    if (!animating.current) return;
    const controls = controlsRef.current;
    camera.position.lerp(goal.current.position, 0.09);
    if (controls) {
      controls.target.lerp(goal.current.target, 0.09);
      controls.update();
    }
    if (camera.position.distanceTo(goal.current.position) < 0.05) {
      animating.current = false;
    }
  });

  return null;
}

export function Scene({
  cells,
  points,
  viewMode,
  elevationOffset,
  overlayMode,
  selectedIndex,
  onSelect,
  framingRadius,
  cameraToken,
}: SceneProps) {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const showPoints = overlayMode === "raw" || overlayMode === "overlay";
  const showCells = overlayMode === "overlay" || overlayMode === "fovea";
  const cellOpacity = overlayMode === "overlay" ? 0.62 : 1;
  const pointOpacity = overlayMode === "overlay" ? 0.32 : 0.55;

  return (
    <Canvas dpr={[1, 1.5]} gl={{ antialias: true }} shadows={false}>
      <color attach="background" args={["#070a12"]} />
      <fog attach="fog" args={["#070a12", 90, 320]} />
      <PerspectiveCamera makeDefault position={[26, 24, 30]} fov={50} near={0.05} far={800} />
      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={3}
        maxDistance={350}
        maxPolarAngle={Math.PI * 0.49}
      />
      <CameraRig cameraToken={cameraToken} framingRadius={framingRadius} controlsRef={controlsRef} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[40, 60, 20]} intensity={1.1} />
      <directionalLight position={[-30, 20, -40]} intensity={0.35} color="#60a5fa" />

      <Grid
        position={[0, 0, 0]}
        args={[10, 10]}
        cellSize={2}
        cellThickness={0.4}
        cellColor="#1c2536"
        sectionSize={10}
        sectionThickness={1}
        sectionColor="#2a3550"
        fadeDistance={220}
        fadeStrength={1.5}
        infiniteGrid
      />
      <SensorOrigin />

      <Suspense fallback={null}>
        {points && showPoints && (
          <PointCloud points={points} elevationOffset={elevationOffset} visible opacity={pointOpacity} />
        )}
        {showCells && (
          <CellField
            cells={cells}
            viewMode={viewMode}
            elevationOffset={elevationOffset}
            selectedIndex={selectedIndex}
            onSelect={onSelect}
            opacity={cellOpacity}
          />
        )}
      </Suspense>
    </Canvas>
  );
}
