import { useMemo } from "react";
import * as THREE from "three";
import type { PointsResponse } from "../api/types";
import { semanticColor } from "../constants";
import { classNameForId } from "../semanticLookup";

interface PointCloudProps {
  points: PointsResponse;
  elevationOffset: number;
  visible: boolean;
  opacity?: number;
}

const ELEVATION_SCALE = 1; // matches CellField -- true-to-scale, no exaggeration

export function PointCloud({ points, elevationOffset, visible, opacity = 0.55 }: PointCloudProps) {
  const { positions, colors } = useMemo(() => {
    const n = points.x.length;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      positions[i * 3] = points.x[i];
      positions[i * 3 + 1] = Math.max(0, (points.z[i] - elevationOffset) * ELEVATION_SCALE);
      positions[i * 3 + 2] = points.y[i];
      c.set(semanticColor(classNameForId(points.semantic_id[i])));
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    return { positions, colors };
  }, [points, elevationOffset]);

  if (!visible) return null;

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.055} vertexColors sizeAttenuation transparent opacity={opacity} />
    </points>
  );
}
