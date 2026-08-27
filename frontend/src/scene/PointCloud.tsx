import { useMemo } from "react";
import * as THREE from "three";
import type { PointsResponse } from "../api/types";
import { elevationColor, intensityColor, semanticColor, type ViewMode } from "../constants";
import { classNameForId } from "../semanticLookup";

interface PointCloudProps {
  points: PointsResponse;
  elevationOffset: number;
  visible: boolean;
  opacity?: number;
  /** Semantic, Elevation, and Intensity are meaningful for a raw point --
   * Resolution/Information fall back to semantic coloring, since there is
   * no per-point resolution level or FOVEA information score (those only
   * exist once a point has been aggregated into a Cell). */
  viewMode?: ViewMode;
}

const ELEVATION_SCALE = 1; // matches CellField -- true-to-scale, no exaggeration

// A small soft-edged circular sprite, generated once on a <canvas> and
// reused for every PointCloud instance/frame (module-level cache, not
// regenerated per render). Three.js's default point sprite is a hard-edged
// square; sampling this texture with alphaTest+transparent instead makes
// each point read as a soft round dot -- a real LiDAR sample, not a flat
// block -- with zero extra points, meshes, or geometry.
let _circleSpriteTexture: THREE.Texture | null = null;
function getCircleSpriteTexture(): THREE.Texture {
  if (_circleSpriteTexture) return _circleSpriteTexture;
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    const r = size / 2;
    const gradient = ctx.createRadialGradient(r, r, 0, r, r, r);
    gradient.addColorStop(0, "rgba(255,255,255,1)");
    gradient.addColorStop(0.55, "rgba(255,255,255,0.9)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  _circleSpriteTexture = tex;
  return tex;
}

export function PointCloud({ points, elevationOffset, visible, opacity = 0.55, viewMode = "semantic" }: PointCloudProps) {
  const { positions, colors, pointSize } = useMemo(() => {
    const n = points.x.length;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const c = new THREE.Color();

    const showElevation = viewMode === "elevation";
    const showIntensity = viewMode === "intensity";
    // Real per-frame ranges (own min/max of the actual values being drawn,
    // not fabricated/assumed). Elevation ramp matches CellField/ChunkField,
    // normalized against this point set instead of the FOVEA cells'.
    let maxElevDisplay = 0.001;
    let minIntensity = Infinity;
    let maxIntensity = -Infinity;
    if (showElevation) {
      for (let i = 0; i < n; i++) {
        const h = Math.max(0, (points.z[i] - elevationOffset) * ELEVATION_SCALE);
        if (h > maxElevDisplay) maxElevDisplay = h;
      }
    }
    if (showIntensity) {
      // Real LiDAR intensity is heavily right-skewed (measured on
      // SemanticPOSS frame 0: median 8, p99 76, max 255 -- a handful of
      // retroreflective returns like signs). A raw min/max normalization
      // crushes the vast majority of points to near-zero (near-invisible
      // against the dark background). Clipping the top 1% instead -- same
      // p99 convention this codebase already uses for range stats
      // (FrameDiagnostics.range_stats_m.p99) -- keeps the ramp meaningful
      // for the points that actually vary; the rare points above p99 just
      // saturate to the brightest color, the same way an overexposed pixel
      // clips, not an error and not a fabricated value.
      const sorted = Array.from(points.intensity).sort((a, b) => a - b);
      minIntensity = sorted[0] ?? 0;
      const p99Index = Math.min(sorted.length - 1, Math.round(0.99 * (sorted.length - 1)));
      maxIntensity = sorted.length ? sorted[p99Index] : 1;
      if (!Number.isFinite(minIntensity)) minIntensity = 0;
      if (!Number.isFinite(maxIntensity) || maxIntensity <= minIntensity) maxIntensity = minIntensity + 1;
    }
    const intensityRange = maxIntensity - minIntensity;

    for (let i = 0; i < n; i++) {
      positions[i * 3] = points.x[i];
      const h = Math.max(0, (points.z[i] - elevationOffset) * ELEVATION_SCALE);
      positions[i * 3 + 1] = h;
      positions[i * 3 + 2] = points.y[i];

      if (showElevation) {
        c.set(elevationColor(h / maxElevDisplay));
      } else if (showIntensity) {
        c.set(intensityColor((points.intensity[i] - minIntensity) / intensityRange));
      } else {
        c.set(semanticColor(classNameForId(points.semantic_id[i])));
      }
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    // Adaptive point size: a denser returned cloud reads better with
    // slightly smaller points (avoids a "mush" of overlapping dots at real
    // sensor density); a sparser/subsampled one reads better a bit larger
    // so the scene doesn't look empty. Two-tier, not per-frame-tuned --
    // cheap and predictable, not a new optimization pass. Nudged down
    // slightly from Phase 4AB now that points render as soft round sprites
    // (see getCircleSpriteTexture) rather than flat squares -- a round dot
    // at the same size already reads as smaller/cleaner, so the base sizes
    // were retuned together with the sprite change, not independently.
    const pointSize = n > 30000 ? 0.04 : 0.06;

    return { positions, colors, pointSize };
  }, [points, elevationOffset, viewMode]);

  if (!visible) return null;

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={pointSize}
        map={getCircleSpriteTexture()}
        alphaTest={0.15}
        vertexColors
        sizeAttenuation
        transparent
        opacity={opacity}
        depthWrite={false}
      />
    </points>
  );
}
