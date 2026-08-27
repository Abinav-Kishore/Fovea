import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import { base64ToArrayBuffer, decodeFvch1, type DecodedChunkSet } from "../api/fvch1";
import type { ChunksResponse, DatasetName } from "../api/types";

export type StreamingStage = "idle" | "loading-coarse" | "coarse" | "loading-fine" | "fine" | "error";

export interface SpatialStreamingState {
  chunkSet: DecodedChunkSet | null;
  meta: ChunksResponse | null; // hierarchy/LOD metadata -- selected_node_count, chunk_count, payload_bytes, timing, lod_config
  stage: StreamingStage;
  error: string | null;
  fetchMs: number | null;
  decodeMs: number | null;
}

// Two fixed LOD passes -- coarse first (fast, few nodes, scene never sits
// blank), then a finer pass that replaces it once ready. Deliberately a
// simple 2-stage cascade, not continuous/unbounded refinement -- see
// src/spatial/lod.py's LODSelectionConfig.detail_threshold for what this
// value means (cell_size_m / distance_to_viewpoint threshold to descend).
const COARSE_DETAIL_THRESHOLD = 0.3;
const FINE_DETAIL_THRESHOLD = 0.02;

const INITIAL_STATE: SpatialStreamingState = {
  chunkSet: null,
  meta: null,
  stage: "idle",
  error: null,
  fetchMs: null,
  decodeMs: null,
};

function cacheKey(detailThreshold: number, vx: number, vy: number): string {
  // Rounded so tiny sub-decimeter viewpoint jitter doesn't miss the cache.
  return `${detailThreshold}|${vx.toFixed(1)}|${vy.toFixed(1)}`;
}

function messageFor(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Chunk streaming failed.";
}

/**
 * Streams the LOD-selected chunk representation for one (dataset, sequence,
 * frameId): calls the EXISTING `/chunks` endpoint (which runs hierarchy
 * construction + LOD selection server-side -- src/spatial/hierarchy.py,
 * lod.py, both unmodified; this hook duplicates none of that logic, it only
 * passes the desired parameters through) and decodes the FVCH1 binary
 * payload client-side (src/api/fvch1.ts).
 *
 * - Loads a COARSE LOD first so the scene is never blank while waiting for
 *   the fine one, then refines to a finer LOD without blocking on it
 *   (progressive loading).
 * - Chunks already fetched for the CURRENT frame + exact parameter
 *   combination (detail threshold + rounded viewpoint) are cached in a
 *   ref and reused -- switching overlay/view mode back and forth, or
 *   returning to a previously-visited viewpoint, never re-fetches.
 * - The cache is reset whenever (dataset, sequence, frameId) changes --
 *   a new frame's chunks are never mixed with a previous frame's.
 * - A monotonic generation counter + AbortController together guard
 *   against a stale (superseded) request's result being applied after a
 *   newer viewpoint/frame change has already started a new one.
 */
export function useSpatialStreaming(
  dataset: DatasetName,
  sequence: string,
  frameId: string | null,
  viewpoint: { x: number; y: number },
  enabled: boolean
): SpatialStreamingState {
  const [state, setState] = useState<SpatialStreamingState>(INITIAL_STATE);
  const cacheRef = useRef<Map<string, { chunkSet: DecodedChunkSet; meta: ChunksResponse }>>(new Map());
  const generationRef = useRef(0);
  const inFlightRef = useRef<Set<string>>(new Set());

  // New frame/dataset -> the previous frame's cached chunks/in-flight keys
  // are meaningless for this one; drop them and reset to idle.
  useEffect(() => {
    cacheRef.current = new Map();
    inFlightRef.current = new Set();
    generationRef.current += 1;
    setState(INITIAL_STATE);
  }, [dataset, sequence, frameId]);

  useEffect(() => {
    if (!enabled || !frameId) return;
    const myGeneration = ++generationRef.current;
    const controller = new AbortController();

    const fetchOne = async (detailThreshold: number, loadingStage: StreamingStage, doneStage: StreamingStage) => {
      const key = cacheKey(detailThreshold, viewpoint.x, viewpoint.y);
      const cached = cacheRef.current.get(key);
      if (cached) {
        if (generationRef.current !== myGeneration) return;
        setState({ chunkSet: cached.chunkSet, meta: cached.meta, stage: doneStage, error: null, fetchMs: 0, decodeMs: 0 });
        return;
      }
      if (inFlightRef.current.has(key)) return; // avoid duplicate concurrent requests for the same key
      inFlightRef.current.add(key);
      setState((s) => ({ ...s, stage: loadingStage }));

      const fetchStart = performance.now();
      try {
        const resp = await api.getChunks(
          dataset,
          sequence,
          frameId,
          { detailThreshold, viewpointX: viewpoint.x, viewpointY: viewpoint.y },
          controller.signal
        );
        const fetchMs = performance.now() - fetchStart;

        const decodeStart = performance.now();
        const buf = base64ToArrayBuffer(resp.payload_base64);
        const chunkSet = decodeFvch1(buf);
        const decodeMs = performance.now() - decodeStart;

        cacheRef.current.set(key, { chunkSet, meta: resp });
        if (generationRef.current !== myGeneration) return; // superseded while this was in flight
        setState({ chunkSet, meta: resp, stage: doneStage, error: null, fetchMs, decodeMs });
      } catch (err) {
        if (controller.signal.aborted) return;
        if (generationRef.current !== myGeneration) return;
        setState((s) => ({ ...s, stage: "error", error: messageFor(err) }));
      } finally {
        inFlightRef.current.delete(key);
      }
    };

    (async () => {
      await fetchOne(COARSE_DETAIL_THRESHOLD, "loading-coarse", "coarse");
      if (generationRef.current !== myGeneration || controller.signal.aborted) return;
      await fetchOne(FINE_DETAIL_THRESHOLD, "loading-fine", "fine");
    })();

    return () => {
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, sequence, frameId, viewpoint.x, viewpoint.y, enabled]);

  return state;
}
