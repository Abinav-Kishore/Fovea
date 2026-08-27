import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { DatasetName, FrameDiagnostics, MapResponse, MapSummary, PointsResponse } from "../api/types";

export interface FrameData {
  // Stage 1: fast metadata, appears first.
  diagnostics: FrameDiagnostics | null;
  diagnosticsLoading: boolean;

  // Stage 2: the FOVEA map -- the main hero visualization. Fetched alongside
  // diagnostics (both are needed early and neither blocks the other), but
  // deliberately BEFORE the raw point cloud, so the resolution/information
  // view is never held hostage by the (usually larger) points payload.
  foveaMap: MapResponse | null;
  foveaLoading: boolean;

  // Stage 3: raw point cloud overlay. Kicked off after stage 1+2 settle.
  points: PointsResponse | null;
  pointsLoading: boolean;

  // Stage 4: secondary analytics for the benchmark panel (uniform +
  // distance-adaptive summaries only -- cells aren't needed for that panel).
  uniformSummary: MapSummary | null;
  distanceSummary: MapSummary | null;
  secondaryLoading: boolean;

  // A hard error on the critical path (diagnostics/FOVEA) -- e.g. the frame
  // doesn't exist. Secondary-stage failures are reported per-section instead
  // of blocking the whole view (see pointsError/secondaryError).
  error: string | null;
  pointsError: string | null;
  secondaryError: string | null;
}

const POINT_CLOUD_CAP = 20000;

const INITIAL_STATE: FrameData = {
  diagnostics: null,
  diagnosticsLoading: false,
  foveaMap: null,
  foveaLoading: false,
  points: null,
  pointsLoading: false,
  uniformSummary: null,
  distanceSummary: null,
  secondaryLoading: false,
  error: null,
  pointsError: null,
  secondaryError: null,
};

function messageFor(err: unknown): string {
  return err instanceof ApiError ? err.message : "Failed to reach FOVEA backend.";
}

/**
 * Fetches everything the dashboard needs for one (sequence, frameId) as a
 * progressive sequence of independent updates, NOT a single Promise.all --
 * so a slow request (e.g. the point cloud) never holds up rendering data
 * that's already ready (e.g. the FOVEA map). Cancels in-flight requests when
 * the selection changes again before they resolve, so rapid frame-stepping
 * never renders a stale frame's data.
 */
export function useFrameData(dataset: DatasetName, sequence: string, frameId: string | null): FrameData {
  const [state, setState] = useState<FrameData>(INITIAL_STATE);

  useEffect(() => {
    if (!frameId) return;
    const controller = new AbortController();
    const { signal } = controller;

    setState({
      ...INITIAL_STATE,
      diagnosticsLoading: true,
      foveaLoading: true,
      pointsLoading: true,
      secondaryLoading: true,
    });

    let cancelled = false;

    (async () => {
      // Stage 1 + 2 together: both are fast/important and independent of
      // each other -- neither should wait on the other.
      const diagnosticsPromise = api
        .getDiagnostics(dataset, sequence, frameId, signal)
        .then((diagnostics) => {
          if (cancelled) return;
          setState((s) => ({ ...s, diagnostics, diagnosticsLoading: false }));
        })
        .catch((err: unknown) => {
          if (cancelled || signal.aborted) return;
          setState((s) => ({ ...s, diagnosticsLoading: false, error: messageFor(err) }));
        });

      const foveaPromise = api
        .getMap(dataset, sequence, frameId, "fovea", { includeCells: true }, signal)
        .then((foveaMap) => {
          if (cancelled) return;
          setState((s) => ({ ...s, foveaMap, foveaLoading: false }));
        })
        .catch((err: unknown) => {
          if (cancelled || signal.aborted) return;
          setState((s) => ({ ...s, foveaLoading: false, error: messageFor(err) }));
        });

      await Promise.allSettled([diagnosticsPromise, foveaPromise]);
      if (cancelled) return;

      // Stage 3: raw point cloud -- fired after the FOVEA map is already in
      // flight/settled, so it never blocks the hero visualization.
      api
        .getPoints(dataset, sequence, frameId, POINT_CLOUD_CAP, signal)
        .then((points) => {
          if (cancelled) return;
          setState((s) => ({ ...s, points, pointsLoading: false }));
        })
        .catch((err: unknown) => {
          if (cancelled || signal.aborted) return;
          setState((s) => ({ ...s, pointsLoading: false, pointsError: messageFor(err) }));
        });

      // Stage 4: secondary analytics for the benchmark panel.
      Promise.all([
        api.getMap(dataset, sequence, frameId, "uniform", { includeCells: false }, signal),
        api.getMap(dataset, sequence, frameId, "distance_adaptive", { includeCells: false }, signal),
      ])
        .then(([uniformMap, distanceMap]) => {
          if (cancelled) return;
          setState((s) => ({
            ...s,
            uniformSummary: uniformMap.summary,
            distanceSummary: distanceMap.summary,
            secondaryLoading: false,
          }));
        })
        .catch((err: unknown) => {
          if (cancelled || signal.aborted) return;
          setState((s) => ({ ...s, secondaryLoading: false, secondaryError: messageFor(err) }));
        });
    })();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [dataset, sequence, frameId]);

  return state;
}
