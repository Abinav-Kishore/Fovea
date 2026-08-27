import type {
  BenchmarkResponse,
  DatasetName,
  DatasetsResponse,
  ExplainResponse,
  FoveaConfig,
  FrameDiagnostics,
  MapMethod,
  MapResponse,
  PointsResponse,
  SemanticClass,
  StatusResponse,
} from "./types";

// Never hard-code localhost -- always resolve through the env var, with a
// same-origin-relative fallback only if the build somehow omits it.
const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { signal });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore parse failure */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

// All per-dataset calls go through /api/datasets/{dataset}/... -- the
// dataset-aware route family (src/api.py). The legacy no-dataset routes
// still exist on the backend for external/backward-compatible callers, but
// this frontend always names its dataset explicitly.
const enc = (s: string) => encodeURIComponent(s);

export const api = {
  baseUrl: API_URL,

  getStatus: (signal?: AbortSignal) => request<StatusResponse>("/api/status", signal),

  getDatasets: (signal?: AbortSignal) => request<DatasetsResponse>("/api/datasets", signal),

  getSequences: (dataset: DatasetName, signal?: AbortSignal) =>
    request<{ dataset: DatasetName; sequences: string[] }>(`/api/datasets/${enc(dataset)}/sequences`, signal),

  getFrames: (dataset: DatasetName, sequence: string, signal?: AbortSignal) =>
    request<{ dataset: DatasetName; sequence: string; frame_count: number; frame_ids: string[] }>(
      `/api/datasets/${enc(dataset)}/sequences/${enc(sequence)}/frames`,
      signal
    ),

  getDiagnostics: (dataset: DatasetName, sequence: string, frameId: string, signal?: AbortSignal) =>
    request<FrameDiagnostics>(
      `/api/datasets/${enc(dataset)}/frames/${enc(sequence)}/${enc(frameId)}/diagnostics`,
      signal
    ),

  getPoints: (dataset: DatasetName, sequence: string, frameId: string, max: number, signal?: AbortSignal) =>
    request<PointsResponse>(
      `/api/datasets/${enc(dataset)}/frames/${enc(sequence)}/${enc(frameId)}/points?max=${max}`,
      signal
    ),

  getMap: (
    dataset: DatasetName,
    sequence: string,
    frameId: string,
    method: MapMethod,
    opts: { includeCells?: boolean } = {},
    signal?: AbortSignal
  ) =>
    request<MapResponse>(
      `/api/datasets/${enc(dataset)}/frames/${enc(sequence)}/${enc(frameId)}/map/${method}?include_cells=${opts.includeCells ?? true}`,
      signal
    ),

  explainCell: (
    dataset: DatasetName,
    sequence: string,
    frameId: string,
    index: number,
    method: MapMethod = "fovea",
    signal?: AbortSignal
  ) =>
    request<ExplainResponse>(
      `/api/datasets/${enc(dataset)}/frames/${enc(sequence)}/${enc(frameId)}/cell/${index}/explain?method=${method}`,
      signal
    ),

  getSemanticClasses: (dataset: DatasetName, signal?: AbortSignal) =>
    request<{ dataset: DatasetName; classes: SemanticClass[] }>(
      `/api/datasets/${enc(dataset)}/semantic-classes`,
      signal
    ),

  getConfig: (signal?: AbortSignal) => request<FoveaConfig>("/api/config", signal),

  getBenchmark: (dataset: DatasetName, signal?: AbortSignal) =>
    request<BenchmarkResponse>(`/api/datasets/${enc(dataset)}/benchmark`, signal),
};

export { ApiError };
