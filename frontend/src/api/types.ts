/**
 * TypeScript mirrors of the Pydantic response models in src/api.py.
 * Kept 1:1 with the backend on purpose -- this file declares shapes only,
 * it never computes FOVEA values itself.
 */

export type MapMethod = "uniform" | "distance_adaptive" | "fovea";
export type DatasetName = "SemanticPOSS" | "RELLIS-3D";
export const DATASET_NAMES: DatasetName[] = ["SemanticPOSS", "RELLIS-3D"];

export interface Cell {
  index: number;
  cell_size_m: number;
  resolution_level: number;
  resolution_label: string;
  band_id: number;
  ix: number;
  iy: number;
  center_x: number;
  center_y: number;
  point_count: number;
  elevation_median: number;
  elevation_std: number;
  majority_class_id: number;
  majority_class_name: string;
  majority_class_fraction: number;
  semantic_priority: number;
  geometry_score: number;
  dynamic_score: number;
  uncertainty_score: number | null; // always null in v0.1 -- never fabricated
  information_score: number;
  range_m: number;
  weights_used: Record<string, number>;
}

export interface MapResponse {
  dataset: DatasetName;
  sequence: string;
  frame_id: string;
  method: MapMethod;
  cells: Cell[];
  accounting: Record<string, unknown>;
  summary: MapSummary;
}

export interface MapSummary {
  method: string;
  output_cell_count: number;
  cells_by_resolution_level: Record<string, number>;
  distinct_cell_sizes_m: number[];
  input_points: number;
  assigned_points: number;
  assignment_rate: number;
  out_of_envelope_points: number;
  invalid_points: number;
  map_generation_latency_s: number;
  estimated_map_storage_bytes: number;
  estimated_map_storage_kb: number;
  semantic_classes_covered: number;
  distance_band_stats: Record<string, {
    cell_count: number;
    point_count: number;
    semantic_classes_present: Record<string, number>;
  }>;
  timing_repeats?: {
    repeats: number;
    mean_latency_s: number;
    std_latency_s: number;
    min_latency_s: number;
    max_latency_s: number;
    output_cell_count: number;
  };
}

export interface PointsResponse {
  dataset: DatasetName;
  sequence: string;
  frame_id: string;
  total_points_in_frame: number;
  returned_points: number;
  subsampled: boolean;
  x: number[];
  y: number[];
  z: number[];
  intensity: number[];
  semantic_id: number[];
}

export interface FrameDiagnostics {
  dataset: DatasetName;
  sequence: string;
  frame_id: string;
  source_bin: string;
  source_label: string | null;
  has_labels: boolean;
  num_points: number;
  attributes: string[];
  x_range: [number, number];
  y_range: [number, number];
  z_range: [number, number];
  intensity_range: [number, number];
  range_stats_m: { min: number; max: number; mean: number; p99: number };
  num_unique_semantic_ids: number;
  class_counts: Record<string, { name: string; count: number }>;
}

export interface StatusResponse {
  status: string;
  dataset_root: string;
  dataset_root_exists: boolean;
  available_sequences: string[];
  available_datasets: DatasetName[];
  default_dataset: DatasetName;
  environment: Record<string, unknown>;
  cached_frames: string[];
}

export interface DatasetInfo {
  name: DatasetName;
  is_default: boolean;
}

export interface DatasetsResponse {
  datasets: DatasetInfo[];
  default: DatasetName;
}

export interface ExplainResponse {
  dataset: DatasetName;
  sequence: string;
  frame_id: string;
  method: string;
  cell_index: number;
  cell: Cell;
  explanation: string;
}

export interface SemanticClass {
  class_id: number;
  class_name: string;
  semantic_priority: number;
  is_dynamic_relevant: boolean;
}

export interface BenchmarkFrameMethodSummary extends MapSummary {}

export interface BenchmarkPerFrameResult {
  frame_id: string;
  num_points: number;
  methods: Record<MapMethod, BenchmarkFrameMethodSummary>;
}

export interface BenchmarkData {
  timestamp: string;
  dataset?: DatasetName;
  sequence: string;
  num_frames_processed: number;
  environment: Record<string, unknown>;
  sequence_processing: {
    total_wall_time_s: number;
    note: string;
    approx_throughput_frames_per_s: number | null;
  };
  per_frame_results: BenchmarkPerFrameResult[];
}

export interface BenchmarkResponse {
  dataset: DatasetName;
  available: boolean;
  source: string;
  data: BenchmarkData | null;
  message: string | null;
}

export interface FoveaConfig {
  data: { root: string; sequence: string; frame: number };
  distance_adaptive: { bands: { max_m: number; cell_size_m: number }[] };
  uniform: { cell_size_m: number };
  fovea: {
    refinement_factor: number;
    split_to_medium_threshold: number;
    split_to_fine_threshold: number;
    weights: { semantic: number; geometry: number; dynamic: number; uncertainty: number };
    geometry: { max_expected_std_m: number };
    dynamic_classes: string[];
    semantic_priority: Record<string, number>;
  };
  visualization: { point_subsample_max: number };
  [key: string]: unknown;
}
