// Mirrors backend/app/schemas/*.py - keep in sync by hand (no shared codegen
// yet; the FastAPI OpenAPI schema at /openapi.json is the source of truth
// if these ever drift).

export type PredictionStatus = "pending" | "processing" | "completed" | "failed";
export type BatchStatus = "pending" | "processing" | "completed" | "failed";

export interface Detection {
  id: string;
  class_name: string;
  confidence: number;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface Prediction {
  id: string;
  status: PredictionStatus;
  original_filename: string;
  annotated_image_url: string | null;
  inference_time_ms: number | null;
  model_version: string | null;
  error_message: string | null;
  created_at: string;
  detections: Detection[];
}

export interface PredictionListItem {
  id: string;
  status: PredictionStatus;
  original_filename: string;
  inference_time_ms: number | null;
  model_version: string | null;
  created_at: string;
  num_detections: number;
  top_class: string | null;
  top_confidence: number | null;
}

export interface PredictionListResponse {
  items: PredictionListItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface BatchSubmitResponse {
  batch_id: string;
  total_images: number;
  status: BatchStatus;
}

export interface BatchStatusResponse {
  batch_id: string;
  status: BatchStatus;
  total_images: number;
  completed_images: number;
  failed_images: number;
  created_at: string;
  completed_at: string | null;
}

export interface ModelInfo {
  version: string;
  dataset_version: string;
  trained_at: string;
  hardware: string;
  val_metrics: Record<string, unknown> & { mAP50: number; mAP50_95: number; precision: number; recall: number };
  test_metrics:
    | (Record<string, unknown> & { mAP50: number; mAP50_95: number; precision: number; recall: number })
    | null;
  config: Record<string, unknown>;
  classes: string[];
}

export interface ConfidenceBucket {
  range_label: string;
  count: number;
}

export interface BatchStats {
  total_batches: number;
  pending: number;
  processing: number;
  completed: number;
  failed: number;
  total_images_via_batch: number;
}

export interface AnalyticsSummary {
  total_predictions: number;
  completed_predictions: number;
  failed_predictions: number;
  images_with_defects: number;
  defect_rate: number;
  defects_by_category: Record<string, number>;
  confidence_distribution: ConfidenceBucket[];
  recent_predictions: PredictionListItem[];
  batch_stats: BatchStats;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  checks: Record<string, string>;
}
