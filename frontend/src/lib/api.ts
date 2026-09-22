import type {
  AnalyticsSummary,
  BatchStatusResponse,
  BatchSubmitResponse,
  HealthResponse,
  ModelInfo,
  Prediction,
  PredictionListResponse,
  PredictionStatus,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON - fall back to statusText
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export function resolveImageUrl(path: string | null): string | null {
  if (!path) return null;
  return `${API_BASE_URL}${path}`;
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  model: () => request<ModelInfo>("/model"),

  predictSingle: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Prediction>("/predict", { method: "POST", body: form });
  },

  submitBatch: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request<BatchSubmitResponse>("/predict/batch", { method: "POST", body: form });
  },

  getBatchStatus: (batchId: string) => request<BatchStatusResponse>(`/batch/${batchId}`),

  getPrediction: (id: string) => request<Prediction>(`/predictions/${id}`),

  listPredictions: (
    params: { limit?: number; offset?: number; status?: PredictionStatus; batchId?: string } = {},
  ) => {
    const query = new URLSearchParams();
    if (params.limit) query.set("limit", String(params.limit));
    if (params.offset) query.set("offset", String(params.offset));
    if (params.status) query.set("status", params.status);
    if (params.batchId) query.set("batch_id", params.batchId);
    const qs = query.toString();
    return request<PredictionListResponse>(`/predictions${qs ? `?${qs}` : ""}`);
  },

  analyticsSummary: () => request<AnalyticsSummary>("/analytics/summary"),
};
