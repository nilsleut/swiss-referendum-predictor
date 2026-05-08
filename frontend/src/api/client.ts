const BASE = import.meta.env["VITE_API_BASE"] ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Types (mirror api/src/types/referendum.ts)
// ---------------------------------------------------------------------------

export interface PredictRequest {
  title: string;
  topic_category?: string;
  date?: string;
  features: number[];
}

export interface PredictResponse {
  prediction_id: number;
  referendum_id: number;
  predicted_turnout: number;
  confidence_interval: [number, number];
  model_version: string;
  prediction_time: string;
}

export interface HistoryEntry {
  prediction_id: number;
  referendum_id: number | null;
  title: string | null;
  topic_category: string | null;
  turnout_predicted: number;
  turnout_actual: number | null;
  confidence_low: number | null;
  confidence_high: number | null;
  model_version: string;
  prediction_time: string;
}

export interface ModelRecord {
  id: number;
  version: string;
  metrics: Record<string, number>;
  hyperparameters: Record<string, unknown>;
  onnx_path: string;
  trained_at: string;
}

export interface HealthStatus {
  status: "ok" | "degraded";
  db: boolean;
  model: boolean;
  model_version: string;
  uptime_seconds: number;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const api = {
  predict: (body: PredictRequest) =>
    request<PredictResponse>("/predict", { method: "POST", body: JSON.stringify(body) }),

  history: (limit = 50) =>
    request<HistoryEntry[]>(`/history?limit=${limit}`),

  models: () =>
    request<ModelRecord[]>("/models"),

  health: () =>
    request<HealthStatus>("/health"),
};

// ---------------------------------------------------------------------------
// Sample feature vector (all zeros → model predicts ~mean turnout)
// ---------------------------------------------------------------------------
export const N_FEATURES = 584;

export function sampleFeatures(): number[] {
  const v = new Array<number>(N_FEATURES).fill(0);
  v[1] = 1;      // f1 is almost always 1 in the training data
  v[2] = 6190;   // f2: mean integer value
  v[3] = 228;    // f3: mean integer value
  return v;
}
