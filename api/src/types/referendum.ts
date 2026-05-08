export interface PredictRequest {
  title: string;
  topic_category?: string;
  date?: string;           // ISO date string, e.g. "2024-03-15"
  features: number[];      // 584 raw (unscaled) feature values
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
