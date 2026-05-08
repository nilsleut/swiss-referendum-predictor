export interface NamedFeatureInput {
  level?: "national" | "cantonal";
  canton?: string;
  theme1?: string;
  theme2?: string;
  theme3?: string;
  institution?: string;
  vote_trigger?: string;
  vote_result_status?: string;
  official_status?: string;
  legal_act_type?: string;
  vote_trigger_actor?: string;
  vote_object?: string;
  author?: string;
  counter_proposal?: string;
  action?: string;
  legal_norm_hierarchy?: string;
  degree_of_revision?: string;
  decision_quorum?: string;
  referendum_text_options?: string;
  institutional_precondition?: string;
  institutional_precondition_decision?: string;
  special_topics?: string;
  excluded_topics?: string;
}

export interface PredictRequest {
  title: string;
  topic_category?: string;
  date?: string;
  named_features: NamedFeatureInput;
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
