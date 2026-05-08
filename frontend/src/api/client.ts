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
// Feature group options (extracted from training data column names)
// ---------------------------------------------------------------------------

export const CANTONS = [
  "Aargau", "Appenzell Ausserrhoden", "Basel Landschaft", "Basel Stadt",
  "Basel-Landschaft", "Bern", "Berne", "Fribourg", "Geneva", "Graubünden",
  "Jura", "Lucerne", "Luzern", "Neuchâtel", "Neuenburg", "Nidwalden",
  "Obwalden", "Schaffhausen", "Schwyz", "Solothurn", "St. Gallen", "Thurgau",
  "Ticino", "Uri", "Valais", "Vaud", "Zug", "Zurich",
];

export const THEMES = [
  "EEA", "EU", "Swiss abroad", "UN", "abolition of the armed forces",
  "addictive substances", "agricultural policy", "agriculture",
  "air quality control", "air transport", "alternative energy",
  "animal protection", "animal testing", "armed forces in general", "arms",
  "banks, stock exchange, insurance", "bilateral treaties",
  "budget cuts and remediation measures",
  "building of residential housing, property ownership",
  "children and young people", "citizens' initiatives", "citizens' rights",
  "citizenship", "civil protection", "competition policy",
  "conscientious objection, civilian service", "constitution",
  "consumer protection", "courts", "criminal law", "crop production",
  "cultural policy", "culture, religion and media", "customs",
  "data protection", "development cooperation", "direct taxation",
  "disability insurance", "division of tasks", "economic policy", "economy",
  "education and research", "education policy", "electoral system",
  "employment", "employment conditions", "employment policy", "energy",
  "energy policy", "environment", "environment and living space",
  "environmental policy", "family policy", "federalism", "finance",
  "financial system", "fishing, hunting and pets", "foreign trade policy",
  "forestry", "fundamental rights", "genetic engineering", "goods traffic",
  "government, administration", "health", "health and accident insurance",
  "health policy", "heavy traffic", "homosexuals", "hospitality", "housing",
  "hydro-electric power", "immigration policy", "independence",
  "indirect taxation", "industrial relations", "institutions",
  "intergovernmental relations", "international organisations", "land law",
  "language policy", "legal system", "legislative procedure", "livestock",
  "lottery and gambling", "maternity insurance", "media and communication",
  "medical research and technology", "medicines", "military facilities",
  "military organisation", "military training", "monetary policy",
  "national bank", "national economic supply", "national identity",
  "neutrality", "noise protection", "nuclear energy", "oil and gas",
  "parliament", "passenger traffic", "pension insurance",
  "persons with disabilities", "police", "political system",
  "position on foreign policy", "post", "pricing policy", "private law",
  "procedure for constitutional reform", "professional and vocational education",
  "protection of nature and cultural heritage", "public expenditure",
  "public finance", "public security", "radio, television and electronic media",
  "rail transport", "referendum", "refugees", "religion, churches",
  "reproductive medicine", "research", "road construction", "road transport",
  "schools", "security policy", "senior citizens", "shipping",
  "short-term economic policy", "social groups", "social policy",
  "social security", "soil", "soil protection", "spatial planning", "sport",
  "state organisation", "state security", "status of women",
  "structural policy", "tax policy", "tax system", "taxation",
  "telecommunications", "tenancy issues", "territorial questions", "tourism",
  "tourism and leisure", "transit traffic", "transport and infrastructure",
  "transport policy", "unemployment insurance", "universities", "urban transport",
  "voting rights", "waste", "water pollution control", "welfare", "working hours",
];

export const INSTITUTIONS = [
  "Citizens' initiative", "Counter proposal", "Governmental referendum",
  "Mandatory referendum", "Not provided", "Optional referendum",
];

export const VOTE_TRIGGERS = ["Automatic", "Bottom up", "Top down"];

export const VOTE_TRIGGER_ACTORS = [
  "Citizens", "Constitution", "Government", "Parliament", "Territorial unit",
];

export const VOTE_OBJECTS = [
  "Legal text (allg. anregung)", "Legal text (ausformulierter vorschlag)",
  "Principle", "Question",
];

export const AUTHORS = [
  "Citizens", "Government", "Institution", "Monarch", "Parliament", "Territorial unit",
];

export const ACTIONS = [
  "Abrogate", "Abrogate, Introduce", "Abrogate, Introduce, Revise",
  "Introduce", "Introduce, Revise", "Revise",
];

export const LEGAL_NORM_HIERARCHIES = [
  "Constitution", "Constitution / law / ordinance", "Constitution / law / other",
  "International treaty", "Law", "Ordinance", "Treaty", "Undefined",
];

export const DEGREES_OF_REVISION = ["Both", "Partial", "Total"];

export const DECISION_QUORUMS = ["Double majority", "Simple majority"];

export const LEGAL_ACT_TYPES = ["Normal", "Urgent"];

export const SPECIAL_TOPICS = [
  "Financial act", "Financial act (expenses)",
  "Financial act (expenses), Treaties", "Financial act (obligations)",
  "Financial act (taxes)", "Financial act, Financial act (expenses)",
  "Financial act, Infrastructural act", "Infrastructural act",
  "Territorial questions", "Territorial questions, Treaties",
  "Total revision of the constitution", "Treaties",
];
