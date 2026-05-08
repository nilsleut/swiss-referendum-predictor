import fs from "fs";
import * as ort from "onnxruntime-node";
import { config } from "../config";
import type { NamedFeatureInput } from "../types/referendum";

interface ScalerParams {
  mean: number[];
  scale: number[];
  n_features: number;
}

interface FeatureGroupItem {
  index: number;
  colName: string;
  label: string;
}

type FeatureGroups = Record<string, FeatureGroupItem[]>;

interface InferenceResult {
  predicted_turnout: number;
  confidence_interval: [number, number];
}

// Maps NamedFeatureInput keys to feature_groups.json keys
const GROUP_KEY_MAP: Record<string, string> = {
  canton:                           "Canton",
  theme1:                           "Theme 1",
  theme2:                           "Theme 2",
  theme3:                           "Theme 3",
  institution:                      "Institution",
  vote_trigger:                     "Vote trigger",
  vote_result_status:               "Vote Result status",
  official_status:                  "Official status",
  legal_act_type:                   "Legal act type",
  vote_trigger_actor:               "Vote trigger actor",
  vote_object:                      "Vote object",
  author:                           "Author of the vote object",
  counter_proposal:                 "Counter proposal",
  action:                           "Action",
  legal_norm_hierarchy:             "Hierarchy of the legal norm",
  degree_of_revision:               "Degree of revision",
  decision_quorum:                  "Decision quorum",
  referendum_text_options:          "Referendum text options",
  institutional_precondition:       "Institutional precondition",
  institutional_precondition_decision: "Institutional precondition decision",
  special_topics:                   "Special topics",
  excluded_topics:                  "Excluded topics",
};

class ModelService {
  private session: ort.InferenceSession | null = null;
  private scaler: ScalerParams | null = null;
  private featureGroups: FeatureGroups | null = null;
  private rmse: number = 12.06;

  async load(): Promise<void> {
    const scalerRaw = fs.readFileSync(config.scalerParamsPath, "utf-8");
    this.scaler = JSON.parse(scalerRaw) as ScalerParams;

    if (fs.existsSync(config.testMetricsPath)) {
      const metricsRaw = fs.readFileSync(config.testMetricsPath, "utf-8");
      const metrics = JSON.parse(metricsRaw) as { rmse?: number };
      if (metrics.rmse) this.rmse = metrics.rmse;
    }

    if (fs.existsSync(config.featureGroupsPath)) {
      const groupsRaw = fs.readFileSync(config.featureGroupsPath, "utf-8");
      this.featureGroups = JSON.parse(groupsRaw) as FeatureGroups;
    }

    this.session = await ort.InferenceSession.create(config.onnxModelPath, {
      executionProviders: ["cpu"],
    });
  }

  isLoaded(): boolean {
    return this.session !== null && this.scaler !== null;
  }

  get nFeatures(): number {
    return this.scaler?.n_features ?? 0;
  }

  // Build a full 584-element feature vector from partial named input.
  // Missing fields are imputed with training means (→ 0 after scaling = neutral).
  // When a group IS specified, the selected column is set to 1 and all other
  // columns in that group are zeroed out (valid one-hot encoding).
  buildVector(input: NamedFeatureInput): number[] {
    const vec = [...this.scaler!.mean];

    for (const [inputKey, groupKey] of Object.entries(GROUP_KEY_MAP)) {
      const selectedLabel = input[inputKey as keyof NamedFeatureInput];
      if (!selectedLabel || !this.featureGroups) continue;

      const group = this.featureGroups[groupKey];
      if (!group) continue;

      for (const item of group) vec[item.index] = 0;

      const selected = group.find((item) => item.label === selectedLabel);
      if (selected) vec[selected.index] = 1;
    }

    // National (col 0) and Kantonal (col 1) are plain binary columns
    if (input.level === "national") {
      vec[0] = 1;
      vec[1] = 0;
    } else if (input.level === "cantonal") {
      vec[0] = 0;
      vec[1] = 1;
    }

    return vec;
  }

  private scale(raw: number[]): Float32Array {
    const { mean, scale } = this.scaler!;
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      out[i] = (raw[i] - mean[i]) / scale[i];
    }
    return out;
  }

  async predict(rawFeatures: number[]): Promise<InferenceResult> {
    if (!this.session || !this.scaler) {
      throw new Error("Model not loaded — call load() first");
    }
    if (rawFeatures.length !== this.scaler.n_features) {
      throw new Error(
        `Expected ${this.scaler.n_features} features, got ${rawFeatures.length}`
      );
    }

    const scaled = this.scale(rawFeatures);
    const tensor = new ort.Tensor("float32", scaled, [1, rawFeatures.length]);
    const results = await this.session.run({ features: tensor });
    const turnout = (results["turnout"].data as Float32Array)[0];
    const rounded = Math.round(turnout * 100) / 100;

    return {
      predicted_turnout: rounded,
      confidence_interval: [
        Math.round((turnout - this.rmse) * 100) / 100,
        Math.round((turnout + this.rmse) * 100) / 100,
      ],
    };
  }
}

export const modelService = new ModelService();
