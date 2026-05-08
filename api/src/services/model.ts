import fs from "fs";
import * as ort from "onnxruntime-node";
import { config } from "../config";

interface ScalerParams {
  mean: number[];
  scale: number[];
  n_features: number;
}

interface InferenceResult {
  predicted_turnout: number;
  confidence_interval: [number, number];
}

class ModelService {
  private session: ort.InferenceSession | null = null;
  private scaler: ScalerParams | null = null;
  private rmse: number = 12.06; // fallback; overwritten from test_metrics.json

  async load(): Promise<void> {
    // Load scaler params
    const scalerRaw = fs.readFileSync(config.scalerParamsPath, "utf-8");
    this.scaler = JSON.parse(scalerRaw) as ScalerParams;

    // Load RMSE from test metrics for confidence intervals
    if (fs.existsSync(config.testMetricsPath)) {
      const metricsRaw = fs.readFileSync(config.testMetricsPath, "utf-8");
      const metrics = JSON.parse(metricsRaw) as { rmse?: number };
      if (metrics.rmse) this.rmse = metrics.rmse;
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

// Singleton — shared across the Fastify instance lifetime
export const modelService = new ModelService();
