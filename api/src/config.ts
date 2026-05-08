import "dotenv/config";
import path from "path";

function require_env(key: string): string {
  const val = process.env[key];
  if (!val) throw new Error(`Missing required env var: ${key}`);
  return val;
}

export const config = {
  port: parseInt(process.env["PORT"] ?? "3000", 10),
  logPretty: process.env["LOG_PRETTY"] === "true",
  databaseUrl: require_env("DATABASE_URL"),
  onnxModelDir: require_env("ONNX_MODEL_DIR"),
  modelVersion: process.env["MODEL_VERSION"] ?? "v1.0.0",

  get onnxModelPath() {
    return path.join(this.onnxModelDir, "model.onnx");
  },
  get scalerParamsPath() {
    return path.join(this.onnxModelDir, "scaler_params.json");
  },
  get testMetricsPath() {
    return path.join(this.onnxModelDir, "test_metrics.json");
  },
  get modelInfoPath() {
    return path.join(this.onnxModelDir, "model_info.json");
  },
  get featureGroupsPath() {
    return path.join(this.onnxModelDir, "feature_groups.json");
  },
};
