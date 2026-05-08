import { Pool } from "pg";
import { config } from "../config";
import type { HistoryEntry, ModelRecord } from "../types/referendum";

let pool: Pool;

export function getPool(): Pool {
  if (!pool) {
    pool = new Pool({ connectionString: config.databaseUrl });
  }
  return pool;
}

export async function checkConnection(): Promise<boolean> {
  try {
    await getPool().query("SELECT 1");
    return true;
  } catch {
    return false;
  }
}

export async function closePool(): Promise<void> {
  if (pool) await pool.end();
}

// ---------------------------------------------------------------------------
// Referendums
// ---------------------------------------------------------------------------

export async function createReferendum(data: {
  title: string;
  topic_category?: string;
  date?: string;
}): Promise<number> {
  const result = await getPool().query<{ id: number }>(
    `INSERT INTO referendums (title, topic_category, date)
     VALUES ($1, $2, $3)
     RETURNING id`,
    [data.title, data.topic_category ?? null, data.date ?? null]
  );
  return result.rows[0].id;
}

// ---------------------------------------------------------------------------
// Predictions
// ---------------------------------------------------------------------------

export async function storePrediction(data: {
  referendum_id: number;
  model_version: string;
  turnout_predicted: number;
  confidence_low: number;
  confidence_high: number;
  features: number[];
}): Promise<{ id: number; prediction_time: Date }> {
  const result = await getPool().query<{ id: number; prediction_time: Date }>(
    `INSERT INTO predictions
       (referendum_id, model_version, turnout_predicted, confidence_low, confidence_high, features)
     VALUES ($1, $2, $3, $4, $5, $6)
     RETURNING id, prediction_time`,
    [
      data.referendum_id,
      data.model_version,
      data.turnout_predicted,
      data.confidence_low,
      data.confidence_high,
      JSON.stringify(data.features),
    ]
  );
  return result.rows[0];
}

export async function getPredictionHistory(limit = 50): Promise<HistoryEntry[]> {
  const result = await getPool().query<HistoryEntry>(
    `SELECT
       p.id          AS prediction_id,
       p.referendum_id,
       r.title,
       r.topic_category,
       p.turnout_predicted,
       r.turnout_actual,
       p.confidence_low,
       p.confidence_high,
       p.model_version,
       p.prediction_time
     FROM predictions p
     LEFT JOIN referendums r ON r.id = p.referendum_id
     ORDER BY p.prediction_time DESC
     LIMIT $1`,
    [limit]
  );
  return result.rows;
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export async function upsertModel(data: {
  version: string;
  metrics: Record<string, number>;
  hyperparameters: Record<string, unknown>;
  onnx_path: string;
}): Promise<void> {
  await getPool().query(
    `INSERT INTO models (version, metrics, hyperparameters, onnx_path)
     VALUES ($1, $2, $3, $4)
     ON CONFLICT (version) DO UPDATE
       SET metrics = EXCLUDED.metrics,
           hyperparameters = EXCLUDED.hyperparameters,
           onnx_path = EXCLUDED.onnx_path`,
    [
      data.version,
      JSON.stringify(data.metrics),
      JSON.stringify(data.hyperparameters),
      data.onnx_path,
    ]
  );
}

export async function listModels(): Promise<ModelRecord[]> {
  const result = await getPool().query<ModelRecord>(
    `SELECT id, version, metrics, hyperparameters, onnx_path, trained_at
     FROM models
     ORDER BY trained_at DESC`
  );
  return result.rows;
}
