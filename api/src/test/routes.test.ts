import { describe, it, expect, vi, beforeEach } from "vitest";
import Fastify from "fastify";
import cors from "@fastify/cors";

// ── Mock heavy dependencies before importing routes ──────────────────────────

vi.mock("../services/db", () => ({
  checkConnection:      vi.fn().mockResolvedValue(true),
  createReferendum:     vi.fn().mockResolvedValue(1),
  storePrediction:      vi.fn().mockResolvedValue({ id: 1, prediction_time: new Date("2026-01-01") }),
  getPredictionHistory: vi.fn().mockResolvedValue([]),
  listModels:           vi.fn().mockResolvedValue([]),
  upsertModel:          vi.fn().mockResolvedValue(undefined),
  closePool:            vi.fn().mockResolvedValue(undefined),
  getPool:              vi.fn(),
}));

vi.mock("../services/model", () => ({
  modelService: {
    isLoaded:    vi.fn().mockReturnValue(true),
    nFeatures:   584,
    buildVector: vi.fn().mockReturnValue(new Array(584).fill(0)),
    predict:     vi.fn().mockResolvedValue({
      predicted_turnout:    42.0,
      confidence_interval:  [30.0, 54.0] as [number, number],
    }),
    load: vi.fn().mockResolvedValue(undefined),
  },
}));

// ── Import routes after mocks ────────────────────────────────────────────────

import { healthRoutes }  from "../routes/health";
import { predictRoutes } from "../routes/predict";
import { historyRoutes } from "../routes/history";
import { modelsRoutes }  from "../routes/models";

async function buildApp() {
  const app = Fastify({ logger: false });
  await app.register(cors, { origin: true });
  await app.register(healthRoutes);
  await app.register(predictRoutes);
  await app.register(historyRoutes);
  await app.register(modelsRoutes);
  return app;
}

// ── /health ──────────────────────────────────────────────────────────────────

describe("GET /health", () => {
  it("returns 200 with status ok when db and model are up", async () => {
    const app = await buildApp();
    const res = await app.inject({ method: "GET", url: "/health" });
    expect(res.statusCode).toBe(200);
    const body = res.json<{ status: string; db: boolean; model: boolean }>();
    expect(body.status).toBe("ok");
    expect(body.db).toBe(true);
    expect(body.model).toBe(true);
  });
});

// ── /predict ─────────────────────────────────────────────────────────────────

describe("POST /predict", () => {
  it("returns 400 when title is missing", async () => {
    const app = await buildApp();
    const res = await app.inject({
      method: "POST",
      url: "/predict",
      payload: { named_features: {} },
    });
    expect(res.statusCode).toBe(400);
  });

  it("returns 400 when named_features is missing", async () => {
    const app = await buildApp();
    const res = await app.inject({
      method: "POST",
      url: "/predict",
      payload: { title: "Test" },
    });
    expect(res.statusCode).toBe(400);
  });

  it("returns 201 with predicted_turnout on valid payload", async () => {
    const app = await buildApp();
    const res = await app.inject({
      method: "POST",
      url: "/predict",
      payload: { title: "Energiegesetz", named_features: { level: "national", institution: "Mandatory referendum" } },
    });
    expect(res.statusCode).toBe(201);
    const body = res.json<{ predicted_turnout: number; confidence_interval: number[] }>();
    expect(body.predicted_turnout).toBe(42.0);
    expect(body.confidence_interval).toHaveLength(2);
  });

  it("returns 400 when named_features is not an object", async () => {
    const app = await buildApp();
    const res = await app.inject({
      method: "POST",
      url: "/predict",
      payload: { title: "Test", named_features: "not-an-object" },
    });
    expect(res.statusCode).toBe(400);
  });
});

// ── /history ─────────────────────────────────────────────────────────────────

describe("GET /history", () => {
  it("returns 200 with an array", async () => {
    const app = await buildApp();
    const res = await app.inject({ method: "GET", url: "/history" });
    expect(res.statusCode).toBe(200);
    expect(Array.isArray(res.json())).toBe(true);
  });

  it("accepts a limit query param", async () => {
    const app = await buildApp();
    const res = await app.inject({ method: "GET", url: "/history?limit=10" });
    expect(res.statusCode).toBe(200);
  });
});

// ── /models ──────────────────────────────────────────────────────────────────

describe("GET /models", () => {
  it("returns 200 with an array", async () => {
    const app = await buildApp();
    const res = await app.inject({ method: "GET", url: "/models" });
    expect(res.statusCode).toBe(200);
    expect(Array.isArray(res.json())).toBe(true);
  });
});
