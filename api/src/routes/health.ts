import type { FastifyInstance } from "fastify";
import { checkConnection } from "../services/db";
import { modelService } from "../services/model";
import { config } from "../config";
import type { HealthStatus } from "../types/referendum";

const startTime = Date.now();

export async function healthRoutes(app: FastifyInstance): Promise<void> {
  app.get<{ Reply: HealthStatus }>("/health", async (_req, reply) => {
    const [db, model] = await Promise.all([
      checkConnection(),
      Promise.resolve(modelService.isLoaded()),
    ]);

    const status: HealthStatus = {
      status: db && model ? "ok" : "degraded",
      db,
      model,
      model_version: config.modelVersion,
      uptime_seconds: Math.floor((Date.now() - startTime) / 1000),
    };

    reply.status(status.status === "ok" ? 200 : 503).send(status);
  });
}
