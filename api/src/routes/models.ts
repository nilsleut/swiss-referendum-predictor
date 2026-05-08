import type { FastifyInstance } from "fastify";
import { listModels } from "../services/db";
import type { ModelRecord } from "../types/referendum";

export async function modelsRoutes(app: FastifyInstance): Promise<void> {
  app.get<{ Reply: ModelRecord[] }>("/models", async (_req, reply) => {
    reply.send(await listModels());
  });
}
