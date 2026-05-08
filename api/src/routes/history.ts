import type { FastifyInstance } from "fastify";
import { getPredictionHistory } from "../services/db";
import type { HistoryEntry } from "../types/referendum";

const querySchema = {
  type: "object",
  properties: {
    limit: { type: "integer", minimum: 1, maximum: 500, default: 50 },
  },
} as const;

export async function historyRoutes(app: FastifyInstance): Promise<void> {
  app.get<{ Querystring: { limit?: number }; Reply: HistoryEntry[] }>(
    "/history",
    { schema: { querystring: querySchema } },
    async (req, reply) => {
      const limit = req.query.limit ?? 50;
      const rows = await getPredictionHistory(limit);
      reply.send(rows);
    }
  );
}
