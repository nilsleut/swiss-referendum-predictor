import type { FastifyInstance } from "fastify";
import { createReferendum, storePrediction } from "../services/db";
import { modelService } from "../services/model";
import { config } from "../config";
import type { PredictRequest, PredictResponse } from "../types/referendum";

const bodySchema = {
  type: "object",
  required: ["title", "features"],
  properties: {
    title:          { type: "string", minLength: 1 },
    topic_category: { type: "string" },
    date:           { type: "string", format: "date" },
    features: {
      type: "array",
      items: { type: "number" },
      minItems: 1,
    },
  },
} as const;

export async function predictRoutes(app: FastifyInstance): Promise<void> {
  app.post<{ Body: PredictRequest; Reply: PredictResponse }>(
    "/predict",
    { schema: { body: bodySchema } },
    async (req, reply) => {
      const { title, topic_category, date, features } = req.body;

      const n = modelService.nFeatures;
      if (features.length !== n) {
        return reply.status(400).send({
          // Fastify will serialise this as an error body
          message: `features array must have exactly ${n} elements, got ${features.length}`,
        } as unknown as PredictResponse);
      }

      // Inference
      const { predicted_turnout, confidence_interval } =
        await modelService.predict(features);

      // Persist
      const referendum_id = await createReferendum({ title, topic_category, date });
      const { id: prediction_id, prediction_time } = await storePrediction({
        referendum_id,
        model_version: config.modelVersion,
        turnout_predicted: predicted_turnout,
        confidence_low: confidence_interval[0],
        confidence_high: confidence_interval[1],
        features,
      });

      reply.status(201).send({
        prediction_id,
        referendum_id,
        predicted_turnout,
        confidence_interval,
        model_version: config.modelVersion,
        prediction_time: prediction_time.toISOString(),
      });
    }
  );
}
