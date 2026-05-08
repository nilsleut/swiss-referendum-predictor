import Fastify from "fastify";
import cors from "@fastify/cors";
import fs from "fs";
import { config } from "./config";
import { healthRoutes } from "./routes/health";
import { predictRoutes } from "./routes/predict";
import { historyRoutes } from "./routes/history";
import { modelsRoutes } from "./routes/models";
import { modelService } from "./services/model";
import { upsertModel, closePool } from "./services/db";

async function build() {
  const app = Fastify({
    logger: config.logPretty
      ? { transport: { target: "pino-pretty" } }
      : true,
  });

  await app.register(cors, { origin: true });

  // Routes
  await app.register(healthRoutes);
  await app.register(predictRoutes);
  await app.register(historyRoutes);
  await app.register(modelsRoutes);

  return app;
}

async function start() {
  const app = await build();

  // Load ONNX model before accepting traffic
  app.log.info(`Loading ONNX model from ${config.onnxModelPath}`);
  await modelService.load();
  app.log.info("Model loaded successfully");

  // Seed the models table with this version's metadata
  if (fs.existsSync(config.modelInfoPath)) {
    const info = JSON.parse(fs.readFileSync(config.modelInfoPath, "utf-8"));
    const metrics = fs.existsSync(config.testMetricsPath)
      ? JSON.parse(fs.readFileSync(config.testMetricsPath, "utf-8"))
      : {};
    await upsertModel({
      version: config.modelVersion,
      metrics,
      hyperparameters: info,
      onnx_path: config.onnxModelPath,
    });
    app.log.info(`Model ${config.modelVersion} registered in DB`);
  }

  await app.listen({ port: config.port, host: "0.0.0.0" });

  // Graceful shutdown
  for (const signal of ["SIGINT", "SIGTERM"] as const) {
    process.on(signal, async () => {
      app.log.info(`Received ${signal}, shutting down`);
      await app.close();
      await closePool();
      process.exit(0);
    });
  }
}

start().catch((err) => {
  console.error(err);
  process.exit(1);
});
