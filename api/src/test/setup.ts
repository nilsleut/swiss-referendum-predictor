// Set required env vars before any module under test is imported.
process.env["DATABASE_URL"]   = "postgresql://test:test@localhost:5432/test";
process.env["ONNX_MODEL_DIR"] = "/tmp/models";
process.env["MODEL_VERSION"]  = "v0.0.0";
process.env["LOG_PRETTY"]     = "false";
