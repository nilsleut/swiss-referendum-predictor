-- Swiss Referendum Predictor — database schema
-- Run once on a fresh PostgreSQL instance (handled automatically by docker-compose).

CREATE TABLE IF NOT EXISTS referendums (
    id              SERIAL PRIMARY KEY,
    date            DATE,
    title           TEXT NOT NULL,
    topic_category  VARCHAR(100),
    turnout_actual  FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id                SERIAL PRIMARY KEY,
    referendum_id     INTEGER REFERENCES referendums(id) ON DELETE SET NULL,
    model_version     VARCHAR(50) NOT NULL,
    turnout_predicted FLOAT NOT NULL,
    confidence_low    FLOAT,
    confidence_high   FLOAT,
    features          JSONB NOT NULL,
    prediction_time   TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS models (
    id               SERIAL PRIMARY KEY,
    version          VARCHAR(50) UNIQUE NOT NULL,
    metrics          JSONB NOT NULL,
    hyperparameters  JSONB NOT NULL,
    onnx_path        TEXT NOT NULL,
    trained_at       TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_referendum ON predictions(referendum_id);
CREATE INDEX IF NOT EXISTS idx_predictions_time      ON predictions(prediction_time DESC);
