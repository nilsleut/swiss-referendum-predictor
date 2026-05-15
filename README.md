# Swiss Referendum Predictor

End-to-end ML system that predicts Swiss referendum voter turnout.
Built as a production-grade portfolio project covering the full ML lifecycle.

**Live demo:** [swiss-referendum-predictor-1.onrender.com](https://swiss-referendum-predictor-1.onrender.com/)
> First request may take ~30 s due to Render free-tier cold start.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  React + Recharts (frontend)       :80  / :5173 dev │
│  Vite · TypeScript · nginx                          │
└────────────────────┬────────────────────────────────┘
                     │ /api/*
┌────────────────────▼────────────────────────────────┐
│  Fastify API                       :3000             │
│  TypeScript · onnxruntime-node · pg                 │
└──────┬─────────────────────────┬────────────────────┘
       │                         │
┌──────▼──────┐       ┌──────────▼──────────┐
│  PostgreSQL │       │  ONNX model          │
│  Supabase   │       │  + scaler_params.json│
└─────────────┘       └──────────────────────┘
```

## Tech stack

| Layer | Technology |
|---|---|
| ML training | Python 3.11, PyTorch, wandb |
| Model serving | ONNX Runtime |
| API | TypeScript, Fastify 5, pg |
| Database | PostgreSQL 16 (Supabase) |
| Frontend | React 18, Recharts, Vite |
| Containers | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | Render.com + Supabase (free tier) |

## Dataset

Source: `grosserDatensatzEingabe.csv` / `grosserDatensatzAusgabe.csv`.
Place both files in `ml/data/` (gitignored — must be added manually).

- **Features**: 6 083 referendums × 584 columns (one-hot encoded, semicolon-delimited)
- **Target**: voter turnout percentage (5.3 – 90.0 %)

---

## Quick start — local dev (no Docker)

### Prerequisites

- Python 3.11, Node.js 22, PostgreSQL 16

### 1 — ML pipeline

```bash
cd ml
pip install -r requirements.txt

# Train (writes models/checkpoints/best_model.pt)
python -m src.train --config experiments/default.yaml

# Evaluate on test set
python -m src.evaluate --config experiments/default.yaml

# Export to ONNX + scaler JSON (writes models/onnx/)
python -m src.export
```

### 2 — API

```bash
cd api
npm install
# Edit .env — DATABASE_URL, ONNX_MODEL_DIR (absolute path to ml/models/onnx)
npm run dev          # http://localhost:3000
```

### 3 — Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173  (proxies /api → :3000)
```

---

## Docker Compose (recommended)

Copy and fill in the root env file:

```bash
cp .env.example .env
# set POSTGRES_PASSWORD, WANDB_API_KEY
```

```bash
# Start db + api + frontend
docker compose up --build

# Frontend  → http://localhost
# API       → http://localhost:3000
# Health    → http://localhost:3000/health
```

Re-train the model inside Docker:

```bash
docker compose --profile training run ml
docker compose --profile training run ml python -m src.export
docker compose restart api
```

---

## API endpoints

| Method | Route | Description |
|---|---|---|
| `POST` | `/predict` | Run inference, store result |
| `GET` | `/history?limit=50` | Past predictions |
| `GET` | `/models` | Registered model versions |
| `GET` | `/health` | DB + model liveness |

### POST /predict — request

Provide as many or as few fields as you like — the model fills in anything left unspecified using training-data patterns.

```json
{
  "title": "Energiegesetz",
  "date": "2024-06-09",
  "named_features": {
    "level": "national",
    "institution": "Mandatory referendum",
    "theme1": "energy policy",
    "counter_proposal": "No",
    "decision_quorum": "Simple majority"
  }
}
```

All `named_features` fields are optional. Available fields:

| Field | Example values |
|---|---|
| `level` | `"national"`, `"cantonal"` |
| `canton` | `"Zurich"`, `"Bern"`, `"Geneva"`, … |
| `theme1` / `theme2` / `theme3` | `"energy policy"`, `"immigration policy"`, … (159 options) |
| `institution` | `"Citizens' initiative"`, `"Mandatory referendum"`, `"Optional referendum"`, … |
| `counter_proposal` | `"Yes"`, `"No"` |
| `action` | `"Introduce"`, `"Revise"`, `"Abrogate"`, … |
| `decision_quorum` | `"Simple majority"`, `"Double majority"` |
| `legal_norm_hierarchy` | `"Constitution"`, `"Law"`, `"Ordinance"`, … |
| `degree_of_revision` | `"Partial"`, `"Total"`, `"Both"` |
| `legal_act_type` | `"Normal"`, `"Urgent"` |
| `vote_trigger` | `"Bottom up"`, `"Top down"`, `"Automatic"` |

### POST /predict — response

```json
{
  "prediction_id": 1,
  "referendum_id": 1,
  "predicted_turnout": 41.83,
  "confidence_interval": [29.77, 53.89],
  "model_version": "v1.0.0",
  "prediction_time": "2026-05-08T10:00:00.000Z"
}
```

---

## Deployment — Render + Supabase (free tier)

### 1 — Supabase database

1. Create a project at [supabase.com](https://supabase.com)
2. Run the schema in **SQL Editor**:  paste the contents of [`db/init.sql`](db/init.sql)
3. Copy the **Transaction pooler** connection string from **Connect → Transaction** (use the pooler URL — Render free tier requires IPv4)

### 2 — Render API service

New Web Service → connect repo:

| Setting | Value |
|---|---|
| Root directory | `api` |
| Runtime | Docker |
| Dockerfile path | `./Dockerfile` |
| Docker build context | `.` |

Environment variables:

| Key | Value |
|---|---|
| `DATABASE_URL` | Supabase transaction pooler URI |
| `ONNX_MODEL_DIR` | `/app/models` |
| `MODEL_VERSION` | `v1.0.0` |
| `PORT` | `3000` |
| `LOG_PRETTY` | `false` |

### 3 — Render Frontend service

New Web Service → same repo:

| Setting | Value |
|---|---|
| Root directory | `frontend` |
| Runtime | Docker |
| Dockerfile path | `./Dockerfile` |
| Docker build context | `.` |

Environment variables:

| Key | Scope | Value |
|---|---|---|
| `VITE_API_BASE` | Build | `https://your-api.onrender.com` |
| `API_UPSTREAM` | Runtime | `https://your-api.onrender.com` |

### 4 — Verify

```
GET https://your-api.onrender.com/health
→ { "status": "ok", "db": true, "model": true }
```

---

## CI — GitHub Actions

Every pull request and push to `main`/`master` runs:

| Job | What it does |
|---|---|
| `api` | `tsc --noEmit` → `tsc` build → `vitest run` (8 route tests) |
| `frontend` | `tsc --noEmit` → `vite build` |
| `ml` | `pip install` → `pytest -v` (dataset + model unit tests) |

---

## Model performance (baseline)

Trained on 4 258 samples, evaluated on 913 held-out samples.

| Metric | Value |
|---|---|
| MAE | 9.26 pp |
| RMSE | 12.06 pp |
| R² | 0.063 |

Low R² is expected for the baseline — features are sparse one-hot encoded vectors and the model
is a shallow MLP. Tune via `ml/experiments/default.yaml`.
