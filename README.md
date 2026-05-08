# Swiss Referendum Predictor

End-to-end ML system that predicts Swiss referendum voter turnout.
Built as a production-grade portfolio project covering the full ML lifecycle.

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
│  :5432      │       │  + scaler_params.json│
└─────────────┘       └──────────────────────┘
```

## Tech stack

| Layer | Technology |
|---|---|
| ML training | Python 3.11, PyTorch, wandb |
| Model serving | ONNX Runtime |
| API | TypeScript, Fastify 5, pg |
| Database | PostgreSQL 16 |
| Frontend | React 18, Recharts, Vite |
| Containers | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | Railway (free tier) + Supabase, or Google Cloud Run |

## Dataset

Source: `grosserDatensatzEingabe.csv` / `grosserDatensatzAusgabe.csv` from the Maturaarbeit repo.
Place both files in `ml/data/` (gitignored — must be added manually).

- **Features**: 6 083 referendums × 584 columns (semicolon-delimited, no header)
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
# Artefacts are written to ml/models/ on the host via the volume mount.
# After training, re-export ONNX and restart the API:
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

```json
{
  "title": "Energiegesetz",
  "topic_category": "environment",
  "date": "2024-06-09",
  "features": [0, 1, 6190, 228, 0, 0]
}
```

`features` must be exactly **584** raw (unscaled) values in the same column order as
`grosserDatensatzEingabe.csv`.  The API applies the StandardScaler internally.

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

## Deployment — Railway + Supabase (free tier)

This is the simplest zero-cost path. Railway gives you two free services (API + Frontend)
and Supabase provides a free managed PostgreSQL instance.

### 1 — Supabase database

1. Create a free account at [supabase.com](https://supabase.com) and create a new project.
2. Copy the **Connection string** from **Project Settings → Database → Connection string → URI**.
   It looks like `postgresql://postgres:[PASSWORD]@db.[ref].supabase.co:5432/postgres`.
3. Run the schema migration once from your machine:
   ```bash
   psql "$DATABASE_URL" -f db/init.sql
   ```

### 2 — Commit your ONNX model

The trained model artefacts must be available at deploy time.
The easiest approach is to commit `ml/models/onnx/` to a private repo or upload them
to a Railway volume / object storage and mount them.

Quick path — bake the model into the API image by placing `ml/models/onnx/` at
`api/models/` before building (or adjust `ONNX_MODEL_DIR` to a path inside the container):

```bash
cp -r ml/models/onnx api/models
```

Then set `ONNX_MODEL_DIR=/app/models` in Railway's environment variables.

### 3 — Deploy to Railway

1. Install the Railway CLI: `npm install -g @railway/cli`
2. Log in: `railway login`
3. Create a new project: `railway init`
4. Deploy the API service:
   ```bash
   cd api
   railway up --service api
   ```
5. Set environment variables in the Railway dashboard (or via CLI):
   ```
   DATABASE_URL=postgresql://...      # from Supabase
   ONNX_MODEL_DIR=/app/models
   MODEL_VERSION=v1.0.0
   LOG_PRETTY=false
   PORT=3000
   ```
6. Deploy the frontend service:
   ```bash
   cd frontend
   # Set VITE_API_URL to your Railway API public URL, e.g.:
   railway up --service frontend
   ```
   Add environment variable:
   ```
   VITE_API_URL=https://api-xxxx.railway.app
   ```

Railway auto-detects Dockerfiles and builds each service from the repo root.
Point each service at its respective Dockerfile (`api/Dockerfile`, `frontend/Dockerfile`).

### Railway environment variables summary

| Variable | Service | Value |
|---|---|---|
| `DATABASE_URL` | api | Supabase connection string |
| `ONNX_MODEL_DIR` | api | `/app/models` |
| `MODEL_VERSION` | api | `v1.0.0` |
| `LOG_PRETTY` | api | `false` |
| `PORT` | api | `3000` |
| `VITE_API_URL` | frontend | Railway API public URL |

---

## Deployment — Google Cloud Run

### Prerequisites

1. GCP project with billing enabled
2. APIs enabled: Cloud Run, Artifact Registry, Secret Manager
3. Service account with roles: `Cloud Run Admin`, `Artifact Registry Writer`, `Secret Manager Accessor`
4. Workload Identity Federation configured for the GitHub repo

### One-time setup

```bash
PROJECT=your-project-id
REGION=europe-west6

# Artifact Registry repo
gcloud artifacts repositories create swiss-referendum \
  --repository-format=docker --location=$REGION

# Cloud SQL (Postgres) or use Supabase free tier and set DATABASE_URL
# If using Cloud SQL, create instance + database:
gcloud sql instances create referendum-db \
  --database-version=POSTGRES_16 --tier=db-f1-micro --region=$REGION
gcloud sql databases create referendum_db --instance=referendum-db
gcloud sql users set-password postgres --instance=referendum-db --password=YOUR_PW

# Store DATABASE_URL as a Secret Manager secret
echo -n "postgresql://postgres:YOUR_PW@/referendum_db?host=/cloudsql/..." | \
  gcloud secrets create database-url --data-file=-
```

### GitHub Actions secrets required

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_WIF_PROVIDER` | Workload Identity provider resource name |
| `GCP_SA_EMAIL` | Service account email |
| `DATABASE_URL` | Full PostgreSQL connection string |

Push to `main` to trigger the deploy workflow (`.github/workflows/deploy.yml`).

### Manual deploy (without CI)

```bash
gcloud auth configure-docker europe-west6-docker.pkg.dev

# API
docker build -t europe-west6-docker.pkg.dev/$PROJECT/swiss-referendum/api:latest ./api
docker push europe-west6-docker.pkg.dev/$PROJECT/swiss-referendum/api:latest
gcloud run deploy referendum-api \
  --image=europe-west6-docker.pkg.dev/$PROJECT/swiss-referendum/api:latest \
  --region=$REGION --allow-unauthenticated --port=3000 \
  --set-env-vars="DATABASE_URL=...,MODEL_VERSION=v1.0.0,LOG_PRETTY=false"

# Frontend
docker build -t europe-west6-docker.pkg.dev/$PROJECT/swiss-referendum/frontend:latest ./frontend
docker push europe-west6-docker.pkg.dev/$PROJECT/swiss-referendum/frontend:latest
gcloud run deploy referendum-frontend \
  --image=europe-west6-docker.pkg.dev/$PROJECT/swiss-referendum/frontend:latest \
  --region=$REGION --allow-unauthenticated --port=80
```

---

## CI — GitHub Actions

Every pull request and push to `main`/`master` runs three jobs:

| Job | What it does |
|---|---|
| `api` | `tsc --noEmit` → `tsc` build → `vitest run` (8 route tests) |
| `frontend` | `tsc --noEmit` → `vite build` |
| `ml` | `pip install` → `pytest -v` (dataset + model unit tests) |

The deploy workflow (`.github/workflows/deploy.yml`) triggers on push to `main` and
pushes Docker images to Artifact Registry then redeploys Cloud Run services.

---

## Git workflow

- Never commit directly to `main`
- Feature branches: `feature/dataloader`, `feature/api`, etc.
- Open a PR — CI runs tests on all three layers
- Merge to `main` triggers the deploy workflow

## Model performance (baseline)

Trained on 4 258 samples, evaluated on 913 held-out samples.

| Metric | Value |
|---|---|
| MAE | 9.26 pp |
| RMSE | 12.06 pp |
| R² | 0.063 |

Low R² is expected for the baseline — the 584 features are sparse one-hot encoded and the model
saw 200 epochs maximum. Tune via `ml/experiments/default.yaml`.
