FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY experiments/ ./experiments/

# Data and models are mounted via Docker volumes (see docker-compose.yml)
VOLUME ["/app/data", "/app/models"]

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.train", "--config", "experiments/default.yaml"]
