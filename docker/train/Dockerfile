# ── Stage 1: instalar dependencias ──────────────────────────────────────────
FROM mirror.gcr.io/library/python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_train.txt .
RUN pip install --no-cache-dir -r requirements_train.txt

# ── Stage 2: imagen final ────────────────────────────────────────────────────
FROM mirror.gcr.io/library/python:3.11-slim
WORKDIR /app/src/backend

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/backend/data_ingestion.py ./
COPY src/backend/features.py       ./
COPY src/backend/train.py          ./
COPY src/backend/__init__.py       ./

CMD ["python", "train.py"]
