# ── Stage 1: instalar dependencias ──────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_train.txt .
RUN pip install --no-cache-dir -r requirements_train.txt

# ── Stage 2: imagen final ────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/data_ingestion.py src/
COPY src/features.py       src/
COPY src/train.py          src/
COPY src/__init__.py       src/

CMD ["python", "src/train.py"]
