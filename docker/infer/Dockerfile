# ── Stage 1: instalar dependencias ──────────────────────────────────────────
FROM mirror.gcr.io/library/python:3.11-slim AS builder
WORKDIR /app

COPY requirements_infer.txt .
RUN pip install --no-cache-dir -r requirements_infer.txt "uvicorn[standard]"

# ── Stage 2: imagen final sin pip ni herramientas de build ──────────────────
FROM mirror.gcr.io/library/python:3.11-slim
WORKDIR /app/src/backend

COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/backend/features.py  ./
COPY src/backend/infer.py     ./
COPY src/backend/main_api.py  ./
COPY src/backend/__init__.py  ./

EXPOSE 8000

CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
