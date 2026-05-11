# ── Dockerfile dira-infer ─────────────────────────────────────────────────────
#
# Imagen para el servidor de inferencia de DIRA.
# Ejecuta main_api.py con uvicorn: expone la API FastAPI en el puerto 8000
# siguiendo el protocolo Open Inference V2 (compatible con MLRun/Nuclio).
#
# También expone /metrics para Prometheus (prometheus-fastapi-instrumentator)
# y escribe inference_log.csv en /model para la detección de deriva de datos.
#
# Construcción multi-stage:
#   Stage 1 (builder): instala dependencias (algunas requieren compilación C)
#   Stage 2 (runtime): imagen limpia solo con los paquetes Python necesarios

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM mirror.gcr.io/library/python:3.11-slim AS builder
WORKDIR /app

COPY requirements_infer.txt .
# uvicorn[standard] incluye uvloop y httptools para mejor rendimiento async
RUN pip install --no-cache-dir -r requirements_infer.txt "uvicorn[standard]"

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Sin compiladores ni pip en la imagen final → menor tamaño y superficie de ataque
FROM mirror.gcr.io/library/python:3.11-slim
# WORKDIR en src/backend para que main_api.py pueda importar infer.py y features.py
# con imports directos sin modificar sys.path manualmente
WORKDIR /app/src/backend

# Copiar paquetes Python instalados desde el builder (FastAPI, uvicorn, LightGBM...)
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar solo los módulos necesarios para inferencia:
# features.py  → preprocesador (embebido en el .pkl vía cloudpickle, pero importado en tests)
# infer.py     → DIRAPredictor: carga el .pkl y ejecuta predict()
# main_api.py  → servidor FastAPI con /metrics y /infer
COPY src/backend/features.py  ./
COPY src/backend/infer.py     ./
COPY src/backend/main_api.py  ./
COPY src/backend/__init__.py  ./

# Puerto donde uvicorn escucha. Prometheus scrape desde este mismo puerto en /metrics.
EXPOSE 8000

# --host 0.0.0.0: escuchar en todas las interfaces (necesario en contenedor)
# --port 8000: debe coincidir con prometheus.io/port en mlrun_project.py
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
