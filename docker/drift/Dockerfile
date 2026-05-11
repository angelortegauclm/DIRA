# ── Dockerfile dira-drift ─────────────────────────────────────────────────────
#
# Imagen ligera para el CronJob de detección de deriva de datos.
# Ejecuta drift_check.py, que compara las inferencias recientes con el
# dataset de referencia usando Evidently AI y empuja las métricas al Pushgateway.
#
# Diseñada para ser pequeña (~270 MB) a propósito: el CronJob arranca y termina
# en ~15 segundos. No necesita MLRun ni el stack de entrenamiento completo.
# Si hubiera que disparar el reentrenamiento, se llama a GitHub Actions por API.
#
# Construcción multi-stage:
#   Stage 1 (builder): instala Evidently y dependencias (algunas con extensiones C)
#   Stage 2 (runtime): imagen limpia solo con los paquetes instalados

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app

COPY requirements_drift.txt .
RUN pip install --no-cache-dir -r requirements_drift.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Sin compiladores ni pip → imagen final más pequeña y con menor superficie de ataque
FROM python:3.11-slim
# WORKDIR en src/monitoring para que Python encuentre drift_check.py directamente
WORKDIR /app/src/monitoring

# Copiar paquetes Python instalados (Evidently, prometheus_client, pandas, requests)
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Solo se necesita drift_check.py: todo lo demás está en los paquetes Python
COPY src/monitoring/drift_check.py .

# El CronJob no necesita CMD personalizado con argumentos:
# toda la configuración llega por variables de entorno (ver drift-cronjob.yaml)
CMD ["python", "drift_check.py"]
