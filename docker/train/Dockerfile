# ── Dockerfile dira-train ─────────────────────────────────────────────────────
#
# Imagen para el Job de entrenamiento de DIRA.
# Ejecuta train.py, que entrena el pipeline LightGBM con RandomizedSearchCV
# y guarda el modelo serializado en /model/modelo_diabetes_DIRA.pkl.
#
# Construcción multi-stage para reducir el tamaño final de la imagen:
#   Stage 1 (builder): instala todas las dependencias de compilación
#   Stage 2 (runtime): copia solo los paquetes Python instalados, sin gcc ni pip

# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Imagen base slim (sin extras) más compiladores necesarios para paquetes
# con extensiones C (lightgbm, scipy, scikit-learn).
FROM mirror.gcr.io/library/python:3.11-slim AS builder
WORKDIR /app

# Instalar compiladores solo en el stage builder. No llegan a la imagen final.
# libffi-dev y libssl-dev son necesarios para cryptography y requests.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_train.txt .
# --no-cache-dir: no guardar el cache de pip → imagen más pequeña
RUN pip install --no-cache-dir -r requirements_train.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Imagen limpia sin compiladores ni pip. Solo los paquetes Python ya instalados
# se copian desde el stage builder, reduciendo la superficie de ataque.
FROM mirror.gcr.io/library/python:3.11-slim
# WORKDIR en src/backend para que Python encuentre los módulos locales
# (data_ingestion.py, features.py) con imports directos sin modificar sys.path
WORKDIR /app/src/backend

# git es necesario en runtime porque MLRun puede clonar repos durante la ejecución
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo los paquetes Python instalados (sin pip, setuptools, wheel)
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar únicamente los módulos Python necesarios para el entrenamiento:
# data_ingestion.py → carga y validación del CSV
# features.py       → preprocesador y split X/y
# train.py          → entrenamiento y serialización del modelo
COPY src/backend/data_ingestion.py ./
COPY src/backend/features.py       ./
COPY src/backend/train.py          ./
COPY src/backend/__init__.py       ./

# Por defecto ejecuta el entrenamiento standalone.
# MLRun sobreescribe el CMD con su propio entrypoint al lanzar como Job.
CMD ["python", "train.py"]
