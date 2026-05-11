"""
main_api.py

Servidor FastAPI que expone la API de inferencia de DIRA siguiendo el protocolo
Open Inference V2 (compatible con MLRun/Nuclio serving).

Responsabilidades:
  - Recibir vectores de 21 variables clínicas y devolver el nivel de riesgo de diabetes
  - Exponer /metrics para Prometheus (prometheus-fastapi-instrumentator)
  - Registrar métricas del modelo: contadores de predicciones, histograma de probabilidades
  - Escribir un log CSV de inferencias en model-pvc para la detección de deriva de datos

Endpoints:
  GET  /v2/health/live             → liveness probe (siempre 200)
  GET  /v2/health/ready            → readiness probe (200 si el modelo está cargado)
  GET  /v2/models/DIRA             → metadatos del modelo
  POST /v2/models/modelo_diabetes_DIRA/infer → inferencia
  GET  /metrics                    → métricas Prometheus (auto-expuesto por instrumentator)
"""

import csv
import os
import sys
import threading
from datetime import datetime

# Añadir el directorio del script al path para importar módulos locales (infer.py)
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Any

# Instrumentator: añade métricas HTTP automáticas (tasa, latencia, errores)
# y las expone en /metrics sin configuración adicional
from prometheus_fastapi_instrumentator import Instrumentator
# prometheus_client: para métricas propias del modelo (contadores, histogramas, gauges)
from prometheus_client import Counter, Histogram, Gauge

from infer import DIRAPredictor

# ── Aplicación FastAPI ────────────────────────────────────────────────────────
app = FastAPI(
    title="DIRA - API",
    description="API MLOps para inferencia de riesgo de diabetes en tiempo real",
    version="1.0.0"
)

# CORS abierto: el frontend nginx hace peticiones desde el navegador del usuario
# y puede estar en cualquier origen. En producción restringir a dominios concretos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Métricas Prometheus ───────────────────────────────────────────────────────

# Contador de predicciones segmentado por nivel de riesgo.
# Permite calcular la distribución de resultados en Grafana y detectar
# si hay un pico anómalo de predicciones de alto riesgo (alerta DiraHighRiskSurge).
PRED_COUNTER = Counter(
    "dira_predictions_total",
    "Predicciones acumuladas por nivel de riesgo",
    ["nivel_riesgo"],   # etiqueta: "BAJO RIESGO", "RIESGO MODERADO", "ALTO RIESGO"
)

# Histograma de probabilidades predichas (valor entre 0.0 y 1.0).
# Los buckets están diseñados para capturar bien los extremos y el umbral
# de decisión (0.30 bajo/moderado, 0.70 moderado/alto).
PROB_HISTOGRAM = Histogram(
    "dira_prediction_probability",
    "Distribución de probabilidades de diabetes predichas",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# Gauge binario: 1 si el modelo está en memoria y listo para inferir, 0 si no.
# La alerta DiraModelNotLoaded se activa cuando este gauge lleva >2 min en 0.
MODEL_LOADED = Gauge(
    "dira_model_loaded",
    "1 si el modelo está cargado en memoria, 0 en caso contrario",
)

# Instrumentar la app: añade automáticamente métricas HTTP (http_requests_total,
# http_request_duration_seconds) y expone el endpoint /metrics.
# Debe llamarse después de registrar el middleware y antes del primer request.
Instrumentator().instrument(app).expose(app)

# ── Logging de inferencias para drift detection ───────────────────────────────
# Cada inferencia se escribe en un CSV compartido con el CronJob de drift.
# El Lock evita que escrituras concurrentes corrompan el fichero cuando llegan
# varios requests simultáneos.
_LOG_LOCK = threading.Lock()
_LOG_PATH = os.getenv("INFERENCE_LOG_PATH", "/model/inference_log.csv")


def _log_inference(features: list, columns: list, prob: float, nivel: str) -> None:
    """Escribe una fila en el CSV de inferencias de forma thread-safe.

    El CSV es leído por drift_check.py (CronJob horario) para detectar si
    la distribución de los datos de entrada ha cambiado respecto al entrenamiento.
    Falla silenciosamente para no interrumpir la respuesta al cliente.
    """
    # Construir la fila con features + metadatos de la predicción
    row = dict(zip(columns, features))
    row["prob_diabetes"] = round(prob, 4)
    row["nivel_riesgo"]  = nivel
    row["ts"]            = datetime.utcnow().isoformat()   # UTC para evitar problemas de zona horaria
    try:
        with _LOG_LOCK:
            # Crear la cabecera solo si el fichero no existe todavía
            new_file = not os.path.exists(_LOG_PATH)
            with open(_LOG_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
    except Exception:
        pass   # el fallo de logging no debe interrumpir la inferencia al usuario


# ── Predictor ─────────────────────────────────────────────────────────────────
# Instancia global del predictor. Se carga lazy (la primera vez que se llama
# a predictor.load()) para no bloquear el arranque del servidor.
predictor = DIRAPredictor(model_path=os.getenv("MODEL_PATH", "./model/modelo_diabetes_DIRA.pkl"))

# ── Health probes ─────────────────────────────────────────────────────────────
# Kubernetes usa estas rutas para saber si el pod está vivo y listo.
# Si /health/ready devuelve error, el pod se saca del balanceador.

@app.get("/v2/health/live")
def health_live():
    """Liveness probe: el proceso está en ejecución. Siempre retorna 200."""
    return {"status": "alive"}


@app.get("/v2/health/ready")
def health_ready():
    """Readiness probe: el modelo está cargado y puede servir predicciones.

    Actualiza el gauge dira_model_loaded para que Prometheus y Alertmanager
    puedan detectar si el modelo no está disponible (alerta DiraModelNotLoaded).
    """
    try:
        predictor.load()
        MODEL_LOADED.set(1)   # modelo OK → gauge a 1
        return {"status": "ready"}
    except Exception as exc:
        MODEL_LOADED.set(0)   # fallo de carga → gauge a 0 → dispara alerta
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {str(exc)}")


# ── Metadatos del modelo ──────────────────────────────────────────────────────
# Variables leídas desde el entorno para poder cambiarlas sin recompilar la imagen
MODEL_NAME     = os.getenv("MODEL_NAME",     "modelo_diabetes_DIRA")
MODEL_VERSION  = os.getenv("MODEL_VERSION",  "1.0.0")
MODEL_PLATFORM = os.getenv("MODEL_PLATFORM", "sklearn_joblib")


@app.get("/v2/models/DIRA")
def model_metadata():
    """Devuelve el esquema del modelo: nombre, versión, entradas y salidas.

    Sigue el protocolo Open Inference V2 para ser compatible con MLRun serving.
    shape: [-1, N] significa "cualquier número de filas, N columnas".
    """
    return {
        "name": MODEL_NAME,
        "versions": [MODEL_VERSION],
        "platform": MODEL_PLATFORM,
        "inputs":  [{"name": "paciente_features", "datatype": "FP32", "shape": [-1, 21]}],
        "outputs": [
            {"name": "prob_diabetes", "datatype": "FP32",  "shape": [-1, 1]},
            {"name": "prediccion",    "datatype": "INT32",  "shape": [-1, 1]},
            {"name": "nivel_riesgo",  "datatype": "BYTES",  "shape": [-1, 1]},
            {"name": "accion",        "datatype": "BYTES",  "shape": [-1, 1]},
        ],
    }


# ── Schemas de entrada ────────────────────────────────────────────────────────
# Pydantic valida automáticamente que el body del POST tenga la estructura esperada
# y devuelve 422 si no se cumple, sin llegar al handler.

class InferenceInput(BaseModel):
    name:     str          # nombre del tensor, p.ej. "paciente_features"
    shape:    List[int]    # dimensiones: [n_pacientes, n_features]
    datatype: str          # tipo de dato, p.ej. "FP32"
    data:     List[Any]    # datos aplanados en row-major order


class InferenceRequest(BaseModel):
    id:     str | None = None   # identificador opcional de la petición
    inputs: List[InferenceInput]


# ── Endpoint de inferencia ────────────────────────────────────────────────────
@app.post(f"/v2/models/{MODEL_NAME}/infer")
def infer(request: InferenceRequest):
    """Ejecuta la inferencia sobre uno o varios pacientes.

    Recibe un tensor de forma [n_pacientes, 21_features], lo pasa al modelo
    LightGBM y devuelve probabilidad, predicción binaria, nivel de riesgo
    y acción recomendada para cada paciente.

    Además de devolver la respuesta, actualiza las métricas de Prometheus
    y escribe cada predicción en el log CSV para la detección de drift.
    """
    try:
        input_tensor = request.inputs[0]
        predictor.load()   # carga lazy: solo lee el fichero la primera vez

        n_pacientes = input_tensor.shape[0]
        n_features  = input_tensor.shape[1]
        n_expected  = len(predictor.expected_columns)

        # Validar dimensiones antes de intentar la inferencia
        if n_features != n_expected:
            raise ValueError(
                f"Número de columnas incorrecto. Se esperaban {n_expected}, se recibieron {n_features}."
            )
        if len(input_tensor.data) != n_pacientes * n_features:
            raise ValueError(
                f"Longitud de datos incorrecta. Se esperaban {n_pacientes * n_features} valores, "
                f"se recibieron {len(input_tensor.data)}."
            )

        # Reconstruir el DataFrame a partir del tensor aplanado en row-major order
        raw = np.array(input_tensor.data).reshape(n_pacientes, n_features)
        df  = pd.DataFrame(raw, columns=predictor.expected_columns)

        # Ejecutar el pipeline completo: preprocesador + LightGBM
        predicciones = predictor.predict(df)
        n_preds      = len(predicciones)

        probs    = predicciones["prob_diabetes"].tolist()
        preds    = predicciones["prediccion"].tolist()
        niveles  = predicciones["nivel_riesgo"].tolist()
        acciones = predicciones["accion"].tolist()

        # Actualizar métricas Prometheus y registrar cada predicción en el CSV
        for i in range(n_preds):
            PRED_COUNTER.labels(nivel_riesgo=niveles[i]).inc()   # incrementar contador por nivel
            PROB_HISTOGRAM.observe(probs[i])                      # añadir al histograma de probabilidades
            _log_inference(raw[i].tolist(), predictor.expected_columns, probs[i], niveles[i])

        # Construir la respuesta en formato Open Inference V2
        response = {
            "model_name":    MODEL_NAME,
            "model_version": MODEL_VERSION,
            "outputs": [
                {"name": "prob_diabetes", "datatype": "FP32",  "shape": [n_preds, 1], "data": probs},
                {"name": "prediccion",    "datatype": "INT32",  "shape": [n_preds, 1], "data": preds},
                {"name": "nivel_riesgo",  "datatype": "BYTES",  "shape": [n_preds, 1], "data": niveles},
                {"name": "accion",        "datatype": "BYTES",  "shape": [n_preds, 1], "data": acciones},
            ],
        }
        # Propagar el ID de la petición si se proporcionó (útil para trazabilidad)
        if request.id:
            response["id"] = request.id

        return response

    except Exception as exc:
        # Devolver 400 con el mensaje de error en lugar de 500 para que el
        # cliente pueda distinguir errores de validación de errores de servidor
        return JSONResponse(
            status_code=400,
            content={"error": f"Error en la inferencia: {str(exc)}"},
        )
