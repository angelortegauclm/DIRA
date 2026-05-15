"""
drift_check.py

Detecta deriva de datos (data drift) en las inferencias recientes de DIRA
comparando su distribución estadística contra el dataset de entrenamiento.

Flujo:
  1. Carga una muestra del dataset de referencia (entrenamiento)
  2. Carga las inferencias recientes registradas por main_api.py
  3. Ejecuta un análisis de deriva con Evidently AI
  4. Empuja las métricas resultantes al Prometheus Pushgateway
  5. Si el drift supera el umbral, dispara un reentrenamiento vía GitHub Actions

Se ejecuta como CronJob de Kubernetes cada hora. Al ser un proceso batch de
vida corta no puede exponer /metrics directamente — por eso usa Pushgateway.

Variables de entorno:
  DATA_PATH           Ruta al CSV de referencia (muestra del dataset de entrenamiento)
  INFERENCE_LOG_PATH  Ruta al CSV de log de inferencias generado por main_api.py
  PUSHGATEWAY_URL     URL del Prometheus Pushgateway
  DRIFT_THRESHOLD     Umbral de drift para disparar reentrenamiento (default: 0.20)
  GITHUB_TOKEN        PAT de GitHub con scope 'workflow' (feedback loop)
  GITHUB_REPO         Repositorio en formato usuario/repo
  GITHUB_BRANCH       Rama objetivo para workflow_dispatch (default: main)
  MIN_SAMPLES         Mínimo de muestras recientes para ejecutar el análisis (default: 50)
"""

import os
import sys
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

# ── Logging ───────────────────────────────────────────────────────────────────
# Formato estándar de contenedor: timestamp + nivel + mensaje.
# El nivel INFO es suficiente para producción; subir a DEBUG si se necesita
# trazar el comportamiento de Evidently.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [drift] %(message)s")
log = logging.getLogger(__name__)

# ── Configuración desde variables de entorno ──────────────────────────────────
# Todos los parámetros se inyectan por entorno para que el mismo contenedor
# funcione en local, staging y producción sin recompilar la imagen.
DATA_PATH          = os.getenv("DATA_PATH", "/model/reference_sample.csv")
INFERENCE_LOG_PATH = os.getenv("INFERENCE_LOG_PATH", "/model/inference_log.csv")
PUSHGATEWAY_URL    = os.getenv("PUSHGATEWAY_URL", "http://pushgateway-prometheus-pushgateway.monitoring.svc.cluster.local:9091")
DRIFT_THRESHOLD    = float(os.getenv("DRIFT_THRESHOLD", "0.30"))
GITHUB_TOKEN       = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO        = os.getenv("GITHUB_REPO", "")
GITHUB_BRANCH      = os.getenv("GITHUB_BRANCH", "main")
MIN_SAMPLES        = int(os.getenv("MIN_SAMPLES", "50"))

# ── Columnas del modelo ───────────────────────────────────────────────────────
# Las 21 variables clínicas que recibe el modelo como entrada.
# Se usan para filtrar tanto la referencia como el log de inferencias,
# descartando columnas auxiliares como timestamp o resultado de predicción.
FEATURE_COLS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]


def load_reference() -> pd.DataFrame:
    """
    Carga la muestra de referencia del dataset de entrenamiento.

    El archivo reference_sample.csv lo genera train.py al final de cada
    entrenamiento y lo guarda en el model-pvc junto al modelo. Usar una
    muestra fija (5000 filas, semilla 42) garantiza reproducibilidad: dos
    ejecuciones del drift check con los mismos datos de inferencia producirán
    exactamente el mismo score.

    Se limita a 5000 filas porque Evidently calcula tests estadísticos
    (Kolmogorov-Smirnov, chi-cuadrado) que no mejoran con más datos pero
    sí aumentan el tiempo de cómputo.
    """
    log.info(f"Cargando referencia desde {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # Muestreo aleatorio reproducible — misma semilla que train.py
    return df[FEATURE_COLS].sample(n=min(5000, len(df)), random_state=42)


def load_current() -> pd.DataFrame | None:
    """
    Carga las inferencias recientes registradas por main_api.py.

    Aplica dos filtros para que el análisis represente el comportamiento
    actual del sistema y no el histórico acumulado:

    1. Ventana temporal: si el log tiene columna 'ts', se prefieren las
       últimas 24 horas. Esto evita que inferencias antiguas (cuando el
       modelo era correcto) diluyan una deriva reciente.

    2. Límite de filas: se toman las últimas 500 del período seleccionado.
       Suficiente para que Evidently tenga potencia estadística sin cargar
       el CSV completo en memoria.

    Devuelve None si no hay suficientes muestras para un análisis fiable.
    Los tests estadísticos pierden validez con menos de ~50 observaciones.
    """
    if not os.path.exists(INFERENCE_LOG_PATH):
        log.warning(f"Log de inferencias no encontrado: {INFERENCE_LOG_PATH}")
        return None

    df = pd.read_csv(INFERENCE_LOG_PATH)

    # Verificar mínimo de muestras antes de cualquier filtrado
    if len(df) < MIN_SAMPLES:
        log.warning(f"Solo {len(df)} muestras en el log (mínimo {MIN_SAMPLES}). Saltando análisis.")
        return None

    # Preferir las muestras de las últimas 24 horas si el log tiene timestamp.
    # Si el período reciente no alcanza el mínimo se usa todo el histórico.
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent = df[df["ts"] >= cutoff]
        if len(recent) >= MIN_SAMPLES:
            df = recent

    # Seleccionar solo las columnas disponibles en el log que coinciden con
    # las features del modelo (puede haber columnas extra como prediccion, prob)
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].tail(500)


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[float, dict]:
    """
    Ejecuta el análisis de deriva de datos con Evidently AI.

    Evidently compara las distribuciones de referencia y actuales usando:
    - Test de Kolmogorov-Smirnov para variables continuas (BMI, MentHlth...)
    - Test chi-cuadrado para variables categóricas/binarias (HighBP, Stroke...)

    Devuelve:
      drift_score   Proporción de columnas que han derivado (0.0 – 1.0).
                    Ejemplo: si 15 de 21 variables han cambiado, score = 0.71
      drift_by_col  Diccionario {nombre_columna: p_value} para empujar
                    métricas por variable al Pushgateway.
    """
    # DatasetDriftMetric: score global (share_of_drifted_columns)
    # DataDriftPreset:    desglose columna a columna con p-values individuales
    report = Report(metrics=[DatasetDriftMetric(), DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()

    # El score global está en el primer metric del resultado
    drift_score = result["metrics"][0]["result"]["share_of_drifted_columns"]

    # Los scores por columna están en metrics[2]["result"]["drift_by_columns"],
    # un dict {nombre_columna: {drift_score, drift_detected, ...}}
    drift_by_col = {
        col: float(data.get("drift_score", 0.0))
        for col, data in result["metrics"][2]["result"]["drift_by_columns"].items()
    }

    return drift_score, drift_by_col


def push_metrics(drift_score: float, drift_by_col: dict) -> None:
    """
    Empuja las métricas de deriva al Prometheus Pushgateway.

    Se usa Pushgateway en lugar de exponer /metrics directamente porque este
    script es un proceso batch de vida corta (~15s). Prometheus scrape cada
    15s — si el pod ya terminó cuando llega el scrape, las métricas se
    pierden. El Pushgateway las almacena hasta el siguiente ciclo de scrape.

    Se crea un CollectorRegistry propio (no el global) para evitar enviar
    métricas internas del proceso Python (GC, memoria, threads) que
    contaminarían el Pushgateway con ruido irrelevante.

    Métricas empujadas:
      dira_data_drift_score              Score global (0.0 – 1.0)
      dira_column_drift_score{column}    Score por variable clínica
    """
    # Registry aislado: solo contiene las métricas de drift, sin ruido del proceso
    registry = CollectorRegistry()

    Gauge("dira_data_drift_score", "Score global de deriva de datos", registry=registry).set(drift_score)

    # Gauge con etiqueta 'column' para poder filtrar por variable en Grafana
    g_col = Gauge("dira_column_drift_score", "Score de deriva por columna", ["column"], registry=registry)
    for col, score in drift_by_col.items():
        g_col.labels(column=col).set(score)

    try:
        push_to_gateway(PUSHGATEWAY_URL, job="dira_drift", registry=registry)
        log.info(f"Métricas empujadas al Pushgateway. Drift global: {drift_score:.4f}")
    except Exception as exc:
        # Error no fatal: el análisis ya está hecho. Se registra pero no se
        # aborta — el umbral de reentrenamiento se evalúa igualmente.
        log.error(f"Error empujando métricas: {exc}")


def trigger_retraining() -> None:
    """
    Dispara un reentrenamiento automático vía GitHub Actions workflow_dispatch.

    En lugar de ejecutar el reentrenamiento directamente desde este contenedor
    (lo que requeriría meter todo el stack de MLRun + LightGBM en la imagen
    de drift, triplicando su tamaño), se llama a la API de GitHub para
    activar el workflow de CI/CD existente con el input run_training=true.

    Esto reutiliza el pipeline de entrenamiento ya probado y mantiene la
    imagen de drift ligera (~270 MB frente a ~1.4 GB del contenedor de train).

    Requiere un Personal Access Token (PAT) con scope 'workflow' almacenado
    en el Secret de Kubernetes 'dira-github-secret'.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        log.warning("GITHUB_TOKEN / GITHUB_REPO no configurados. Reentrenamiento no disparado.")
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/ci-cd.yml/dispatches"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
        # workflow_dispatch requiere ref (rama) e inputs definidos en el YAML del workflow
        json={"ref": GITHUB_BRANCH, "inputs": {"run_training": "true"}},
        timeout=10,
    )
    if resp.status_code == 204:
        # 204 No Content es la respuesta correcta de GitHub para workflow_dispatch
        log.info("Reentrenamiento disparado vía GitHub Actions workflow_dispatch.")
    else:
        log.error(f"Error disparando reentrenamiento: HTTP {resp.status_code} – {resp.text}")


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=== DIRA Drift Check ===")

    # Paso 1: cargar datos de referencia (muestra fija del dataset de entrenamiento)
    reference = load_reference()

    # Paso 2: cargar inferencias recientes del log generado por main_api.py
    current = load_current()

    # Si no hay suficientes inferencias recientes, salir sin error.
    # El CronJob marcará el job como Completed (no Failed) para no generar
    # alertas falsas en las primeras horas tras un despliegue.
    if current is None:
        log.info("Sin muestras suficientes para el análisis. Saliendo.")
        sys.exit(0)

    log.info(f"Referencia: {len(reference)} filas | Actual: {len(current)} filas")

    # Paso 3: ejecutar análisis estadístico de deriva con Evidently
    drift_score, drift_by_col = run_drift_report(reference, current)

    # Mostrar las 5 variables con mayor deriva para diagnóstico rápido en logs
    log.info(f"Drift score global: {drift_score:.4f} (umbral: {DRIFT_THRESHOLD})")
    for col, score in sorted(drift_by_col.items(), key=lambda x: -x[1])[:5]:
        log.info(f"  {col}: {score:.4f}")

    # Paso 4: enviar métricas a Prometheus vía Pushgateway
    push_metrics(drift_score, drift_by_col)

    # Paso 5: feedback loop — si hay deriva significativa, reentrenar el modelo
    if drift_score > DRIFT_THRESHOLD:
        log.warning(f"Drift {drift_score:.4f} > umbral {DRIFT_THRESHOLD}. Disparando reentrenamiento...")
        trigger_retraining()

    log.info("=== Análisis completado ===")
