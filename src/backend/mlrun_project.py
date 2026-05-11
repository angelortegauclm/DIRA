"""
mlrun_project.py

Define y ejecuta el ciclo de vida completo del modelo DIRA usando MLRun CE:

  1. Crea (o recupera) el proyecto MLRun 'diraproject'
  2. Registra la función de entrenamiento como Kubernetes Job
  3. Registra la función de serving como deployment Nuclio (FastAPI)
  4. Lanza el Job de entrenamiento y espera a que termine
  5. Si el entrenamiento finaliza correctamente, reinicia el pod de inferencia
     para que cargue el nuevo modelo desde el model-pvc

Este script se ejecuta desde el contenedor dira-train, que tiene MLRun instalado.
En CI/CD lo lanza el workflow de GitHub Actions tras construir y subir las imágenes.

Variables de entorno inyectadas por el Helm chart (ConfigMap dira-config):
  MLRUN_DBPATH     URI del API de MLRun, p.ej. http://mlrun-api.mlrun.svc.cluster.local:8080
  TRAIN_IMAGE      Imagen Docker del entrenamiento (dira-train:<sha>)
  INFER_IMAGE      Imagen Docker de la inferencia (dira-infer:<sha>)
  DATA_DIR         Directorio del dataset montado desde data-pvc
  DATA_FILENAME    Nombre del fichero CSV del dataset
  MODEL_PATH       Ruta donde se guarda el modelo entrenado en model-pvc
  RANDOM_STATE     Semilla aleatoria para reproducibilidad
  N_ITER           Número de iteraciones de RandomizedSearchCV
  CV_FOLDS         Número de folds de validación cruzada
"""

import os
import subprocess
import mlrun
from mlrun.runtimes.mounts import mount_pvc

# ── Configuración ─────────────────────────────────────────────────────────────
MLRUN_DBPATH  = os.getenv("MLRUN_DBPATH", "http://localhost:8080")
PROJECT_NAME  = "diraproject"

# Imágenes precompiladas: en CI/CD llevan el SHA del commit como tag.
# En local se usan las imágenes importadas manualmente en k3s con tag 'latest'.
TRAIN_IMAGE   = os.getenv("TRAIN_IMAGE", "dira-train:latest")
INFER_IMAGE   = os.getenv("INFER_IMAGE",  "dira-infer:latest")

# Parámetros pasados al handler mlrun_train dentro de train.py.
# Sobreescribibles por variables de entorno para ajustar hiperparámetros
# sin reconstruir la imagen.
PARAMS = {
    "data_path":    os.getenv("DATA_DIR", "/data") + "/" + os.getenv("DATA_FILENAME", "diabetes_binary_health_indicators_BRFSS2015.csv"),
    "model_path":   os.getenv("MODEL_PATH", "/model/modelo_diabetes_DIRA.pkl"),
    "random_state": int(os.getenv("RANDOM_STATE", "42")),
    "n_iter":       int(os.getenv("N_ITER", "25")),
    "cv_folds":     int(os.getenv("CV_FOLDS", "5")),
}

# ── Proyecto MLRun ────────────────────────────────────────────────────────────
# get_or_create_project: si el proyecto ya existe en el tracking server lo
# recupera; si no, lo crea. Esto hace el script idempotente para re-ejecuciones.
project = mlrun.get_or_create_project(
    name=PROJECT_NAME,
    context="./",
    user_project=False,   # sin sufijo de usuario (proyecto compartido)
)

# ── Función de entrenamiento (Kubernetes Job) ─────────────────────────────────
# Se usa kind="job" para que MLRun ejecute el entrenamiento como un Job de
# Kubernetes en lugar de en local. Esto permite usar los PVCs del clúster
# donde están los datos y guardar el modelo.
#
# NO se pasa func= ni source_code_target_dir para evitar que MLRun intente
# construir una nueva imagen — se usa la imagen precompilada directamente.
# El handler "train.mlrun_train" apunta a /app/src/backend/train.py::mlrun_train
train_fn = project.set_function(
    name="dira-train",
    kind="job",
    image=TRAIN_IMAGE,
    handler="train.mlrun_train",   # módulo.función dentro del PYTHONPATH del contenedor
)

# WORKDIR del contenedor es /app/src/backend donde residen train.py, features.py, etc.
train_fn.spec.pythonpath = "/app/src/backend"
train_fn.spec.image_pull_policy = "IfNotPresent"   # no re-descargar si ya está en k3s

# Límites de recursos para el Job de entrenamiento.
# El entrenamiento de LightGBM con RandomizedSearchCV necesita CPU y RAM suficientes.
train_fn.with_limits(cpu="2", mem="2Gi")
train_fn.with_requests(cpu="1", mem="1Gi")

# Montar los PVCs creados por el Helm chart de DIRA en el namespace dira.
# mount_pvc(claim_name, volume_name, mount_path)
train_fn.apply(mount_pvc("data-pvc",  "data-pvc",  "/data"))    # dataset CSV de entrenamiento
train_fn.apply(mount_pvc("model-pvc", "model-pvc", "/model"))   # donde se guarda el modelo .pkl

# ── Función de serving (Nuclio / FastAPI) ─────────────────────────────────────
# kind="serving" hace que MLRun despliegue la función como un deployment de
# Nuclio en el namespace mlrun, usando la imagen dira-infer que contiene
# FastAPI + prometheus-fastapi-instrumentator.
serving_fn = project.set_function(
    name="dira-infer",
    kind="serving",
    image=INFER_IMAGE,
)

# El model-pvc contiene el modelo .pkl guardado por el Job de entrenamiento.
# Se monta en /model para que main_api.py pueda cargarlo al arrancar.
serving_fn.apply(mount_pvc("model-pvc", "model-pvc", "/model"))

# Probes de salud: Kubernetes sondea estas rutas para saber si el pod está
# listo para recibir tráfico (ready) y si el proceso sigue vivo (live).
serving_fn.set_config("spec.readinessProbe.httpGet.path", "/v2/health/ready")
serving_fn.set_config("spec.livenessProbe.httpGet.path",  "/v2/health/live")

# Anotaciones de Prometheus para que el scrape de additionalScrapeConfigs
# descubra el pod y sepa en qué puerto y ruta están las métricas.
# Estas anotaciones son leídas por el job 'dira-infer' de values-prometheus.yaml.
serving_fn.metadata.annotations = {
    "prometheus.io/scrape": "true",   # activar el scrape de este pod
    "prometheus.io/port":   "8000",   # puerto donde escucha FastAPI
    "prometheus.io/path":   "/metrics",  # ruta del endpoint de métricas
}

# ── Guardar el proyecto ───────────────────────────────────────────────────────
# Genera project.yaml con la definición del proyecto para reproducibilidad.
project.save()
print(f"[mlrun_project] Proyecto '{PROJECT_NAME}' guardado.")

# ── Lanzar entrenamiento ──────────────────────────────────────────────────────
# local=False: ejecuta en k3s como Kubernetes Job (no en el proceso actual)
# watch=True:  bloquea el script y muestra logs en tiempo real hasta que termine
# output_path: directorio donde MLRun guarda artefactos (métricas, modelo registrado)
print("[mlrun_project] Lanzando Job de entrenamiento en k3s...")
train_run = project.run_function(
    "dira-train",
    params=PARAMS,
    local=False,
    watch=True,
    output_path="/model/mlrun-artifacts",
)

# ── Resultado ─────────────────────────────────────────────────────────────────
print("\n[mlrun_project] ── Resultado del entrenamiento ──────────────────")
print(f"  Run UID  : {train_run.uid()}")
print(f"  Estado   : {train_run.state()}")
print("  Métricas :")
for k, v in train_run.outputs.items():
    print(f"    {k}: {v}")

# ── Reiniciar pod de inferencia ───────────────────────────────────────────────
# Tras un entrenamiento exitoso, el nuevo modelo .pkl está en model-pvc.
# El pod de inferencia tiene el modelo anterior en memoria. Hay que reiniciarlo
# para que cargue el nuevo fichero al arrancar (la carga es lazy en health/ready).
if train_run.state() == "completed":
    print("\n[mlrun_project] Reiniciando pod de inferencia para cargar el nuevo modelo...")
    result = subprocess.run(
        ["kubectl", "rollout", "restart", "deployment/dira-infer", "-n", "mlrun"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("[mlrun_project] Pod reiniciado correctamente.")
    else:
        print(f"[mlrun_project] Error al reiniciar el pod: {result.stderr}")
