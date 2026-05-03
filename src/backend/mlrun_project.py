"""
mlrun_project.py

Define y ejecuta el ciclo de vida del modelo DIRA usando MLRun:
  1. Crea el proyecto MLRun 'dira'
  2. Registra la función de entrenamiento (Job en k3s) apuntando al handler mlrun_train
  3. Registra la función de serving (Nuclio) apuntando al servidor FastAPI existente
  4. Lanza el Job de entrenamiento y espera a que termine
  5. Imprime el UID del run y las métricas registradas

Variables de entorno:
  MLRUN_DBPATH   URI del API de MLRun, p.ej. http://<node-ip>:<nodeport>
"""

import os
import subprocess
import mlrun
from mlrun.runtimes.mounts import mount_pvc

# ---------------------------------------------------------------------------
# Configuración del proyecto
# ---------------------------------------------------------------------------
MLRUN_DBPATH  = os.getenv("MLRUN_DBPATH", "http://localhost:8080")
PROJECT_NAME  = "diraproject"
# En CI/CD estas vars las inyecta el workflow con la imagen de Docker Hub + SHA
# En local se usan los valores por defecto (imágenes importadas manualmente en k3s)
TRAIN_IMAGE   = os.getenv("TRAIN_IMAGE", "dira-train:latest")
INFER_IMAGE   = os.getenv("INFER_IMAGE",  "dira-infer:latest")

# Parámetros de entrenamiento (sobreescribibles por variables de entorno)
PARAMS = {
    "data_path":    os.getenv("DATA_DIR", "/data") + "/" + os.getenv("DATA_FILENAME", "diabetes_binary_health_indicators_BRFSS2015.csv"),
    "model_path":   os.getenv("MODEL_PATH", "/model/modelo_diabetes_DIRA.pkl"),
    "random_state": int(os.getenv("RANDOM_STATE", "42")),
    "n_iter":       int(os.getenv("N_ITER", "25")),
    "cv_folds":     int(os.getenv("CV_FOLDS", "5")),
}

# ---------------------------------------------------------------------------
# Conexión al servidor MLRun y creación del proyecto
# ---------------------------------------------------------------------------
project = mlrun.get_or_create_project(
    name=PROJECT_NAME,
    context="./",
    user_project=False,
)

# ---------------------------------------------------------------------------
# Función de entrenamiento
# Usamos la imagen precompilada directamente — NO pasamos func= para evitar
# que MLRun intente construir y subir una nueva imagen a Docker Hub.
# El handler "train.mlrun_train" apunta a /app/src/train.py::mlrun_train
# ---------------------------------------------------------------------------
train_fn = project.set_function(
    name="dira-train",
    kind="job",
    image=TRAIN_IMAGE,
    handler="train.mlrun_train",
)

# Python path: WORKDIR del contenedor es /app/src/backend, ahí residen los módulos
train_fn.spec.pythonpath = "/app/src/backend"
train_fn.spec.image_pull_policy = "IfNotPresent"

# Recursos del Job
train_fn.with_limits(cpu="2", mem="2Gi")
train_fn.with_requests(cpu="1", mem="1Gi")

# Montar los PVCs creados por el Helm chart de DIRA
train_fn.apply(mount_pvc("data-pvc",  "data-pvc",  "/data"))
train_fn.apply(mount_pvc("model-pvc", "model-pvc", "/model"))

# ---------------------------------------------------------------------------
# Función de serving (FastAPI existente)
# ---------------------------------------------------------------------------
serving_fn = project.set_function(
    name="dira-infer",
    kind="serving",
    image=INFER_IMAGE,
)
serving_fn.apply(mount_pvc("model-pvc", "model-pvc", "/model"))
serving_fn.set_config("spec.readinessProbe.httpGet.path", "/v2/health/ready")
serving_fn.set_config("spec.livenessProbe.httpGet.path",  "/v2/health/live")

# ---------------------------------------------------------------------------
# Guardar el proyecto (genera project.yaml)
# ---------------------------------------------------------------------------
project.save()
print(f"[mlrun_project] Proyecto '{PROJECT_NAME}' guardado.")

# ---------------------------------------------------------------------------
# Lanzar entrenamiento
# ---------------------------------------------------------------------------
print("[mlrun_project] Lanzando Job de entrenamiento en k3s...")
train_run = project.run_function(
    "dira-train",
    params=PARAMS,
    local=False,   # ejecuta en k3s como Kubernetes Job
    watch=True,    # bloquea hasta que el Job termine y muestra logs
    output_path="/model/mlrun-artifacts",
)

print("\n[mlrun_project] ── Resultado del entrenamiento ──────────────────")
print(f"  Run UID  : {train_run.uid()}")
print(f"  Estado   : {train_run.state()}")
print("  Métricas :")
for k, v in train_run.outputs.items():
    print(f"    {k}: {v}")

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
