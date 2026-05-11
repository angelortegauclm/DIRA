# DIRA – Guía de instalación y ejecución

Stack: **k3s** (clúster Kubernetes ligero) · **Helm** (despliegue de infraestructura) · **MLRun CE** (ciclo de vida del modelo)

---

## Prerrequisitos

### Sistema y herramientas base

| Requisito | Versión mínima | Instalación | Comprobación |
|---|---|---|---|
| Sistema operativo | Ubuntu 22.04 / Debian 12 (Linux) | — | `uname -r` |
| Docker Engine | 24.x | `curl -fsSL https://get.docker.com \| sh` | `docker --version` |
| Python | 3.9 – 3.11 | Ver nota Conda abajo | `python3 --version` |
| Miniconda | cualquiera | `curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \| bash` | `conda --version` |
| git | 2.x | `sudo apt install git` | `git --version` |
| curl | cualquiera | `sudo apt install curl` | `curl --version` |

> **Python y Conda:** `mlrun==1.10.0` requiere Python `>=3.9, <3.12`. Crear el entorno
> conda con Python 3.9 antes del paso 7:
> ```bash
> conda create -n dira python=3.9 -y
> conda activate dira
> ```
> Usar siempre `conda activate dira` para ejecutar `mlrun_project.py`.

> Docker debe estar corriendo y el usuario actual debe poder ejecutarlo sin `sudo`:
> ```bash
> sudo usermod -aG docker $USER
> # Cerrar sesión y volver a entrar para que el grupo surta efecto
> ```

### Herramientas que se instalan durante esta guía

Estos componentes **no necesitas tenerlos antes** — los pasos 1 y 2 los instalan:

| Herramienta | Paso de instalación | Función |
|---|---|---|
| k3s | Paso 1 | Clúster Kubernetes ligero (incluye `kubectl`) |
| Helm | Paso 2 | Gestor de paquetes de Kubernetes |
| MLRun CE | Paso 4 | Ciclo de vida del modelo (API, UI, Nuclio, Minio, MySQL) |
| mlrun SDK | Paso 7 | Librería Python para orquestar Jobs desde local |

### Dataset

El CSV de entrada debe estar presente antes de ejecutar el entrenamiento:

```
./data/diabetes_binary_health_indicators_BRFSS2015.csv
```

Descargable en [Kaggle – Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).

### Puertos que deben estar libres

| Puerto | Servicio |
|---|---|
| 6443 | API de Kubernetes (k3s) |
| 8080 | MLRun API (NodePort interno) |
| 8060 | MLRun UI (port-forward local, solo para monitorización) |
| 9000 | Minio (almacenamiento de artefactos) |
| 30090 | Prometheus (NodePort) |
| 30030 | Grafana (NodePort) |
| 30093 | Alertmanager (NodePort) |

---

## 1. Instalar k3s

```bash
curl -sfL https://get.k3s.io | sh -

# Copiar kubeconfig al directorio del usuario
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER ~/.kube/config

# Verificar que el clúster está listo
kubectl get nodes
# Esperado: NAME   STATUS   ROLES                  AGE   VERSION
#           ...    Ready    control-plane,master   ...   v1.x.x
```

> k3s usa **containerd** en lugar del daemon Docker.  
> Las imágenes construidas con Docker deben importarse explícitamente (ver paso 3).

---

## 2. Instalar Helm

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verificar
helm version
# Esperado: version.BuildInfo{Version:"v3.x.x", ...}
```

---

## 3. Construir e importar las imágenes Docker en k3s

```bash
# Desde la raíz del repositorio
docker build -t dira-train:latest -f docker/train/Dockerfile .
docker build -t dira-infer:latest  -f docker/infer/Dockerfile .
docker build -t dira-front:latest  -f docker/front/Dockerfile .

# Importar en el containerd de k3s (necesario porque k3s no usa el daemon Docker)
docker save dira-train:latest | sudo k3s ctr images import -
docker save dira-infer:latest  | sudo k3s ctr images import -
docker save dira-front:latest  | sudo k3s ctr images import -

# Verificar que están disponibles para k3s
sudo k3s ctr images ls | grep dira
```

> **Nota sobre paquetes Python:** el Dockerfile de entrenamiento instala `xgboost-cpu`
> (variante sin CUDA), no `xgboost`. Si cambias la imagen base a una con GPU deberás
> ajustar también `requirements_train.txt`.
>
> **Nota sobre versiones Python:** los Dockerfiles de entrenamiento e inferencia usan
> `python:3.11-slim`. Es imprescindible que ambos usen la misma versión principal de
> Python para que el modelo serializado con `cloudpickle` sea deserializable en el
> contenedor de inferencia.

---

## 4. Desplegar MLRun Community Edition con Helm

MLRun CE despliega automáticamente: API server, UI, MySQL (metadata), Minio (artefactos) y Nuclio (serving).

```bash
# Añadir repositorio oficial de MLRun
helm repo add mlrun-ce https://mlrun.github.io/ce
helm repo update

# Crear namespace e instalar (puede tardar 3-5 minutos)
helm install mlrun mlrun-ce/mlrun-ce \
  --namespace mlrun \
  --create-namespace \
  --set nuclio.dashboard.enabled=true \
  --wait --timeout 10m

# Verificar que todos los pods están Running
kubectl get pods -n mlrun
```

Una vez desplegado, obtener el NodePort del API de MLRun:

```bash
export MLRUN_NODEPORT=$(kubectl get svc -n mlrun mlrun-api \
  -o jsonpath='{.spec.ports[0].nodePort}')
export NODE_IP=$(kubectl get nodes \
  -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

export MLRUN_DBPATH="http://${NODE_IP}:${MLRUN_NODEPORT}"
echo "MLRun API: $MLRUN_DBPATH"
```

---

## 5. Desplegar la infraestructura DIRA con Helm

El chart DIRA crea el namespace, el ConfigMap, los dos PVCs (`data-pvc` y `model-pvc`) y el frontend web.

```bash
# Obtener IP y NodePort del servicio de inferencia (si ya está desplegado)
NODE_IP=$(kubectl get nodes \
  -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
INFER_PORT=$(kubectl get svc dira-infer -n mlrun \
  -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "")

# Instalar el chart (con URL del backend si ya se conoce)
helm install dira helm/dira --namespace dira --create-namespace \
  ${INFER_PORT:+--set frontend.apiUrl=http://${NODE_IP}:${INFER_PORT}}

# Verificar
kubectl get configmap,pvc,pods -n dira
```

> Si el servicio de inferencia aún no existe en el momento de la instalación inicial,
> omite el flag `--set frontend.apiUrl`. El CI/CD lo actualizará automáticamente en
> cada `helm upgrade` una vez que el servicio esté desplegado.

Para cambiar cualquier parámetro sin modificar el código:

```bash
helm upgrade dira helm/dira --namespace dira \
  --set config.nIter=50 \
  --set config.cvFolds=10
```

---

## 6. Cargar el dataset en el PVC

El CSV debe estar en el PVC `data-pvc` antes de lanzar el entrenamiento.

```bash
# Levantar un pod temporal con acceso al PVC
kubectl run data-loader \
  --image=busybox:1.36 \
  --restart=Never \
  --namespace dira \
  --overrides='{
    "spec": {
      "volumes": [{"name":"data","persistentVolumeClaim":{"claimName":"data-pvc"}}],
      "containers": [{
        "name": "data-loader",
        "image": "busybox:1.36",
        "command": ["sleep","3600"],
        "volumeMounts": [{"mountPath":"/data","name":"data"}]
      }]
    }
  }'

# Esperar a que el pod esté Running
kubectl wait pod/data-loader -n dira --for=condition=Ready --timeout=60s

# Copiar el CSV
kubectl cp \
  data/diabetes_binary_health_indicators_BRFSS2015.csv \
  dira/data-loader:/data/diabetes_binary_health_indicators_BRFSS2015.csv

# Verificar la copia
kubectl exec -n dira data-loader -- ls -lh /data/

# Eliminar el pod temporal
kubectl delete pod data-loader -n dira
```

---

## 7. Instalar el SDK de MLRun (entorno local)

El script `mlrun_project.py` se ejecuta desde la máquina local y se comunica con el API de MLRun en k3s. Debe ejecutarse dentro del entorno conda `dira` (Python 3.9), ya que `mlrun==1.10.0` no es compatible con Python 3.12.

```bash
conda activate dira
pip install "mlrun==1.10.0"
```

> **Importante:** usa exactamente `mlrun==1.10.0`. Versiones anteriores (p.ej. 1.5.x) no
> son compatibles con la API que se despliega con la versión actual del chart MLRun CE.
> El fichero `requirements_train.txt` también especifica esta versión para mantener la
> coherencia entre el entorno local y la imagen de entrenamiento.

---

## 8. Ejecutar el ciclo de vida del modelo

```bash
conda activate dira

# Asegurarse de que MLRUN_DBPATH apunta al API de MLRun en k3s (ver paso 4)
echo $MLRUN_DBPATH

# Lanzar entrenamiento + registro en MLRun
python src/backend/mlrun_project.py
```

El script:
1. Crea el proyecto `diraproject` en MLRun
2. Lanza un **Kubernetes Job** en k3s con la imagen `dira-train:latest`
3. El Job entrena el modelo XGBoost con RandomizedSearchCV
4. Registra automáticamente en MLRun: parámetros, métricas (`ap_test`, `recall`, `precision`, `f1`, `tp`, `fp`, `fn`, `tn`) y el artefacto `.pkl`
5. Registra el modelo en el **Model Registry** de MLRun con el nombre `dira-pipeline`
6. Despliega la función de serving `dira-infer` via Nuclio

Salida esperada al finalizar:

```
[mlrun_project] Proyecto 'diraproject' guardado.
[mlrun_project] Lanzando Job de entrenamiento en k3s...
...
[mlrun_project] ── Resultado del entrenamiento ──────────────────
  Run UID  : <uid>
  Estado   : completed
  Métricas :
    ap_test: 0.xxxx
    recall: 0.xxxx
    ...
```

> **Nota sobre pythonpath:** el Job configura `spec.pythonpath = "/app/src/backend"` para
> que los imports dentro del contenedor (`from data_ingestion import ...`) se resuelvan
> correctamente. El `WORKDIR` del contenedor es también `/app/src/backend`. Sin esta
> configuración el Job falla con `ModuleNotFoundError`.

---

## 9. Monitorizar en la UI de MLRun

```bash
# Abrir tunnel al UI de MLRun (accesible en http://localhost:8060)
kubectl port-forward -n mlrun svc/mlrun-ui 8060:80
```

Navegar a `http://localhost:8060`:
- **Projects → diraproject → Jobs**: ver historial de runs con métricas
- **Projects → diraproject → Models**: ver `dira-pipeline` en el Model Registry
- **Projects → diraproject → Functions**: ver `dira-train` y `dira-infer`

---

## 10. Verificar la API de inferencia

Una vez que el serving esté desplegado por MLRun/Nuclio, obtener su endpoint:

```bash
kubectl get svc -n nuclio | grep dira-infer
```

Probar con una petición de ejemplo (protocolo Open Inference V2).  
El campo `data` es un **array plano** de números; `shape` indica `[n_pacientes, n_features]`:

```bash
curl -X POST http://<node-ip>:<port>/v2/models/modelo_diabetes_DIRA/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "paciente_features",
      "shape": [1, 21],
      "datatype": "FP32",
      "data": [1.0, 1.0, 1.0, 35.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
               1.0, 1.0, 0.0, 4.0, 10.0, 15.0, 1.0, 1.0, 9.0, 4.0, 3.0]
    }]
  }'
```

> **Formato del campo `data`:** lista plana de `shape[0] × shape[1]` valores numéricos,
> en el mismo orden que las columnas del dataset de entrenamiento.  
> La API valida que `shape[1]` coincida con el número de features del modelo (21).  
> El campo `id` a nivel de request es opcional (compatibilidad OIP V2).

---

## 11. Acceder a la interfaz web

Una vez que el chart DIRA está instalado y el contenedor del frontend está Running:

```bash
# Verificar que el pod del frontend está activo
kubectl get pods -n dira -l app=dira-front

# Obtener la IP del nodo
kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}'
```

La interfaz web está disponible en `http://<node-ip>:30100`.

> Si el frontend muestra el formulario pero la respuesta de la API es un error de red,
> verificar que `frontend.apiUrl` apunta al NodePort correcto del servicio `dira-infer`:
> ```bash
> helm upgrade dira helm/dira --namespace dira \
>   --set frontend.apiUrl=http://<node-ip>:<infer-port>
> ```

---

## 12. Reentrenar el modelo

Para lanzar un nuevo ciclo de entrenamiento con parámetros distintos:

```bash
# Opción A: sobreescribir variables de entorno antes de ejecutar
N_ITER=50 CV_FOLDS=10 python src/backend/mlrun_project.py

# Opción B: actualizar el ConfigMap vía Helm y relanzar
helm upgrade dira helm/dira --namespace dira --set config.nIter=50
python src/backend/mlrun_project.py
```

Cada ejecución genera un nuevo Run en MLRun con su propio UID, métricas y artefacto versionado. El historial completo queda disponible en la UI.

---

## 13. Configurar el pipeline CI/CD (GitHub Actions)

El pipeline automatiza: build de imágenes → push a Docker Hub → despliegue en k3s → entrenamiento MLRun si el código del modelo cambia.

### 13.1 Instalar el runner self-hosted en la máquina con k3s

El job `deploy` del workflow necesita acceso directo al clúster, por lo que el runner debe correr en la misma máquina que k3s.

En GitHub: **Settings → Actions → Runners → New self-hosted runner** y seguir las instrucciones. En resumen:

```bash
# Crear directorio y descargar el runner (sustituir la URL por la que da GitHub)
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64.tar.gz -L https://github.com/actions/runner/releases/download/v2.x.x/actions-runner-linux-x64-2.x.x.tar.gz
tar xzf actions-runner-linux-x64.tar.gz

# Registrar el runner en el repositorio (usar el token que da GitHub)
./config.sh --url https://github.com/<usuario>/<repo> --token <TOKEN>

# Instalar y arrancar como servicio del sistema
sudo ./svc.sh install
sudo ./svc.sh start
```

Verificar que el runner aparece como **Online** en GitHub Settings → Actions → Runners.

### 13.2 Crear los secretos en GitHub

En **Settings → Secrets and variables → Actions → New repository secret**:

| Secreto | Valor |
|---|---|
| `DOCKERHUB_USERNAME` | Tu usuario de Docker Hub |
| `DOCKERHUB_TOKEN` | Access token de Docker Hub (no la contraseña) |
| `MLRUN_DBPATH` | URL del API de MLRun, p.ej. `http://192.168.1.x:PORT` |

> Para obtener un token de Docker Hub: **Docker Hub → Account Settings → Security → New Access Token**.
>
> Para obtener el valor de `MLRUN_DBPATH` dinámicamente desde el clúster:
> ```bash
> export PORT=$(kubectl get svc -n mlrun mlrun-api -o jsonpath='{.spec.ports[0].nodePort}')
> export IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
> echo "http://${IP}:${PORT}"
> ```

### 13.3 Comportamiento del pipeline

```
push a main
    │
    ├─► Job build (GitHub runner)
    │       ├─ docker build dira-train  → push <user>/dira-train:<sha> + :latest
    │       ├─ docker build dira-infer  → push <user>/dira-infer:<sha> + :latest
    │       └─ docker build dira-front  → push <user>/dira-front:<sha> + :latest
    │
    └─► Job deploy (self-hosted runner en k3s)  [necesita build]
            ├─ docker pull + k3s ctr images import  (tres imágenes)
            ├─ helm upgrade --install dira           (ConfigMap + PVCs + frontend)
            │     └─ calcula NodePort infer y pasa --set frontend.apiUrl=...
            ├─ [si src/backend/train.py|features.py|data_ingestion.py|requirements_train.txt cambió
            │   O se activó manualmente]
            │       ├─ conda activate dira + pip install mlrun==1.10.0
            │       ├─ python src/backend/mlrun_project.py  (Job entrenamiento + serving)
            │       └─ kubectl rollout restart deployment/dira-infer  (auto-restart)
            └─ [tras entrenamiento] curl /v2/health/ready  (health check)
```

### 13.4 Forzar un entrenamiento sin cambiar código

```
GitHub → Actions → DIRA CI/CD → Run workflow → ✓ Forzar lanzamiento del entrenamiento MLRun → Run
```

---

## 14. Desplegar el stack de monitorización

### 14.1 Instalar kube-prometheus-stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  -f helm/monitoring/values-prometheus.yaml
```

Acceso una vez desplegado:
- **Prometheus**: `http://<node-ip>:30090`
- **Grafana**: `http://<node-ip>:30030` (usuario: `admin`, contraseña: `dira-admin`)
- **Alertmanager**: `http://<node-ip>:30093`

### 14.2 Instalar Prometheus Pushgateway

```bash
helm install pushgateway prometheus-community/prometheus-pushgateway \
  --namespace monitoring
```

### 14.3 Aplicar las reglas de alerta

```bash
kubectl apply -f helm/monitoring/dira-rules.yaml

# Verificar que Prometheus las ha cargado
kubectl get prometheusrule -n monitoring
```

### 14.4 Configurar Telegram

Antes de instalar el stack, editar `helm/monitoring/values-prometheus.yaml` y rellenar:

```yaml
telegram_configs:
  - bot_token: "TOKEN_DEL_BOT"   # BotFather → /newbot
    chat_id: 123456789            # ID del grupo o canal destino
```

Para obtener el `chat_id` de un grupo, añadir el bot al grupo y consultar:
```
https://api.telegram.org/bot<TOKEN>/getUpdates
```

### 14.5 Construir e importar la imagen de drift

```bash
docker build -t dira-drift:latest -f docker/drift/Dockerfile .
docker save dira-drift:latest | sudo k3s ctr images import -
```

### 14.6 Crear el secret de GitHub (feedback loop)

El CronJob de drift dispara un reentrenamiento automático vía GitHub Actions cuando detecta deriva. Requiere un PAT con scope `workflow`:

```bash
kubectl create secret generic dira-github-secret \
  --from-literal=token=<GITHUB_PAT> \
  --namespace dira
```

> Crear el PAT en **GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens**, con permiso `Actions: write` sobre el repositorio DIRA.

### 14.7 Activar el CronJob de drift

El CronJob se despliega automáticamente con el chart DIRA. Para verificarlo:

```bash
kubectl get cronjob -n dira
# Esperado: dira-drift-check   0 * * * *   ...

# Forzar una ejecución manual
kubectl create job --from=cronjob/dira-drift-check drift-manual -n dira
kubectl logs -n dira -l job-name=drift-manual -f
```

### 14.8 Simular un entorno anómalo

Para probar las alertas de drift y alto riesgo, enviar tráfico anómalo a la API:

```bash
python src/monitoring/simulate_anomaly.py \
  --url http://<node-ip>:<infer-port> \
  --n 200 \
  --delay 0.05
```

Tras la ejecución, el CronJob de drift (o la ejecución manual del paso anterior) detectará la deriva y disparará la alerta en Telegram.

---

## 15. Resolución de problemas conocidos

### `ModuleNotFoundError: No module named 'data_ingestion'` en el Job de MLRun

El Job usa `WORKDIR=/app/src/backend`. La configuración
`train_fn.spec.pythonpath = "/app/src/backend"` en `mlrun_project.py` resuelve esto.
Verificar que esa línea está presente y que los ficheros del backend están en
`src/backend/` (no en `src/` directamente).

### `unknown opcode 0` al cargar el modelo en el contenedor de inferencia

`cloudpickle` serializa bytecode Python. Si el modelo fue entrenado con Python 3.11 y el
contenedor de inferencia usa Python 3.12 (u otra versión principal distinta), la
deserialización falla con este error.

Solución: asegurarse de que ambos Dockerfiles (`docker/train/Dockerfile` y
`docker/infer/Dockerfile`) usan la misma versión base:
```dockerfile
FROM mirror.gcr.io/library/python:3.11-slim
```
Después, regenerar ambas imágenes, reimportarlas en k3s y relanzar el entrenamiento para
que el nuevo artefacto `.pkl` sea compatible.

### `TypeError: XGBClassifier.__init__() got an unexpected keyword argument 'use_label_encoder'`

El parámetro `use_label_encoder` fue eliminado en XGBoost ≥ 2.0. Si aparece este error,
eliminar ese argumento del constructor `XGBClassifier(...)` en `train.py`.

### `ImportError: cannot import name 'unique_values' from 'numpy'`

`numpy.unique_values` no existe. La función correcta es `numpy.unique`. Si aparece este
error en `data_ingestion.py`, eliminar la línea `from numpy import unique_values`.

### El endpoint de inferencia devuelve error 422 (Unprocessable Entity)

Verificar que el campo `data` del request es una **lista plana de números**, no una
lista de objetos/dicts. La longitud de `data` debe ser `shape[0] × shape[1]` (p.ej.
2 pacientes × 21 features = 42 valores). Si `shape[1]` no coincide con las 21 columnas
del modelo la API devuelve error 400 con mensaje descriptivo.

### `AttributeError: 'DIRAPredictor' object has no attribute '_pipeline'` al arrancar

Ocurre si se intenta acceder a `expected_columns` o al modelo antes de llamar a `.load()`.
El predictor usa lazy-loading: el modelo se carga la primera vez que llega una petición
(no al iniciar la aplicación). La sonda de liveness (`/v2/health/live`) responde sin
cargar el modelo; la de readiness (`/v2/health/ready`) responde solo cuando está cargado.

### El CI/CD falla con `ModuleNotFoundError: No module named 'mlrun'` en el job `deploy`

El SDK de MLRun no está preinstalado en el runner self-hosted. El workflow instala
automáticamente `mlrun==1.10.0` antes de ejecutar `mlrun_project.py`. Si el paso
"Instalar SDK MLRun" no aparece en el workflow, actualizar `.github/workflows/ci-cd.yml`.

### El CI/CD falla con `No matching distribution found for mlrun==1.10.0` en el job `deploy`

El runner self-hosted usa Python 3.12 por defecto y `mlrun==1.10.0` requiere `<3.12`.
El workflow debe activar el entorno conda `dira` (Python 3.9) antes de instalar mlrun:
```yaml
run: |
  source /home/angel/.miniconda3/etc/profile.d/conda.sh
  conda activate dira
  pip install --quiet "mlrun==1.10.0"
```

---

## Resumen de comandos de referencia

```bash
# Estado del clúster
kubectl get nodes
kubectl get pods -n mlrun
kubectl get pods -n dira
kubectl get pods -n monitoring

# Obtener MLRUN_DBPATH actual
kubectl get svc -n mlrun mlrun-api -o jsonpath='{.spec.ports[0].nodePort}'

# Logs del Job de entrenamiento más reciente
kubectl logs -n mlrun -l mlrun/class=job --tail=100

# Logs del CronJob de drift
kubectl logs -n dira -l job-name=dira-drift-check --tail=100

# Forzar ejecución manual del drift check
kubectl create job --from=cronjob/dira-drift-check drift-manual -n dira

# Simular entorno anómalo
python src/monitoring/simulate_anomaly.py --url http://<node-ip>:<port> --n 200

# Actualizar parámetros del chart DIRA
helm upgrade dira helm/dira --namespace dira --set config.nIter=50

# Actualizar configuración de monitorización (ej. umbral de drift)
helm upgrade monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring -f helm/monitoring/values-prometheus.yaml

# Desinstalar todo
helm uninstall dira        --namespace dira
helm uninstall mlrun       --namespace mlrun
helm uninstall monitoring  --namespace monitoring
helm uninstall pushgateway --namespace monitoring
sudo k3s-uninstall.sh
```

---

## Estructura del repositorio

```
DIRA/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # CI/CD: build → Docker Hub → k3s → MLRun
├── docker/
│   ├── train/Dockerfile           # imagen dira-train (Python 3.11, xgboost-cpu)
│   ├── infer/Dockerfile           # imagen dira-infer (Python 3.11, FastAPI + /metrics)
│   ├── front/
│   │   ├── Dockerfile             # imagen dira-front (nginx)
│   │   └── 40-envsubst.sh         # inyecta API_URL en config.js al arrancar
│   └── drift/
│       └── Dockerfile             # imagen dira-drift (Evidently AI)
├── helm/
│   ├── dira/
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/
│   │       ├── configmap.yaml     # ConfigMap con variables de entrenamiento
│   │       ├── pvc.yaml           # data-pvc y model-pvc
│   │       ├── front-deployment.yaml
│   │       ├── front-service.yaml
│   │       └── drift-cronjob.yaml # CronJob horario de detección de deriva
│   └── monitoring/
│       ├── values-prometheus.yaml # kube-prometheus-stack (Prometheus+Grafana+Alertmanager)
│       └── dira-rules.yaml        # PrometheusRule: 7 alertas operativas y de modelo
├── src/
│   ├── backend/
│   │   ├── data_ingestion.py
│   │   ├── features.py
│   │   ├── train.py               # + handler mlrun_train() para MLRun
│   │   ├── mlrun_project.py       # orquestación del ciclo de vida
│   │   ├── infer.py               # DIRAPredictor (cloudpickle)
│   │   └── main_api.py            # FastAPI, OIP V2, métricas Prometheus, log inferencias
│   ├── frontend/
│   │   ├── index.html
│   │   ├── css/styles.css
│   │   └── js/
│   │       ├── config.js          # const API_URL = '${API_URL}' (sustituido por envsubst)
│   │       └── app.js
│   └── monitoring/
│       ├── drift_check.py         # Evidently + Pushgateway + feedback loop (GitHub Actions)
│       └── simulate_anomaly.py    # simulación de entorno anómalo
├── requirements_train.txt         # mlrun==1.10.0, xgboost-cpu, cloudpickle
├── requirements_infer.txt         # fastapi, uvicorn, cloudpickle, prometheus-fastapi-instrumentator
├── requirements_drift.txt         # evidently, prometheus-client, requests
└── INSTALL.md                     # este archivo
```