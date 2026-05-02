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

# Importar en el containerd de k3s (necesario porque k3s no usa el daemon Docker)
docker save dira-train:latest | sudo k3s ctr images import -
docker save dira-infer:latest  | sudo k3s ctr images import -

# Verificar que están disponibles para k3s
sudo k3s ctr images ls | grep dira
```

> **Nota sobre paquetes Python:** el Dockerfile de entrenamiento instala `xgboost-cpu`
> (variante sin CUDA), no `xgboost`. Si cambias la imagen base a una con GPU deberás
> ajustar también `requirements_train.txt`.

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

El chart DIRA crea el namespace, el ConfigMap y los dos PVCs (`data-pvc` y `model-pvc`).

```bash
# Crear namespace
kubectl create namespace dira

# Instalar el chart
helm install dira helm/dira --namespace dira

# Verificar
kubectl get configmap,pvc -n dira
```

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
python src/mlrun_project.py
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

> **Nota sobre pythonpath:** el Job configura `spec.pythonpath = "/app/src"` para que
> los imports relativos dentro del contenedor (`from data_ingestion import ...`) se
> resuelvan correctamente. Sin esta configuración el Job falla con `ModuleNotFoundError`.

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

## 11. Reentrenar el modelo

Para lanzar un nuevo ciclo de entrenamiento con parámetros distintos:

```bash
# Opción A: sobreescribir variables de entorno antes de ejecutar
N_ITER=50 CV_FOLDS=10 python src/mlrun_project.py

# Opción B: actualizar el ConfigMap vía Helm y relanzar
helm upgrade dira helm/dira --namespace dira --set config.nIter=50
python src/mlrun_project.py
```

Cada ejecución genera un nuevo Run en MLRun con su propio UID, métricas y artefacto versionado. El historial completo queda disponible en la UI.

---

## 12. Configurar el pipeline CI/CD (GitHub Actions)

El pipeline automatiza: build de imágenes → push a Docker Hub → despliegue en k3s → entrenamiento MLRun si el código del modelo cambia.

### 12.1 Instalar el runner self-hosted en la máquina con k3s

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

### 12.2 Crear los secretos en GitHub

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

### 12.3 Comportamiento del pipeline

```
push a main
    │
    ├─► Job build (GitHub runner)
    │       ├─ docker build dira-train  → push <user>/dira-train:<sha> + :latest
    │       └─ docker build dira-infer  → push <user>/dira-infer:<sha> + :latest
    │
    └─► Job deploy (self-hosted runner en k3s)  [necesita build]
            ├─ docker pull + k3s ctr images import  (ambas imágenes)
            ├─ helm upgrade --install dira           (ConfigMap + PVCs)
            ├─ [si src/train.py|features.py|data_ingestion.py|requirements_train.txt cambió
            │   O se activó manualmente]
            │       ├─ pip install mlrun==1.10.0     (SDK en el runner)
            │       └─ python src/mlrun_project.py  (lanza Job de entrenamiento + serving)
            └─ [tras entrenamiento] curl /v2/health/ready  (health check)
```

### 12.4 Forzar un entrenamiento sin cambiar código

```
GitHub → Actions → DIRA CI/CD → Run workflow → ✓ Forzar lanzamiento del entrenamiento MLRun → Run
```

---

## 13. Resolución de problemas conocidos

### `ModuleNotFoundError: No module named 'data_ingestion'` en el Job de MLRun

El Job corre con `WORKDIR=/app` pero los módulos están en `/app/src`. La configuración
`train_fn.spec.pythonpath = "/app/src"` en `mlrun_project.py` resuelve esto. Verificar
que esa línea está presente antes de relanzar.

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

# Obtener MLRUN_DBPATH actual
kubectl get svc -n mlrun mlrun-api -o jsonpath='{.spec.ports[0].nodePort}'

# Logs del Job de entrenamiento más reciente
kubectl logs -n dira -l mlrun/class=job --tail=100

# Actualizar parámetros del chart DIRA
helm upgrade dira helm/dira --namespace dira --set config.nIter=50

# Desinstalar todo
helm uninstall dira     --namespace dira
helm uninstall mlrun    --namespace mlrun
sudo k3s-uninstall.sh
```

---

## Estructura del repositorio

```
DIRA/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # pipeline CI/CD: build → Docker Hub → k3s → MLRun
├── docker/
│   ├── train/Dockerfile        # imagen dira-train (Python 3.11, xgboost-cpu)
│   └── infer/Dockerfile        # imagen dira-infer (Python 3.12, FastAPI)
├── helm/
│   └── dira/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── configmap.yaml  # ConfigMap con variables de entorno
│           └── pvc.yaml        # data-pvc y model-pvc
├── src/
│   ├── data_ingestion.py
│   ├── features.py
│   ├── train.py                # + handler mlrun_train() para MLRun
│   ├── mlrun_project.py        # orquestación del ciclo de vida
│   ├── infer.py
│   └── main_api.py
├── requirements_train.txt      # mlrun==1.10.0, xgboost-cpu==3.2.0
├── requirements_infer.txt
└── INSTALL.md                  # este archivo
```