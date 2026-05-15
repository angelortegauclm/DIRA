# DIRA — Diabetes Intelligent Risk Assessment

DIRA es un sistema MLOps de producción para la **estimación temprana del riesgo de diabetes tipo 2** mediante aprendizaje automático aplicado a indicadores de salud poblacional.

El objetivo no es realizar un diagnóstico médico sino **apoyar la identificación de perfiles de riesgo en población adulta**, permitiendo priorizar estrategias preventivas. Incluye un pipeline completo de entrenamiento, despliegue, monitorización y reentrenamiento automático.

Desarrollado en el marco del **Máster en Inteligencia Artificial de la UCLM** dentro de la asignatura *Desarrollo e Integración de Servicios de IA*.

---

## Arquitectura general

```
                        ┌─────────────────┐
                        │  Usuario final  │
                        │  Navegador web  │
                        └────────┬────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Frontend Nginx        │
                    │   :30100                │
                    └────────────┬────────────┘
                                 │ HTTP
                    ┌────────────▼────────────┐
                    │   API FastAPI (Nuclio)   │
                    │   KServe v2 · :31995    │
                    │   /metrics · /infer     │
                    └──────┬─────────┬────────┘
                           │         │
              ┌────────────▼──┐  ┌───▼──────────────┐
              │  LightGBM     │  │  inference_log   │
              │  model.pkl    │  │  (model-pvc)     │
              └───────────────┘  └───────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  CronJob drift      │
                              │  Evidently AI       │
                              │  cada 5 min         │
                              └──────────┬──────────┘
                                         │ push métricas
                    ┌────────────────────▼────────────────────┐
                    │              Prometheus                  │
                    │   scrape · alertas · series temporales  │
                    └──────────┬──────────────┬───────────────┘
                               │              │
              ┌────────────────▼──┐    ┌──────▼──────────┐
              │    Grafana        │    │  Alertmanager   │
              │    Dashboards     │    │  → Telegram     │
              └───────────────────┘    └─────────────────┘
```

---

## Dataset

**Diabetes Health Indicators Dataset** — BRFSS (Behavioral Risk Factor Surveillance System), EE.UU. 2015.

| Característica | Valor |
|---|---|
| Registros | 253.680 |
| Variables predictoras | 21 |
| Variable objetivo | `Diabetes_binary` (0/1) |
| Prevalencia de diabetes | ~13.9% |
| Desbalanceo | ~86% negativo / ~14% positivo |

Las 21 variables cubren: IMC, hipertensión, colesterol, actividad física, alimentación, alcohol, tabaco, historial cardiovascular, salud mental/física percibida, dificultad para caminar, sexo, edad, educación e ingresos.

---

## Modelo

### Selección

Se compararon ~56 configuraciones (7 algoritmos × 8 estrategias de balanceo):

**Algoritmos:** Logistic Regression, Ridge, Random Forest, Balanced Random Forest, XGBoost, LightGBM, CatBoost

**Balanceo:** SMOTE, SMOTE-NC, Random Under-Sampler, NearMiss-1/2/3

**Modelo final:** LightGBM + Random Under-Sampler — mejor equilibrio rendimiento/eficiencia.

### Entrenamiento en producción

```
train.py
  ├── RandomizedSearchCV (25 iter, 5-fold estratificado)
  ├── Guarda modelo_diabetes_DIRA.pkl en model-pvc
  ├── Guarda reference_sample.csv (5.000 filas) para drift detection
  └── Registra métricas y artefactos en MLRun Model Registry
```

### Clasificación de riesgo

| Probabilidad | Nivel | Color |
|---|---|---|
| < 0.30 | BAJO RIESGO | Verde |
| 0.30 – 0.70 | RIESGO MODERADO | Amarillo |
| > 0.70 | ALTO RIESGO | Rojo |

### Interpretabilidad (XAI)

Se usa **SHAP** (`TreeExplainer`) para explicar las predicciones:
- Summary plot — variables más influyentes globalmente
- Force plot — explicación de predicciones individuales
- Dependence plots — interacciones entre variables (e.g., BMI × GenHlth)

---

## Infraestructura

Desplegado en **k3s single-node** con Helm. Nodo: `192.168.1.131`.

| Componente | Tecnología | Namespace |
|---|---|---|
| Frontend web | Nginx | `dira` |
| API de inferencia | FastAPI + Nuclio | `mlrun` |
| Entrenamiento | MLRun Jobs | `mlrun` |
| Drift detection | CronJob (Evidently) | `mlrun` |
| Métricas | Prometheus + Pushgateway | `monitoring` |
| Dashboards | Grafana | `monitoring` |
| Alertas | Alertmanager → Telegram | `monitoring` |
| CI/CD | GitHub Actions (self-hosted) | — |

### Volúmenes persistentes

| PVC | Tamaño | Contenido |
|---|---|---|
| `data-pvc` | 1 Gi | Dataset CSV BRFSS |
| `model-pvc` | 500 Mi | `modelo_diabetes_DIRA.pkl`, `inference_log.csv`, `reference_sample.csv` |

---

## CI/CD

Pipeline en **GitHub Actions** con runner self-hosted en el mismo nodo k3s.

```
Push a main ──► Job 1: Build & Push
                  ├── dira-train:latest + :<sha>
                  ├── dira-infer:latest + :<sha>
                  └── dira-front:latest + :<sha>

             ──► Job 2: Deploy to k3s
                  ├── Sincroniza DIRA_GH_PAT → Secret Kubernetes
                  ├── Importa imágenes en containerd (k3s no usa Docker)
                  ├── helm upgrade dira
                  └── Si cambia train.py/features.py:
                        ├── MLRun lanza job de entrenamiento
                        └── Health check API de inferencia
```

**Secrets necesarios en GitHub:**

| Secret | Valor |
|---|---|
| `DOCKERHUB_USERNAME` | Usuario Docker Hub |
| `DOCKERHUB_TOKEN` | Token Docker Hub |
| `MLRUN_DBPATH` | URL interna del API MLRun |
| `DIRA_GH_PAT` | PAT con scope `workflow` (feedback loop) |

---

## Monitorización

### Métricas recogidas

| Fuente | Métricas |
|---|---|
| API inferencia (scrape directo) | `http_requests_total`, `http_request_duration_*`, `dira_predictions_total`, `dira_model_loaded`, `dira_prediction_probability` |
| Pushgateway (drift CronJob) | `dira_data_drift_score`, `dira_column_drift_score{column}` |
| kube-state-metrics | `container_cpu_usage_seconds_total`, `container_memory_working_set_bytes` |

### Dashboard Grafana

5 secciones en `http://192.168.1.131:30030` (admin / *ver values-secrets.yaml*):

1. **Resumen del sistema** — modelo cargado, peticiones/min, latencia P95, score drift
2. **Tráfico HTTP** — peticiones por estado, latencia P50/P95/P99
3. **Predicciones del modelo** — totales y tasa por nivel de riesgo
4. **Deriva de datos** — evolución temporal del score + top 10 columnas derivadas
5. **Recursos del pod** — CPU y memoria de dira-infer

### Alertas (Alertmanager → Telegram)

| Grupo | Alerta | Condición |
|---|---|---|
| Disponibilidad | `DiraInferDown` | Pod caído > 2 min |
| Disponibilidad | `DiraModelNotLoaded` | Modelo sin cargar > 2 min |
| Calidad HTTP | `DiraInferHighErrorRate` | > 5% errores 5xx en 5 min |
| Calidad HTTP | `DiraInferHighLatency` | P95 > 500 ms en 5 min |
| Recursos | `DiraInferHighCPU` | > 80% CPU en 5 min |
| Recursos | `DiraInferHighMemory` | > 700 MB RAM en 5 min |
| Modelo | `DiraDriftNormalizado` | Drift < 20% estable 10 min → `ℹ️` |
| Modelo | `DiraDataDriftWarning` | Drift 20–30% → `🚨 warning` |
| Modelo | `DiraDataDriftCritical` | Drift > 30% → `🚨 critical` + reentrenamiento |
| Modelo | `DiraHighRiskSurge` | > 50% predicciones alto riesgo en 10 min |

---

## Detección de deriva y feedback loop

El CronJob `dira-drift-check` se ejecuta cada hora (configurable):

```
1. Carga reference_sample.csv — 5.000 filas del último entrenamiento
2. Carga las últimas 500 inferencias de inference_log.csv (ventana 24h)
3. Evidently AI compara distribuciones columna a columna:
   • Variables continuas   → test Kolmogorov-Smirnov
   • Variables categóricas → test Chi-cuadrado
4. Empuja métricas al Pushgateway:
   • dira_data_drift_score          (score global 0.0–1.0)
   • dira_column_drift_score{column} (score por variable)
5. Si drift_score > 0.30:
   → workflow_dispatch en GitHub Actions → reentrenamiento automático
```

**Umbrales de deriva:**

| Score | Significado | Acción |
|---|---|---|
| < 20% | Normal | Ninguna |
| 20–30% | Deriva leve | Alerta warning, vigilar |
| > 30% | Deriva crítica | Alerta critical + reentrenamiento |

---

## Herramientas de prueba

```bash
# Tráfico realista (distribuciones BRFSS reales) — drift score < 20%
python src/monitoring/traffic_generator.py \
  --url http://192.168.1.131:31995 \
  --mode production --duration 0 --rps 2

# Tráfico sintético con perfiles extremos — drift score ~66%
python src/monitoring/traffic_generator.py \
  --url http://192.168.1.131:31995 \
  --mode synthetic --duration 300 --rps 3 --mix 60:25:15

# Inyección de deriva artificial para validar el pipeline completo
python src/monitoring/simulate_anomaly.py \
  --url http://192.168.1.131:31995 --duration 120
```

---

## Estructura del proyecto

```
DIRA/
├── .github/workflows/ci-cd.yml        # CI/CD: build → push → deploy → train
├── docker/
│   ├── train/Dockerfile               # dira-train  (Python 3.11, LightGBM)
│   ├── infer/Dockerfile               # dira-infer  (FastAPI, prometheus-client)
│   ├── front/Dockerfile               # dira-front  (Nginx + envsubst)
│   └── drift/Dockerfile               # dira-drift  (Evidently AI)
├── helm/
│   ├── dira/
│   │   ├── Chart.yaml
│   │   ├── values.yaml                # Parámetros del chart
│   │   └── templates/
│   │       ├── configmap.yaml         # Variables de entorno del modelo
│   │       ├── pvc.yaml               # data-pvc y model-pvc
│   │       ├── front-deployment.yaml  # Deployment del frontend
│   │       ├── front-service.yaml     # NodePort :30100
│   │       └── drift-cronjob.yaml     # CronJob de drift en namespace mlrun
│   └── monitoring/
│       ├── values-prometheus.yaml     # kube-prometheus-stack completo
│       ├── values-secrets.yaml        # Credenciales (gitignoreado)
│       ├── values-secrets.yaml.example
│       ├── dira-rules.yaml            # PrometheusRule — 10 alertas
│       └── grafana-dashboard-dira.yaml # ConfigMap con dashboard JSON
├── src/
│   ├── backend/
│   │   ├── features.py                # Definición de las 21 variables
│   │   ├── data_ingestion.py          # Carga y preprocesamiento del dataset
│   │   ├── train.py                   # Entrenamiento + MLRun + reference_sample
│   │   ├── infer.py                   # Lógica de predicción y log
│   │   ├── main_api.py                # FastAPI KServe v2 + métricas Prometheus
│   │   └── mlrun_project.py           # Orquestación MLRun (train + deploy)
│   ├── frontend/
│   │   ├── index.html
│   │   ├── css/styles.css
│   │   └── js/
│   │       ├── config.js              # API_URL (sustituido por envsubst)
│   │       └── app.js                 # Formulario + llamada API + resultado
│   └── monitoring/
│       ├── drift_check.py             # Evidently + Pushgateway + feedback loop
│       ├── traffic_generator.py       # Generador de tráfico (modo production/synthetic)
│       └── simulate_anomaly.py        # Inyección de deriva artificial
├── requirements_train.txt
├── requirements_infer.txt
├── requirements_drift.txt
├── INSTALL.md                         # Guía de instalación paso a paso
└── DIRA.ipynb                         # Notebook EDA + comparación de modelos
```

---

## URLs de acceso

| Servicio | URL | Credenciales |
|---|---|---|
| Frontend | `http://192.168.1.131:30100` | — |
| API inferencia | `http://192.168.1.131:31995` | — |
| Grafana | `http://192.168.1.131:30030` | admin / *values-secrets.yaml* |
| Prometheus | `http://192.168.1.131:30091` | — |
| Alertmanager | `http://192.168.1.131:30093` | — |

---

## Tecnologías

| Categoría | Tecnologías |
|---|---|
| ML / Data | Python, pandas, numpy, scikit-learn, LightGBM, XGBoost, CatBoost, imbalanced-learn, SHAP |
| Serving | FastAPI, prometheus-fastapi-instrumentator, KServe v2 |
| MLOps | MLRun CE, Nuclio, Docker, Docker Hub, Helm, k3s |
| CI/CD | GitHub Actions, self-hosted runner |
| Monitorización | Prometheus, Grafana, Alertmanager, Prometheus Pushgateway |
| Drift detection | Evidently AI, prometheus-client |
| Infraestructura | Kubernetes (k3s), Nginx, containerd |
