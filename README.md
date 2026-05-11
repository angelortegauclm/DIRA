# DIRA — Diabetes Intelligent Risk Assessment

DIRA (Diabetes Intelligent Risk Assessment) es un proyecto de inteligencia artificial orientado a la **estimación temprana del riesgo de diabetes tipo 2** mediante técnicas de aprendizaje automático aplicadas a indicadores de salud poblacional.

El objetivo del sistema no es realizar un diagnóstico médico, sino **apoyar la identificación de perfiles de riesgo en población adulta**, permitiendo priorizar estrategias preventivas y mejorar la toma de decisiones en contextos sanitarios.

Este proyecto se desarrolla en el marco del **Máster en Inteligencia Artificial de la Universidad de Castilla-La Mancha (UCLM)** dentro de la asignatura *Desarrollo e Integración de Servicios de IA*.

---

## Problema que aborda el proyecto

La diabetes tipo 2 constituye uno de los principales desafíos sanitarios actuales debido a su elevada prevalencia, su evolución progresiva y el impacto que genera en los sistemas de salud.

En muchos casos, la enfermedad se desarrolla durante años sin síntomas evidentes, lo que dificulta su detección temprana. Esta situación hace especialmente relevante la identificación de **personas con mayor probabilidad de desarrollar diabetes antes de que aparezcan manifestaciones clínicas claras**.

En este contexto, el proyecto DIRA explora cómo un sistema de inteligencia artificial puede analizar variables relacionadas con:

- hábitos de vida  
- estado general de salud  
- antecedentes médicos  
- características demográficas  

para **estimar el riesgo individual de diabetes o prediabetes** y apoyar estrategias de prevención.

---

## Dataset

El proyecto utiliza el **Diabetes Health Indicators Dataset**, derivado del sistema epidemiológico estadounidense **BRFSS (Behavioral Risk Factor Surveillance System)**.

Características principales del dataset:

- **253.680 registros**
- **21 variables predictoras**
- **1 variable objetivo**: `Diabetes_binary` (0 = sin diabetes/prediabetes, 1 = diabetes/prediabetes)

El conjunto de datos presenta un **importante desbalanceo de clases** (~14,5% positivo vs ~85,5% negativo), lo que condiciona tanto el preprocesamiento como la evaluación del modelo.

El conjunto de datos incluye indicadores relacionados con:

- índice de masa corporal  
- hipertensión  
- colesterol elevado  
- actividad física  
- hábitos alimentarios  
- consumo de alcohol y tabaco  
- percepción del estado de salud  
- variables sociodemográficas  

Estos indicadores permiten capturar distintas dimensiones del riesgo metabólico en población adulta.

---

## Variables relevantes

Entre los factores analizados destacan:

- **BMI** — Índice de masa corporal  
- **HighBP** — Hipertensión  
- **HighChol** — Colesterol elevado  
- **PhysActivity** — Actividad física  
- **Fruits / Veggies** — Consumo de frutas y verduras  
- **Stroke** — Antecedente de ictus  
- **HeartDiseaseorAttack** — Enfermedad cardiovascular previa  
- **GenHlth** — Percepción del estado de salud  
- **DiffWalk** — Dificultad para caminar  
- **Age** — Grupo de edad  

---

## Enfoque del modelo

El problema se plantea como una **tarea de clasificación binaria** en la que el sistema debe distinguir entre:

- individuos sin diagnóstico de diabetes  
- individuos con prediabetes o diabetes  

El pipeline de machine learning incluye:

1. Exploración de datos (EDA)
2. Preprocesamiento del dataset y tratamiento del desbalanceo de clases
3. División en conjunto de entrenamiento y prueba (80/20 estratificado)
4. Entrenamiento y comparación de modelos supervisados
5. Evaluación del rendimiento
6. Interpretación del modelo mediante XAI

### Modelos evaluados

Se han comparado los siguientes algoritmos bajo distintas técnicas de balanceo de datos:

- Logistic Regression
- Ridge Classifier
- Random Forest Classifier
- Balanced Random Forest Classifier
- XGBoost
- LightGBM
- CatBoost

### Técnicas de balanceo de clases

Para compensar el desbalanceo del dataset se han evaluado seis estrategias:

- **SMOTE** — sobremuestreo sintético de la clase minoritaria
- **SMOTE-NC** — variante de SMOTE que respeta variables categóricas y ordinales
- **Random Under-Sampler (RUS)** — submuestreo aleatorio de la clase mayoritaria
- **NearMiss-1, NearMiss-2, NearMiss-3** — submuestreo inteligente basado en distancias

### Modelo seleccionado

Tras comparar ~56 configuraciones (7 modelos × 8 datasets), el **modelo final seleccionado es LightGBM entrenado con el dataset balanceado mediante RUS**, por ofrecer el mejor equilibrio entre rendimiento y eficiencia computacional.

El modelo se ha guardado en `model/modelo_diabetes_DIRA.pkl` y está listo para su uso en producción.

---

## Interpretabilidad del modelo (Explainable AI)

En aplicaciones sanitarias no basta con obtener predicciones precisas. También es necesario comprender **por qué el modelo toma determinadas decisiones**.

Por este motivo se incorpora la técnica de **Explainable Artificial Intelligence (XAI)**:

### SHAP (SHapley Additive Explanations)

Permite analizar la contribución de cada variable al resultado del modelo mediante valores de Shapley procedentes de la teoría de juegos.

Se utiliza `TreeExplainer` optimizado para modelos de gradient boosting y genera:

- **Summary plot** — impacto medio de cada variable sobre las predicciones
- **Force plot** — explicación de predicciones individuales
- **Dependence plots** — análisis de interacciones entre variables (e.g., BMI × GenHlth)

Esto facilita:

- comprender qué variables influyen más en el modelo
- identificar patrones de riesgo
- interpretar el comportamiento global del sistema

---

## Referencia clínica: FINDRISC

Para contextualizar el rendimiento del modelo se utiliza como referencia conceptual el cuestionario **FINDRISC (Finnish Diabetes Risk Score)**, una herramienta ampliamente utilizada en Europa para estimar el riesgo de diabetes tipo 2.

DIRA analiza si los modelos de aprendizaje automático pueden **identificar patrones más complejos de riesgo** que los sistemas tradicionales basados en puntuación, aprovechando la mayor dimensionalidad del dataset BRFSS.

---

## Evaluación del modelo

Debido al desbalanceo del dataset, la evaluación del modelo no se basa únicamente en accuracy.

Se utilizan métricas más representativas:

- **F1-score**
- **Precision**
- **Recall (sensibilidad)**
- **Matriz de confusión**
- **Curva ROC y AUC**
- **Curva Precision-Recall**

Estas métricas permiten evaluar la capacidad del modelo para identificar correctamente individuos con riesgo.

---

## Interfaz web

DIRA incluye una interfaz web desplegada como contenedor nginx que permite introducir los parámetros del paciente y obtener el resultado de forma visual:

- Formulario con los 21 factores clínicos, de estilo de vida y demográficos
- Resultado con código de color: verde (riesgo < 30%), naranja (30–70%), rojo (> 70%)
- Desplegada como contenedor en k3s, accesible en `http://<node-ip>:30100`

---

## Infraestructura MLOps

El sistema completo se despliega en un clúster Kubernetes ligero mediante:

| Componente | Rol |
|---|---|
| **k3s** | Clúster Kubernetes local |
| **Helm** | Gestión del despliegue (ConfigMaps, PVCs, frontend, CronJobs) |
| **MLRun CE** | Ciclo de vida del modelo: Jobs, Model Registry, Serving |
| **Nuclio** | Runtime serverless para la función de inferencia |
| **FastAPI** | API de inferencia (protocolo Open Inference V2) |
| **nginx** | Servidor del frontend web |
| **Docker Hub** | Registro de imágenes |
| **GitHub Actions** | CI/CD: build → push → despliegue → entrenamiento |
| **Prometheus** | Recogida de métricas operativas y de modelo |
| **Grafana** | Dashboards de métricas operativas y deriva de datos |
| **Alertmanager** | Enrutamiento de alertas con notificaciones por Telegram |
| **Evidently AI** | Detección de deriva de datos en inferencias recientes |

El pipeline CI/CD detecta cambios en el código del modelo y lanza automáticamente un nuevo ciclo de entrenamiento y despliegue sin intervención manual.

---

## Monitorización y observabilidad

El sistema incluye una capa de observabilidad completa desplegada en el namespace `monitoring`:

**Métricas operativas** (recogidas por Prometheus desde `/metrics` del pod de inferencia):
- Peticiones por segundo y por minuto, latencia HTTP
- Tasa de errores 5xx
- Uso de CPU y memoria del pod

**Métricas del modelo** (contadores y distribuciones registradas en la propia API):
- `dira_predictions_total` — contador de predicciones por nivel de riesgo (BAJO / MODERADO / ALTO)
- `dira_prediction_probability` — histograma de probabilidades predichas
- `dira_model_loaded` — indicador de modelo cargado en memoria

**Detección de deriva de datos** (CronJob horario con Evidently AI):
- Compara la distribución de las inferencias recientes contra el dataset de entrenamiento
- Empuja el score de drift a Prometheus Pushgateway
- Si el score supera el umbral (0.20), dispara automáticamente un reentrenamiento vía GitHub Actions

**Alertas** (Alertmanager → Telegram):
- `DiraInferHighCPU` / `DiraInferHighMemory` — recursos operativos
- `DiraInferHighErrorRate` / `DiraInferDown` — disponibilidad del servicio
- `DiraDataDriftDetected` — deriva de datos detectada
- `DiraHighRiskSurge` — incremento anómalo de predicciones de alto riesgo
- `DiraModelNotLoaded` — modelo no cargado en memoria

**Simulación de entorno anómalo**:

```bash
python src/monitoring/simulate_anomaly.py \
  --url http://<node-ip>:<port> --n 200
```

---

## Estructura del proyecto

```
DIRA/
├── .github/workflows/ci-cd.yml   # CI/CD: build → Docker Hub → k3s → MLRun
├── docker/
│   ├── train/Dockerfile           # imagen dira-train (Python 3.11)
│   ├── infer/Dockerfile           # imagen dira-infer (Python 3.11, FastAPI + /metrics)
│   ├── front/Dockerfile           # imagen dira-front (nginx)
│   └── drift/Dockerfile           # imagen dira-drift (Evidently AI)
├── helm/
│   ├── dira/                      # chart DIRA: ConfigMap, PVCs, frontend, drift CronJob
│   └── monitoring/
│       ├── values-prometheus.yaml # kube-prometheus-stack (Prometheus+Grafana+Alertmanager)
│       └── dira-rules.yaml        # PrometheusRule con 7 reglas de alerta
├── src/
│   ├── backend/
│   │   ├── data_ingestion.py
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── mlrun_project.py
│   │   ├── infer.py
│   │   └── main_api.py            # + métricas Prometheus + log de inferencias
│   ├── frontend/
│   │   ├── index.html
│   │   ├── css/styles.css
│   │   └── js/
│   │       ├── config.js
│   │       └── app.js
│   └── monitoring/
│       ├── drift_check.py         # Evidently + Pushgateway + feedback loop
│       └── simulate_anomaly.py    # simulación de entorno anómalo
├── requirements_train.txt
├── requirements_infer.txt
└── requirements_drift.txt
```

---

## Tecnologías utilizadas

El proyecto ha sido desarrollado en **Python** utilizando las siguientes librerías:

- pandas  
- numpy  
- scipy  
- matplotlib  
- seaborn  
- plotly  
- scikit-learn  
- imbalanced-learn  
- xgboost  
- lightgbm  
- catboost  
- shap  
- dython  
- cloudpickle  
- evidently  
- prometheus-client  
- prometheus-fastapi-instrumentator  
