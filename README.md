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
- joblib  
