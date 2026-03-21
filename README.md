# DIRA — Diabetes Intelligent Risk Assessment

DIRA (Diabetes Intelligent Risk Assessment) es un proyecto de inteligencia artificial orientado a la **estimación temprana del riesgo de diabetes tipo 2** mediante técnicas de aprendizaje automático aplicadas a indicadores de salud poblacional.

El objetivo del sistema no es realizar un diagnóstico médico, sino **apoyar la identificación de perfiles de riesgo en población adulta**, permitiendo priorizar estrategias preventivas y mejorar la toma de decisiones en contextos sanitarios.

Este proyecto se desarrolla en el marco del **Máster en Inteligencia Artificial de la Universidad de Castilla-La Mancha (UCLM)** dentro de la asignatura *Desarrollo e Integración de Servicios de IA*.

---

# Problema que aborda el proyecto

La diabetes tipo 2 constituye uno de los principales desafíos sanitarios actuales debido a su elevada prevalencia, su evolución progresiva y el impacto que genera en los sistemas de salud.

En muchos casos, la enfermedad se desarrolla durante años sin síntomas evidentes, lo que dificulta su detección temprana. Esta situación hace especialmente relevante la identificación de **personas con mayor probabilidad de desarrollar diabetes antes de que aparezcan manifestaciones clínicas claras**.

En este contexto, el proyecto DIRA explora cómo un sistema de inteligencia artificial puede analizar variables relacionadas con:

- hábitos de vida  
- estado general de salud  
- antecedentes médicos  
- características demográficas  

para **estimar el riesgo individual de diabetes o prediabetes** y apoyar estrategias de prevención.

---

# Dataset

El proyecto utiliza el **Diabetes Health Indicators Dataset**, derivado del sistema epidemiológico estadounidense **BRFSS (Behavioral Risk Factor Surveillance System)**.

Características principales del dataset:

- **253.680 registros**
- **21 variables predictoras**
- **1 variable objetivo**

Variable objetivo:

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

# Variables relevantes

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

# Enfoque del modelo

El problema se plantea como una **tarea de clasificación binaria** en la que el sistema debe distinguir entre:

- individuos sin diagnóstico de diabetes  
- individuos con prediabetes o diabetes  

El pipeline de machine learning incluye:

1. Exploración de datos (EDA)
2. Preprocesamiento del dataset
3. División en conjunto de entrenamiento y prueba
4. Entrenamiento de modelos supervisados
5. Evaluación del rendimiento
6. Interpretación del modelo

---

# Interpretabilidad del modelo (Explainable AI)

En aplicaciones sanitarias no basta con obtener predicciones precisas. También es necesario comprender **por qué el modelo toma determinadas decisiones**.

Por este motivo se incorporan técnicas de **Explainable Artificial Intelligence (XAI)**:

### SHAP (SHapley Additive Explanations)

Permite analizar la contribución de cada variable al resultado del modelo mediante valores de Shapley procedentes de la teoría de juegos.

Esto facilita:

- comprender qué variables influyen más en el modelo
- identificar patrones de riesgo
- interpretar el comportamiento global del sistema

### LIME (Local Interpretable Model-agnostic Explanations)

Permite explicar **predicciones individuales** aproximando el comportamiento del modelo en torno a un caso concreto.

Esto resulta especialmente útil en contextos clínicos donde los profesionales necesitan entender **por qué el sistema considera que un individuo presenta mayor riesgo**.

---

# Comparación con herramientas clínicas (benchmark)

Para contextualizar el rendimiento del modelo se utiliza como referencia conceptual el cuestionario **FINDRISC (Finnish Diabetes Risk Score)**.

FINDRISC es una herramienta ampliamente utilizada en Europa para estimar el riesgo de diabetes tipo 2 a partir de variables como:

- edad
- índice de masa corporal
- perímetro abdominal
- actividad física
- dieta
- antecedentes familiares

El objetivo de DIRA es analizar si los modelos de aprendizaje automático pueden **identificar patrones más complejos de riesgo** que los sistemas tradicionales basados en puntuación.

---

# Evaluación del modelo

Debido al desbalanceo del dataset, la evaluación del modelo no se basa únicamente en accuracy.

Se utilizan métricas más representativas:

- **F1-score**
- **Precision**
- **Recall (sensibilidad)**
- **Matriz de confusión**
- **Curva ROC**
- **Curva Precision-Recall**

Estas métricas permiten evaluar la capacidad del modelo para identificar correctamente individuos con riesgo.

---

# Tecnologías utilizadas

El proyecto ha sido desarrollado en **Python** utilizando las siguientes librerías:

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- imbalanced-learn  
- shap  
- lime  



