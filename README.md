# DIRA: Diabetes Intelligent Risk Assessment
DIRA (Diabetes Intelligent Risk Assessment), se centra en el análisis de indicadores de salud para evaluar el riesgo de diabetes. Forma parte del máster en Inteligencia Artificial de la UCLM.

El proyecto utiliza el conjunto de datos de indicadores de salud de la diabetes de la encuesta BRFSS 2015, que incluye indicadores de salud binarios para predecir el riesgo de diabetes o prediabetes.

## Dataset
* Fuente: Conjunto de datos de indicadores de salud relacionados con la diabetes
* Descripción: Contiene 253 680 filas y 22 columnas, incluida la variable objetivo Diabetes_binary (0: sin diabetes, 1: diabetes/prediabetes) y predictores como el IMC, la presión arterial, el tabaquismo, etc.
* Características principales:
 - Variables binarias (por ejemplo, HighBP, Smoker)
 - Variables ordinales (por ejemplo, Age, Education)
 - Variables continuas (por ejemplo, BMI, MentHlth)

## Requisitos
* Python 3.x
* Bibliotecas: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, dython