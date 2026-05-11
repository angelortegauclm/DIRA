"""
train.py

Este módulo se encarga del entrenamiento del modelo, es decir, de la construcción del pipeline de entrenamiento, la selección y configuración del modelo, 
y la ejecución del proceso de entrenamiento.
El modelo final seleccionado de hitos anteriores una vez realizados los distintos análisis con varios algoritmos y con el dataset tratado para el desbalanceo, 
es un model XGBoost con hiperparámetros optimizados mediante RandomizedSearchCV.

Variables de entorno necesarias para la ejecución del módulo:
   DATA_DIR      Ruta al directorio donde se encuentra el archivo CSV de entrada, por defecto "./data"
   DATA_FILENAME Nombre del archivo CSV de entrada, por defecto "diabetes_binary_health_indicators_BRFSS2015.csv"
   MODEL_PATH    Ruta donde se guardará el modelo entrenado, por defecto "./model/modelo_diabetes_DIRA.pkl"
   RANDOM_STATE  Semilla aleatoria, por defecto 42
   N_ITER        Iteraciones de RandomizedSearchCV, por defecto 25
   CV_FOLDS      Folds de validación cruzada, por defecto 5
"""

import argparse
import os
import sys
import time

import cloudpickle
import xgboost as xgb
from scipy.stats import loguniform, uniform, randint
from sklearn import set_config
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import make_pipeline

from data_ingestion import load_data
from features import construct_preprocessor, split_X_y

# Configuración de los valores por defecto para las variables de entorno
# BASE_DIR se utiliza para construir rutas relativas al proyecto, partimos de la carpeta donde se encuentra el fichero actual y subimos un nivel para situarnos en la raíz del proyecto.
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "/data")
DEFAULT_DATA_FILENAME = os.getenv("DATA_FILENAME", "diabetes_binary_health_indicators_BRFSS2015.csv")
DEFAULT_DATA_PATH = os.path.join(DEFAULT_DATA_DIR, DEFAULT_DATA_FILENAME)
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "/model/modelo_diabetes_DIRA.pkl")
DEFAULT_RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
DEFAULT_N_ITER = int(os.getenv("N_ITER", 25))
DEFAULT_CV_FOLDS = int(os.getenv("CV_FOLDS", 5))

# Función para parsear los argumentos de la línea de comandos. Si no se proporciona un argumento, se utiliza los valores por defecto definidos.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DIRA – Entrenamiento del modelo.")
    parser.add_argument(
        "--data-path", type=str, default=DEFAULT_DATA_PATH,
        help="Ruta completa al CSV. Si no se especifica, se compone de DATA_DIR + DATA_FILENAME.",
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        help="Ruta donde se guardará el pipeline con el preprocessor y el modelo entrenado. Env: MODEL_PATH",
    )
    parser.add_argument(
        "--random-state", type=int, default=DEFAULT_RANDOM_STATE,
        help="Semilla aleatoria. Env: RANDOM_STATE",
    )
    parser.add_argument(
        "--n-iter", type=int, default=DEFAULT_N_ITER,
        help="Iteraciones de RandomizedSearchCV. Env: N_ITER",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=DEFAULT_CV_FOLDS,
        help="Folds de validación cruzada. Env: CV_FOLDS",
    )
    return parser.parse_args()
###
# Función principal de entrenamiento. Carga los datos, construye el preprocesador, define el modelo y el pipeline,
#  realiza la búsqueda de hiperparámetros, entrena el modelo final con los mejores hiperparámetros y guarda el pipeline completo.
# Argumentos:
#   data_path: Ruta al CSV de entrada
#   model_path: Ruta donde se guardará el modelo entrenado
#   random_state: Semilla aleatoria para reproducibilidad
#   n_iter: Número de iteraciones para RandomizedSearchCV   
#   cv_folds: Número de folds para validación cruzada
# Retorna un diccionario con los resultados del entrenamiento, incluyendo el mejor modelo, los mejores hiperparámetros, el score de validación y el tiempo de entrenamiento.

def train (data_path: str, model_path: str, random_state: int = 42, n_iter: int = 25, cv_folds: int = 5) -> dict:
          
    # Configuramos scikit-learn para que devuelve DataFrames desde los transformadores
    set_config(transform_output="pandas")

    # Cargar los datos utilizando la función load_raw_data del módulo data_ingestion y separar en dos dataframe X e y.
    df = load_data(data_path)
    X, y = split_X_y(df)

    # Dividimos el dataset en train y test con una proporción de 80/20
    test_size = 0.2 
    stratify = y  # Etiqueta de la clase objetivo para la estrificación
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    #Imprimimos el número de muestras en train y test y el porcentaje de positivos en train para verificar que la división se ha realizado correctamente y que el dataset está estratificado.
    print("[train]--- Proporción de Diabetes en el dataset original ---")
    print(y.value_counts(normalize=True))

    print("\n[train]--- Proporción de Diabetes en el conjunto de Entrenamiento (Train) ---")
    print(y_train.value_counts(normalize=True))

    print("\n[train]--- Proporción de Diabetes en el conjunto de Prueba (Test) ---")
    print(y_test.value_counts(normalize=True))

    # Construir el preprocesador utilizando la función construct_preprocessor del módulo features,
    # se pasa el dataframe de entrenamiento para que pueda identificar las columnas y pueda definir las transformaciones que sean necesarias.
    preprocessor = construct_preprocessor(X_train)

    # APLICAMOS PREPROCESSOR
    # Ajustamos el preprocesador al dataset de entrenamiento actual
    X_tr_procesado_final = preprocessor.fit_transform(X_train)

    # aplicamos el transform del preprocesor al conjunto de test (usamos el X_test puro que separamos al principio)
    X_te_procesado_final = preprocessor.transform(X_test) 

    # El modelo que XGBoost no tiene un tratamiento específico para el desbalanceo, pero se puede configurar el hiperparámetro scale_pos_weight para que el modelo le de más peso 
    # a la clase minoritaria. Para ello se calcula el ratio entre la cantidad de negativos y positivos en el conjunto de entrenamiento y se pasa ese valor al hiperparámetro 
    # scale_pos_weight del modelo.
    ratio_peso = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"[train] scale_pos_weight para XGBoost: {ratio_peso:.3f}")

    # Definimos el param_grid para RandomizedSearchCV con los hiperparámetros que se van a optimizar y sus respectivos rangos de valores.
    xgb_model = xgb.XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        objective="binary:logistic",
        scale_pos_weight=ratio_peso,
    )
 
    param_distributions = {
        'n_estimators': randint(100, 401),
        'max_depth': randint(3, 9),
        'learning_rate': loguniform(0.01, 0.2),
        # uniform: Elige un decimal al azar en un rango lineal.
        'subsample': uniform(0.6, 0.4)      # uniform(loc, scale) -> de 0.6 a 1.0 (0.6 + 0.4)
    }

    # Validación cruzada estratificada con 5 folds (sobre el conjunto de entrenamiento)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


    # Creamos el RandomizedSearchCV
    # 'n_iter=25' es un buen equilibrio entre tiempo y calidad
    # 'scoring' puede ser 'average_precision' o 'f1'

    randomSearch = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_distributions,
            n_iter=n_iter, 
            scoring='average_precision', 
            cv=cv,
            verbose=0,
            n_jobs=-1,
            random_state=random_state
    )

    # vamos a calcular el tiempo que tarda en entrenar para tenerlo como una métrica de coste computacional
    print(f"\n[train] Iniciando RandomizedSearchCV ({n_iter} iter, {cv_folds} folds)...")
    inicio = time.time()
    # Ajustamos el modelo al dataset (X_tr_procesado, y_train)
    randomSearch.fit(X_tr_procesado_final, y_train)
    fin = time.time()
    training_time = fin - inicio 

    print(f"[train] Búsqueda de hiperparámetros completada en {training_time/60:.1f} min")
    print(f"[train] Mejor AP (CV): {randomSearch.best_score_:.4f}")
    print(f"[train] Mejores hiperparámetros: {randomSearch.best_params_}")

    # Evaluamos el modelo con los mejores hiperparámetros usando los datos de TEST que tenemos guardados
    mejor_modelo = randomSearch.best_estimator_
    y_pred_best_model = mejor_modelo.predict(X_te_procesado_final)
    y_pred_proba_best_model = mejor_modelo.predict_proba(X_te_procesado_final)[:, 1]  # Probabilidades de la clase positiva

    # Métricas de evaluación en el conjunto de test
    ap_test = average_precision_score(y_test, y_pred_proba_best_model)
    cm = confusion_matrix(y_test, y_pred_best_model)
    # Cálculos manuales basados en la matriz de confusión para obtener recall, precisión y f1, 
    # además de los valores de tn, fp, fn y tp para tener una visión completa del rendimiento del modelo.
    tn, fp, fn, tp = cm.ravel()
        
    recall = tp / (tp + fn)                                     # De todos los diabéticos reales que hay en el estudio, qué % se detectó?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0        # De todas las personas predichas como diabéticas, qué % era realmente diabético?
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
  
    # Guardamos las metricas de evaluación en un diccionario para tenerlas organizadas y poder imprimirlas de forma clara.
    metrics = {
        "ap_test": round(ap_test, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    # Imprimimos las métricas de evaluación en test para tener una visión clara del rendimiento del modelo en datos no vistos.
    print(f"[train] Métricas en test:")
    for k, v in metrics.items():
        print(f"         {k}: {v}")
 
    # Guardamos el pipeline completo (preprocesador + modelo) utilizando cloudpickle para poder cargarlo posteriormente en el módulo de inferencia.
    # Reconstruimos el preprocesador desde cero para garantizar que aprende solo de X_train.
    preprocessor_final = construct_preprocessor(X_train)

    # Creamos el pipeline
    pipeline_final_DIRA = make_pipeline(preprocessor_final, mejor_modelo)

    # Nos aseguramos que el preprocessor y el modelo creado se entrenan con el dataset original de entramiento
    pipeline_final_DIRA.fit(X_train, y_train)
    print("[train] Pipeline para el sistema DIRA entrenado y sincronizado con el dataset Original.")

    # Guardamos el objeto completo: Preprocesador + XGBoost
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        cloudpickle.dump(pipeline_final_DIRA, f)

    print(f"[train] Pipeline guardado como '{model_path}'")

    # Guardar muestra de referencia para drift detection (accesible desde model-pvc)
    ref_path = os.path.join(os.path.dirname(model_path), "reference_sample.csv")
    X_train.sample(n=min(5000, len(X_train)), random_state=random_state).to_csv(ref_path, index=False)
    print(f"[train] Muestra de referencia guardada en '{ref_path}'")

    return metrics

def mlrun_train(context):
    """Entry point para MLRun.

    MLRun inyecta 'context' (MLClientCtx) al lanzar el Job en Kubernetes.
    Esta función extrae los parámetros del contexto, llama a train() y
    registra en el tracking server de MLRun:
      - Parámetros de configuración del experimento (random_state, n_iter...)
      - Métricas de evaluación en test (AP, F1, recall, precision, matriz de confusión)
      - El artefacto del modelo (.pkl) para versionado y reproducibilidad
    """
    p = context.parameters
    data_path   = p.get("data_path",   DEFAULT_DATA_PATH)
    model_path  = p.get("model_path",  DEFAULT_MODEL_PATH)
    random_state = int(p.get("random_state", DEFAULT_RANDOM_STATE))
    n_iter       = int(p.get("n_iter",       DEFAULT_N_ITER))
    cv_folds     = int(p.get("cv_folds",     DEFAULT_CV_FOLDS))

    metrics = train(
        data_path=data_path,
        model_path=model_path,
        random_state=random_state,
        n_iter=n_iter,
        cv_folds=cv_folds,
    )

    # Parámetros de configuración del experimento
    context.log_results({
        "random_state": random_state,
        "n_iter":        n_iter,
        "cv_folds":      cv_folds,
    })

    # Métricas de evaluación en test
    context.log_results(metrics)

    # Artefacto: pipeline completo (preprocesador + XGBoost)
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    context.log_model(
        "dira-pipeline",
        body=model_bytes,
        model_file="modelo_diabetes_DIRA.pkl",
        framework="sklearn",
        metrics=metrics,
        labels={"dataset": "BRFSS2015", "target": "diabetes_binary"},
    )


## Función main para ejecutar el módulo de entrenamiento. Se parsean los argumentos, se llama a la función de entrenamiento y se imprimen los resultados.
if __name__ == "__main__":
    # Parsear los argumentos de la línea de comandos.
    args = parse_args()
    # Ejecutar la función de entrenamiento con los argumentos parseados.
    metrics = train(
        data_path=args.data_path,
        model_path=args.model_path,
        random_state=args.random_state,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds
    )
    sys.exit(0)
