"""
features.py

Este módulo se encarga de la ingeniería de características y de la construcción del pipeline de preprocesamiento, 
Contiene:
   - Definición de grupos de los grupos de variables según el tratamiento que necesitan
   - Función split_X_y: Separa el dataset cargado en dos dataframes: uno con las características (X) y otro con la variable objetivo (y).
   - Función bmi_clipper: limita el BMI a 60 para eliminar outliers extremos, puesto que en el análisis de datos de los hitos anteriores se observaron 
                        valores extremos de BMI que podrían afectar negativamente al modelo y dominarlo.
   - Función construct_preprocessor(): construye y devuelve el ColumnTransformer listo para ser encadenado en cualquier Pipeline de entrenamiento o inferencia

Solo define la lógica de preprocesamiento, sin incluir la carga de datos ni el entrenamiento del modelo.
"""

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)   

# Condiguración variable objetivo
TARGET_COLUMN = "Diabetes_binary"

# Eliminar variables sin varianza útil AnyHealthcare y CholCheck
DROP_COLUMNS = ['AnyHealthcare', 'CholCheck']

# Función para parsear los argumentos de la línea de comandos. Si no se proporciona un argumento, se utiliza los valores por defecto definidos.
def split_X_y(df: pd.DataFrame, target: str = TARGET_COLUMN):
    """Separa el dataset cargado en dos dataframes: uno con las características (X) y otro con la variable objetivo (y).
    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos cargados.
        target (str): El nombre de la columna que se utilizará como variable objetivo. Por defecto es 'Diabetes_binary'.

    Returns:
        tuple: Una tupla que contiene dos DataFrames: uno con las características (X) y otro con la variable objetivo (y).
    """
    # Se comprueba primero que la variable objetivo existe en el dataframe original antes de intentar separarla, si no existe lanza una excepción.
    if target not in df.columns:
        raise ValueError(f"[features] La columna objetivo '{target}' no existe en el dataset.")
    
    # Eliminamos la columna objetivo para generar el dataframe de característicias (X) y se guarda la columna objetivo en un dataframe separado (y).
    y = df[target]
    axis = "columns"  # Elimina etiquetas por columnas
    X = df.drop(target, axis=axis)

    return X, y



###
# Función para construir el ColumnTransformer de preprocesamiento.  
# Aplica el preprocesamiento adecuado a cada grupo de variables según su tipo y las necesidades detectadas en el EDA.
# Elimina las columnas AnyHealthcare y CholCheck al no aportar varianza útil.
# Devuelve un ColumnTransformer listo para ser incluido en cualquier Pipeline de entrenamiento o inferencia.
###

def construct_preprocessor(X: pd.DataFrame):
    """
    Construye y devuelve un ColumnTransformer que aplica el preprocesamiento adecuado a cada grupo de variables.
    Args:
        X (pd.DataFrame): El DataFrame que contiene las características (ya sin la variable objetivo).
    Returns:
        ColumnTransformer: Un ColumnTransformer que aplica el preprocesamiento adecuado a cada grupo de variables.
    """

    # Definida localmente para que cloudpickle embeba el bytecode en el .pkl
    # y el contenedor de inferencia no necesite importar features.py
    def bmi_clipper_function(X):
        return np.clip(X, a_min=None, a_max=60)

    # Eliminar variables sin varianza útil AnyHealthcare y CholCheck
    drop_columns = ['AnyHealthcare', 'CholCheck']

    # Creamos listas que agrupen las variables por el tipo exacto de preprocesamiento que van a recibir. 
    # Identificamos las columnas que solo tienen 0.0 y 1.0 y no incluimos en la lista ni AnyHealthcare, ni CholCheck, lo que hará que
    # se eliminen del dataset generado tras el preprocesamiento.
    features_binary = [
        col for col in X.columns 
        if col not in drop_columns and set(X[col].dropna().unique()) == {0.0, 1.0} 
    ]
    feature_standarScaler = ['GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
    feature_robustScaler = ['BMI']
    
    # Construimos los pipelines de preprocesamiento
    # Pipeline de preprocesamiento para variables que necesitan un escalador que trate mejor los outliers
    strategy = "median"  # Imputación de valores faltantes por la mediana
    imputer = SimpleImputer(strategy=strategy)
    transformer = FunctionTransformer(bmi_clipper_function, validate=False)
    scaler = RobustScaler()
    robust_scaler_transformer = make_pipeline(imputer, transformer, scaler)

    # Pipeline de preprocesamiento para variables que necesitan un escalado estandard
    strategy = "median"  # Imputación de valores faltantes por la mediana
    imputer = SimpleImputer(strategy=strategy)
    transformer = PowerTransformer(method="yeo-johnson")
    scaler = StandardScaler()
    standard_scaler_transformer = make_pipeline(imputer, transformer, scaler)

    # Pipeline de preprocesamiento para variables binarias
    strategy = "most_frequent"  # Imputación de valores faltantes por la moda
    imputer = SimpleImputer(strategy=strategy)
    binary_transformer = make_pipeline(imputer)
        
    # ColumnTransformer único
    preprocessor = make_column_transformer(
        # Variables con necesidad de escalado robusto
        (robust_scaler_transformer, feature_robustScaler),
        
        # Variables con necesidad de escalado estandard
        (standard_scaler_transformer, feature_standarScaler),

        # Variables binarias
        (binary_transformer, features_binary),
            
        remainder="drop",  # descartar columnas no especificadas en el preprocesamiento con esto eliminamos las columnas AnyHealthcare y CholCheck
        verbose_feature_names_out=False  # le decimos al transformer que no cambie el nombre de las columnas para que en la visualización tengamos los nombres originales
    )
    return preprocessor