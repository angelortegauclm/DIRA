"""
data_ingestion.py

Este módulo se encarga de la ingesta de los datos, es decir, la lectura y preparación de los datos para su posterior procesamiento.
Proporciona una función para cargar los datos desde un archivo CSV y devolver un DataFrame listo para su uso.
La ruta al fichero CSV puede configurarse mediante argumento en la linea de comandos o mediante una variable de entorno.

Variable de entorno utilizada:
    DATA_PATH: Ruta al archivo CSV de origen, por defecto se utiliza "/data/diabetes_binary_health_indicators_BRFSS2015.csv"  
    DATA_FILENAME: Nombre del archivo CSV, por defecto se utiliza "diabetes_binary_health_indicators_BRFSS2015.csv"
Contiene:
    - Función parse_args(): Parsea los argumentos de la línea de comandos para configurar la ruta al archivo CSV de origen o si están vacíos selecciona el valor desde 
    la variable de entorno, deja en último lugar como valor, el valor por defecto definido.
    - Función validate_data_structure(df): Valida que el DataFrame cargado contiene las columnas esperadas para el modelo, tanto en estructura como en rango de valores.
     Si la validación falla, se lanza una excepción con un mensaje de error detallado y el resto del proceso de entrenamiento se detiene.
    - Función load_data(data_path): Carga el CSV desde data_path y devuelve un DataFrame de pandas con los datos cargados. Verifica que el archivo existe y manejo de errores. 
    Si el archivo no se encuentra o hay un error al cargarlo, se muestra un mensaje de error y el programa termina.
    - Función main: Ejecuta el módulo de ingesta de datos, parseando los argumentos, cargando los datos y mostrando las primeras filas del DataFrame resultante para verificar
      que se han cargado correctamente.
"""

from numpy import unique_values
import pandas as pd
import argparse
import os
import sys


# Configuración de las rutas para la carga de datos, utilizando variables de entorno o valores por defecto.
DEFAULT_DATA_DIR      = os.getenv("DATA_DIR", "/data")
DEFAULT_DATA_FILENAME = os.getenv("DATA_FILENAME", "diabetes_binary_health_indicators_BRFSS2015.csv")
DEFAULT_DATA_PATH     = os.path.join(DEFAULT_DATA_DIR, DEFAULT_DATA_FILENAME)
 

# Función para parsear los argumentos de la línea de comandos. Si no se proporciona un argumento, se utiliza los valores por defecto definidos.
def parse_args() -> argparse.Namespace:

    """
    Parsea los argumentos de la línea de comandos para configurar la ruta al archivo CSV de origen.
    Si no se proporciona un argumento, se utiliza la ruta por defecto definida en DEFAULT_DATA_PATH.
    
    Returns:    
        argparse.Namespace: Un objeto que contiene los argumentos parseados.  
    """

    parser = argparse.ArgumentParser(
        description="DIRA – Carga de datos desde fichero CSV"
    )

    # Se pasa como argumento la ruta informada por comando "--data-path" o coge la variable DEFAULT_DATA_PATH, que ya ha recogido los valores de las variables de entorno
    # DATA_DIR y DATA_FILENAME o si estas no están definidas utilizar sus valores por defecto. De esta forma, se da flexibilidad para configurar la ruta al archivo CSV 
    # de origen mediante diferentes métodos (argumento en línea de comandos o variables de entorno).
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.getenv(DEFAULT_DATA_PATH),
        help="Ruta al archivo CSV de origen. "
    )
    return parser.parse_args()

### 
# Función para validar que la estructura de los datos recibidos es la esperada así como el rango de su contenido, es decir, que contienen las columnas necesarias para el modelo.
# Si la validación falla, se lanza una excepción con un mensaje de error detallado y el resto del proceso de entrenamiento se detiene.
# Los rangos para el dataset original son los siguientes
#   - Variables binarias (0/1): Diabetes_binary, HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex.
#   - Variables numéricas ordinales: GenHlth (1–5), Age (1–13), Education (1–6), Income (1–8), MentHlth (0-30), PhysHlth(0-30)
#   - Variables numéricas continuas: BMI (10-150, aunque se aplicará un recorte a 60 en el preprocesamiento para eliminar outliers extremos)
###
# Lo primero que definimos son las colimnas que se esperan en el dataset, tanto las características (features) como la variable objetivo (target).


# Definimos las columnas binarias, para validar el dataset en estructura y contenido
BINARY_COLUMNS = [
    'Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'DiffWalk', 'Sex'
]

# Definimos los rangos esperados para las columnas numéricas, para validar el dataset en estructura y contenido
NUMERIC_RANGES = {
    'BMI': (10.0, 150.0),
    'GenHlth': (1.0, 5.0),
    'MentHlth': (0.0, 30.0),
    'PhysHlth': (0.0, 30.0),
    'Age': (1.0, 13.0),
    'Education': (1.0, 6.0),
    'Income': (1.0, 8.0)
}

# Todas las columnas que el CSV DEBE tener para que el proyecto funcione (suma de binarias + numéricas)
EXPECTED_COLUMNS = BINARY_COLUMNS + list(NUMERIC_RANGES.keys())

def validate_data_structure(df: pd.DataFrame):
    """
    Valida que el DataFrame cargado contiene las columnas esperadas para el modelo.
    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos cargados.
    Raises:
        ValueError: Si alguna de las columnas esperadas no está presente en el DataFrame.
    """

    # Comprobamos primero que la estructura es correcta y que el dataframe contiene todas las columnas que se esperan
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"[data_ingestion] El dataset cargado no contiene las siguientes columnas esperadas: {missing_columns}\n"
            f"El dataset debe contener exactamente las siguientes columnas: {EXPECTED_COLUMNS}"
        )
    else:
        print("[data_ingestion] Validación de esquema superada: todas las columnas requeridas están presentes.")
    
    # Comprobamos que las columnas binarias solo contienen 0.0 y 1.0, si no es así, se lanza una excepción.
    valores_permitidos = {0.0, 1.0}
    for col in BINARY_COLUMNS:
        valores_unicos = set(df[col].dropna().unique())
        if not valores_unicos.issubset(valores_permitidos):
            raise ValueError(
                f"[data_ingestion] La columna '{col}' contiene valores no binarios {valores_permitidos} y contiene: {valores_unicos}\n"
            )

    # Comprobamos que las columnas numéricas están en su rango.
    # No tenemos en cuenta los valores nulos, porque el modelo puede manejar nulos en el preprocesamiento mediante imputación y no queremos que la validación falle por eso, 
    # solo queremos asegurarnos de que los valores presentes estén dentro del rango esperado.
    for col, (min_val, max_val) in NUMERIC_RANGES.items():
        valores_fuera_rango = df[(df[col] < min_val) | (df[col] > max_val)]
        if not valores_fuera_rango.empty:
            raise ValueError(
                f"[data_ingestion] La columna '{col}' contiene valores fuera del rango esperado [{min_val}, {max_val}].\n"
                f"Valores encontrados: {df[col].dropna().unique()}"
            )
###
# Función para cargar los datos desde un archivo CSV. Verifica que el archivo existe y manejo de errores. 
# Devuelve un DataFrame de pandas con los datos cargados.
# Si el archivo no se encuentra o hay un error al cargarlo, se muestra un mensaje de error y el programa termina.
###

def load_data(data_path) -> pd.DataFrame:
    """
    Carga el CSV desde *data_path* y devuelve un DataFrame de pandas con los datos cargados.

    Args:
        data_path (str): La ruta al archivo CSV que contiene los datos.

    Returns:
        pd.DataFrame: Un DataFrame que contiene los datos cargados.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"[data_ingestion] No se encontró el archivo de datos en: {data_path}\n"
            "La ruta debe configurarse con --data-path o la variable de entorno DATA_PATH."
        )
    print(f"[data_ingestion] Cargando datos desde: {data_path}")
    try:
        data = pd.read_csv(data_path)
        # Validamos el dataset, Si falla, saltará al except y se frena la ejecución
        validate_data_structure(data)
        print(f"[data_ingestion] Dataset cargado y validado: {data.shape[0]} filas x {data.shape[1]} columnas.")
        return data
    except Exception as e:
        print(f"[data_ingestion] Se ha producido un error al cargar los datos: {e}")
        sys.exit(1) # Finaliza el proceso con un código de error para indicar que la carga de datos ha fallado.

### Función main para ejecutar el módulo de ingesta de datos. ###
if __name__ == "__main__":
    # Parsear los argumentos de la línea de comandos.
    args = parse_args()
    # Ruta al archivo CSV que contiene los datos.
    data_path = args.data_path      
    # Cargar los datos utilizando la función load_data.
    data = load_data(data_path)
    # Se verifica que los datos se han cargado de forma correcta y se muestran las primeras filas.
    if data is not None:
        print(data.head())
    sys.exit(0)