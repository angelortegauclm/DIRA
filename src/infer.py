"""
infer.py

Carga el pipeline con el preprocessor y el modelo entrenados y expone el método predict() para inferencia.

Variables de entorno utilizadas:
    MODEL_PATH    Ruta al archivo .pkl del modelo
    UMBRAL_BAJO   Umbral de probabilidad para clasificar como riesgo bajo (predicción negativa)
    UMBRAL_ALTO   Umbral de probabilidad para clasificar como riesgo alto (predicción positiva)
"""

import argparse
import os
import sys
 
import joblib
import numpy as np
import pandas as pd

# Ruta por defecto al modelo entrenado, configurable mediante variable de entorno MODEL_PATH
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "/model/modelo_diabetes_DIRA.pkl")

# Definición de umbrales de riesgo para la clasificación de la predicción de diabetes riesgo bajo y riesgo alto, configurables mediante variable de entorno UMBRAL_BAJO y UMBRAL_ALTO
UMBRAL_BAJO = float(os.getenv("UMBRAL_BAJO", "0.30"))
UMBRAL_ALTO = float(os.getenv("UMBRAL_ALTO", "0.70"))




# Clase DIRAPredictor
class DIRAPredictor:
    """
    Interfaz de inferencia para el modelo DIRA.
 
    Esta clase carga el pipeline de preprocesamiento y el modelo entrenado desde disco y proporciona un método predict() para realizar inferencia sobre nuevos datos de pacientes.
    El método predict() devuelve un DataFrame con la probabilidad de diabetes, la predicción binaria, el nivel de riesgo y la acción recomendada para cada paciente. 
    
    """
 
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._pipeline = None
        # Las necesitamos para la API posteriormente
        self.expected_columns = self.model.feature_names_in_.tolist()
 
    def load(self):
        """Carga el pipeline desde disco"""
        if self._pipeline is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f" [infer] No se encontró el modelo en: {self.model_path}\n"
                    "La ruta debe configurarse por comando con --model-path o la variable de entorno MODEL_PATH.\n"
                )
            print(f"[infer] Cargando modelo desde: {self.model_path}")
            self._pipeline = joblib.load(self.model_path)
            print("[infer] Modelo cargado.")
        return self
 
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza la predicción sobre el dataframe que recibe con los datos de uno o más pacientes y devuelve un dataframe con la probabilidad de diabetes, la predicción binaria, 
        el nivel de riesgo y la acción recomendada para cada paciente.
 
        Parámetros
        ----------
        X : pd.DataFrame
            Una o más filas con las mismas columnas que el dataset de entrenamiento.
           
 
        Retorna
        -------
        pd.DataFrame con columnas:
            - prob_diabetes   : probabilidad de pertenecer a la clase positiva [0, 1]
            - prediccion      : 0 (No diabético) / 1 (Diabético/Pre-diabético)
            - nivel_riesgo    : "BAJO" | "MODERADO" | "ALTO"
            - accion          : recomendación clínica de triaje
        """
    
        self.load()  # cargar el pipeline con el preprocesamiento y el modelo entrenado
 
        probs = self._pipeline.predict_proba(X)[:, 1]
        preds = self._pipeline.predict(X)
 
        resultados = []
        for prob, pred in zip(probs, preds):
            if prob < UMBRAL_BAJO:
                nivel = "BAJO RIESGO (Verde)"
                accion = "Mantener hábitos saludables. Revisión preventiva en 2 años."
            elif prob < UMBRAL_ALTO:
                nivel = "RIESGO MODERADO / SEGUIMIENTO (Amarillo"
                accion = "Seguimiento en 6 meses y plan de hábitos saludables."
            else:
                nivel = "ALTO RIESGO / PRIORITARIO (Rojo)"
                accion = "Derivación inmediata para pruebas diagnósticas de confirmación."
 
            resultados.append({
                "prob_diabetes": round(float(prob), 4),
                "prediccion": int(pred),
                "nivel_riesgo": nivel,
                "accion": accion,
            })
 
        return pd.DataFrame(resultados)
 
    # Función para obtener solo las probabilidades de la clase positiva (diabetes).
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve las probabilidades de la clase positiva."""
        self.load()
        return self._pipeline.predict_proba(X)[:, 1]


# Función para parsear los argumentos de la línea de comandos. Si no se proporciona un argumento, se utiliza los valores por defecto definidos.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DIRA – Inferencia.")
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        help="Ruta al pipeline serializado (.pkl). Env: MODEL_PATH",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    modelo_DIRA = DIRAPredictor(model_path=args.model_path)

    # Simulamos la inferencia con dos pacientes nuevos que contienen datos que el modelo no ha visto anteriormente
    datos_pacientes = pd.DataFrame([
        {
            'HighBP': 1.0,              # Tiene presión alta
            'HighChol': 1.0,            # Tiene colesterol alto
            'CholCheck': 1.0,
            'BMI': 35.0,                # Obesidad (Clase II)
            'Smoker': 1.0,
            'Stroke': 0.0,
            'HeartDiseaseorAttack': 1.0,
            'PhysActivity': 0.0,        # No hace ejercicio
            'Fruits': 0.0,
            'Veggies': 1.0,
            'HvyAlcoholConsump': 1.0,   # Bebedor habitual
            'AnyHealthcare': 1.0,
            'NoDocbcCost': 0.0,
            'GenHlth': 4.0,             # Salud percibida: Mala
            'MentHlth': 10.0,
            'PhysHlth': 15.0,
            'DiffWalk': 1.0,            # Dificultad para caminar
            'Sex': 1.0,
            'Age': 9.0,                 # Rango de edad avanzado (e.g., 60-64 años)
            'Education': 4.0,
            'Income': 3.0
        },
        {
            'HighBP': 0.0,              # Presión normal
            'HighChol': 0.0,            # Colesterol normal
            'CholCheck': 1.0,
            'BMI': 22.0,                # Peso saludable
            'Smoker': 0.0,
            'Stroke': 0.0,
            'HeartDiseaseorAttack': 0.0,
            'PhysActivity': 1.0,        # Deportista
            'Fruits': 1.0,
            'Veggies': 1.0,
            'HvyAlcoholConsump': 1.0,   # Bebedor habitual
            'AnyHealthcare': 1.0,
            'NoDocbcCost': 0.0,
            'GenHlth': 1.0,             # Salud percibida: Excelente
            'MentHlth': 0.0,
            'PhysHlth': 0.0,
            'DiffWalk': 0.0,
            'Sex': 0.0,
            'Age': 3.0,                 # Joven (e.g., 30-34 años)
            'Education': 6.0,
            'Income': 8.0
        }
    ])

    # Usar el Pipeline cargado para predecir
    probabilidades = modelo_DIRA.predict(datos_pacientes)
 
    print("\n── Resultados de inferencia ──────────────────────")
    for i, row in probabilidades.iterrows():
        print(f"\nPaciente {i + 1}:")
        print(f"  Probabilidad de diabetes : {row['prob_diabetes']:.2%}")
        print(f"  Predicción               : {'Diabético/Pre-diabético' if row['prediccion'] else 'No diabético'}")
        print(f"  Nivel de riesgo          : {row['nivel_riesgo']}")
        print(f"  Acción sugerida          : {row['accion']}")
    sys.exit(0)