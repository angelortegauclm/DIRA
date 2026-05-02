import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Any
from infer import DIRAPredictor  # Nuestra clase que realiza la predicción y carga el preprocesamiento y el modelo entrenado desde la ruta especificada.

app = FastAPI(
    title="DIRA - API",
    description="API MLOps para inferencia de riesgo de diabetes en tiempo real",
    version="1.0.0"
)


# Cargamos el predictor en el contexto global (se ejecuta al arrancar el servidor)
predictor = DIRAPredictor(model_path=os.getenv("MODEL_PATH", "./model/modelo_diabetes_DIRA.pkl"))

# Vamos a utilizar el protocolo Open Inference Protocol (OIP) V2 para definir el contrato de entrada y salida de la API, 
# lo que facilitará su integración con otros sistemas y herramientas de MLOps.
# https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol


# Endpoints de salud (Health Probes para Kubernetes)
@app.get("/v2/health/live")
def health_live():
    """
    Endpoint de liveness para Kubernetes. Devuelve 200 OK si la aplicación está viva.
    """
    return {"status": "alive"}  

@app.get("/v2/health/ready")
def health_ready():
    """
    Endpoint de readiness para Kubernetes. Devuelve 200 OK si la aplicación está lista para recibir tráfico.
    En este caso, se verifica que el modelo se ha cargado correctamente.
    """
    try:
        predictor.load()  # Intentamos cargar el modelo para verificar que está disponible
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {str(e)}")
    
# Endpoints de metadatos
# Leemos los metadatos a devolver desde las variables de entorno
MODEL_NAME = os.getenv("MODEL_NAME", "modelo_diabetes_DIRA")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
# Usamos el estándar para archivos .pkl de scikit-learn
MODEL_PLATFORM = os.getenv("MODEL_PLATFORM", "sklearn_joblib")
@app.get("/v2/models/DIRA")
def model_metadata():
    """
    Endpoint para obtener información del modelo.
    """
    # Implementación para obtener metadatos del modelo
    return {
        "name": MODEL_NAME,
        "versions": [MODEL_VERSION],
        "platform": MODEL_PLATFORM,
        "inputs": [
            {"name": "paciente_features", "datatype": "FP32", "shape": [-1, 20]}
        ],
        "outputs": [
            {"name": "prob_diabetes", "datatype": "FP32", "shape": [-1, 1]},
            {"name": "prediccion", "datatype": "INT32", "shape": [-1, 1]},
            {"name": "nivel_riesgo", "datatype": "BYTES", "shape": [-1, 1]},
            {"name": "accion", "datatype": "BYTES", "shape": [-1, 1]}
        ]
    }

# EndPoint de inferencia
# El protocolo V2 obliga a que el JSON tenga un formato estandarizado
# https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol#inference-response-json-object

class InferenceInput(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    data: List[dict]  # Esta variable contendrá los datos de los pacientes

class InferenceRequest(BaseModel):
    id: str | None = None
    inputs: List[InferenceInput]

@app.post(f"/v2/models/{MODEL_NAME}/infer")
def infer(request: InferenceRequest):
    try:
        input_tensor = request.inputs[0]
        
        n_pacientes = input_tensor.shape[0]
        if len(input_tensor.data) != n_pacientes:
            raise ValueError(f"Longitud de datos incorrecta. Se esperaban {n_pacientes} pacientes, se recibieron {len(input_tensor.data)}.")

        df = pd.DataFrame(input_tensor.data)
        
        # Hacemos la predicción llamando al método predict() de la clase DIRAPredictor, que devuelve un DataFrame con la probabilidad de diabetes,
        # la predicción binaria, el nivel de riesgo y la acción recomendada para cada paciente.
        predicciones = predictor.predict(df)
        n_preds = len(predicciones)
        
        # Desagregamos los resultados devuelto por el método predict() que recordemos devuelve un dataframe con 4 columnas:
        #   "prob_diabetes": round(float(prob), 4),
        #   "prediccion": int(pred),
        #   "nivel_riesgo": nivel,
        #   "accion": accion,
        # en listas separadas para cada output, lo que nos permitirá formatear la respuesta siguiendo el protocolo V2 de MLOps.
        probs = predicciones["prob_diabetes"].tolist()
        preds = predicciones["prediccion"].tolist()
        niveles = predicciones["nivel_riesgo"].tolist()
        acciones = predicciones["accion"].tolist()
        
        # Formateamos la salida con MÚLTIPLES outputs siguiendo el protocolo V2
        response = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "outputs": [
                {
                    "name": "prob_diabetes",
                    "datatype": "FP32",
                    "shape": [n_preds, 1],
                    "data": probs
                },
                {
                    "name": "prediccion",
                    "datatype": "INT32",
                    "shape": [n_preds, 1],
                    "data": preds
                },
                {
                    "name": "nivel_riesgo",
                    "datatype": "BYTES", # En el estándar V2, los strings se marcan como BYTES
                    "shape": [n_preds, 1],
                    "data": niveles
                },
                {
                    "name": "accion",
                    "datatype": "BYTES",
                    "shape": [n_preds, 1],
                    "data": acciones
                }
            ]
        }
        
        if request.id:
            response["id"] = request.id
            
        return response

    except Exception as e:
        # 2. EL OBJETO JSON DE ERROR PERFECTO
        # Cumplimos el estándar devolviendo estado HTTP 400 y la clave "error"
        return JSONResponse(
            status_code=400, 
            content={"error": f"Error en la inferencia: {str(e)}"}
        )