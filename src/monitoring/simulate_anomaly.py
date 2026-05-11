"""
simulate_anomaly.py

Simula un entorno anómalo enviando peticiones con pacientes de muy alto riesgo
para provocar una deriva artificial en la distribución de inferencias recientes.

Propósito:
  Permite probar el sistema de monitorización sin esperar tráfico real anómalo:
  - Rellena inference_log.csv con perfiles extremos
  - Provoca que Evidently detecte drift (score ≈ 1.0)
  - Dispara las alertas DiraDataDriftDetected y DiraHighRiskSurge en Prometheus
  - Genera una notificación en el bot de Telegram DIRAProject_bot

Los perfiles enviados tienen valores muy alejados de la distribución del
dataset BRFSS: BMI extremo, edad avanzada, todos los factores de riesgo a 1.
Esto garantiza que Evidently detecte deriva en prácticamente todas las variables.

Uso:
  python src/monitoring/simulate_anomaly.py \\
    --url http://192.168.1.131:31995 \\
    --n 200 \\
    --delay 0.05
"""

import argparse
import random
import time

import requests

# Ruta del endpoint de inferencia siguiendo el protocolo Open Inference V2
ENDPOINT = "/v2/models/modelo_diabetes_DIRA/infer"

# ── Perfil base de paciente de muy alto riesgo ────────────────────────────────
# Valores fijos para todos los factores de riesgo conocidos del dataset BRFSS.
# Orden: HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
#        PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
#        NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age,
#        Education, Income
_HIGH_RISK_BASE = [
    1, 1, 1,    # HighBP=1, HighChol=1, CholCheck=1
    45.0,       # BMI: obesidad severa (valor base, se randomiza en _random_high_risk)
    1, 1, 1,    # Smoker=1, Stroke=1, HeartDiseaseorAttack=1
    0, 0, 0, 1, # PhysActivity=0, Fruits=0, Veggies=0, HvyAlcoholConsump=1
    0, 1,       # AnyHealthcare=0, NoDocbcCost=1
    5, 20, 25,  # GenHlth=5 (muy mala), MentHlth=20 días, PhysHlth=25 días
    1,          # DiffWalk=1 (dificultad para caminar)
    1, 11, 1, 1 # Sex=1, Age=11 (70-74 años), Education=1 (mínima), Income=1 (mínimo)
]


def _random_high_risk() -> list:
    """Genera un perfil de alto riesgo con variación aleatoria en los valores continuos.

    Randomizar BMI, edad y días de salud evita que Evidently reciba siempre
    el mismo vector exacto, lo que haría el análisis estadístico menos realista.
    La variación se mantiene dentro de rangos extremos para asegurar el drift.
    """
    p = _HIGH_RISK_BASE.copy()
    p[3]  = round(random.uniform(35, 70), 1)   # BMI entre obesidad II y obesidad extrema
    p[13] = random.choice([4, 5])              # GenHlth: mala (4) o muy mala (5)
    p[14] = random.randint(15, 30)             # MentHlth: 15-30 días con mala salud mental
    p[15] = random.randint(15, 30)             # PhysHlth: 15-30 días con mala salud física
    p[18] = random.randint(9, 13)              # Age: grupos 60-65 a 80+ años
    return p


def _send(base_url: str, data: list) -> str:
    """Envía una petición de inferencia y devuelve el nivel de riesgo resultante.

    Usa el formato Open Inference V2: tensor con nombre 'paciente_features',
    shape [1, 21] y datos como lista plana de floats.
    """
    resp = requests.post(
        base_url.rstrip("/") + ENDPOINT,
        json={"inputs": [{"name": "paciente_features", "shape": [1, 21], "datatype": "FP32", "data": data}]},
        timeout=5,
    )
    resp.raise_for_status()
    # El tercer output (índice 2) es el nivel_riesgo como string
    return resp.json()["outputs"][2]["data"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="DIRA – Simulación de entorno anómalo")
    parser.add_argument("--url",   required=True, help="URL base de la API (ej: http://192.168.1.131:31995)")
    parser.add_argument("--n",     type=int,   default=200,  help="Número de peticiones a enviar")
    parser.add_argument("--delay", type=float, default=0.05, help="Segundos entre peticiones")
    args = parser.parse_args()

    print(f"Enviando {args.n} pacientes de alto riesgo a {args.url}...\n")
    ok = errores = 0

    for i in range(1, args.n + 1):
        try:
            nivel = _send(args.url, _random_high_risk())
            print(f"[{i:03d}] {nivel}")
            ok += 1
        except Exception as exc:
            print(f"[{i:03d}] ERROR: {exc}")
            errores += 1
        time.sleep(args.delay)

    print(f"\nResumen: {ok} ok, {errores} errores de {args.n} peticiones.")
    print("\nPróximo paso: ejecutar el drift check para ver el score en Prometheus:")
    print("  kubectl create job --from=cronjob/dira-drift-check drift-test-$(date +%s) -n mlrun")


if __name__ == "__main__":
    main()
