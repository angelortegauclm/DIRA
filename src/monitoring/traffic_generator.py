"""
traffic_generator.py

Genera tráfico continuo hacia la API de inferencia de DIRA.

Modos:
  --mode production  (recomendado) Muestrea de las distribuciones reales del
                     dataset BRFSS 2015. Produce un drift score bajo (<20%)
                     porque los datos son estadísticamente similares a la
                     referencia de entrenamiento. Simula uso real en producción.

  --mode synthetic   Perfiles exagerados BAJO/MODERADO/ALTO con distribuciones
                     extremas para probar alertas y detección de deriva.
                     Produce drift ~66%. Útil para validar el pipeline.

Uso:
  python src/monitoring/traffic_generator.py --url http://192.168.1.131:31995
  python src/monitoring/traffic_generator.py --url http://192.168.1.131:31995 \\
      --mode production --duration 3600 --rps 2
  python src/monitoring/traffic_generator.py --url http://192.168.1.131:31995 \\
      --mode synthetic --duration 300 --rps 3 --mix 60:25:15
"""

import argparse
import math
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests

ENDPOINT = "/v2/models/modelo_diabetes_DIRA/infer"

# ── Perfiles de paciente ──────────────────────────────────────────────────────
# Cada función devuelve un vector de 21 features en el orden esperado por la API:
# HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
# PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
# GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income

def _perfil_bajo_riesgo() -> list:
    return [
        random.randint(0, 0),           # HighBP: no
        random.randint(0, 0),           # HighChol: no
        1,                              # CholCheck: sí
        round(random.uniform(18, 26), 1),  # BMI normal
        random.randint(0, 1),           # Smoker
        0,                              # Stroke: no
        0,                              # HeartDiseaseorAttack: no
        1,                              # PhysActivity: sí
        random.randint(0, 1),           # Fruits
        random.randint(0, 1),           # Veggies
        0,                              # HvyAlcoholConsump: no
        1,                              # AnyHealthcare: sí
        0,                              # NoDocbcCost: no
        random.randint(1, 2),           # GenHlth: excelente/muy buena
        random.randint(0, 5),           # MentHlth: pocos días
        random.randint(0, 5),           # PhysHlth: pocos días
        0,                              # DiffWalk: no
        random.randint(0, 1),           # Sex
        random.randint(2, 5),           # Age: 25-44
        random.randint(4, 6),           # Education: universitaria
        random.randint(6, 8),           # Income: alta
    ]


def _perfil_moderado_riesgo() -> list:
    return [
        random.randint(0, 1),           # HighBP: variable
        random.randint(0, 1),           # HighChol: variable
        1,                              # CholCheck
        round(random.uniform(26, 35), 1),  # BMI sobrepeso/obesidad I
        random.randint(0, 1),           # Smoker
        0,                              # Stroke: no
        random.randint(0, 1),           # HeartDiseaseorAttack
        random.randint(0, 1),           # PhysActivity
        random.randint(0, 1),           # Fruits
        random.randint(0, 1),           # Veggies
        random.randint(0, 1),           # HvyAlcoholConsump
        1,                              # AnyHealthcare
        random.randint(0, 1),           # NoDocbcCost
        random.randint(2, 4),           # GenHlth: buena/regular
        random.randint(0, 15),          # MentHlth
        random.randint(0, 15),          # PhysHlth
        random.randint(0, 1),           # DiffWalk
        random.randint(0, 1),           # Sex
        random.randint(5, 9),           # Age: 45-64
        random.randint(3, 5),           # Education: secundaria/alguna universidad
        random.randint(3, 6),           # Income: media
    ]


def _perfil_alto_riesgo() -> list:
    return [
        1,                              # HighBP: sí
        1,                              # HighChol: sí
        1,                              # CholCheck
        round(random.uniform(32, 55), 1),  # BMI obesidad II/III
        random.randint(0, 1),           # Smoker
        random.randint(0, 1),           # Stroke
        random.randint(0, 1),           # HeartDiseaseorAttack
        0,                              # PhysActivity: no
        0,                              # Fruits: no
        0,                              # Veggies: no
        random.randint(0, 1),           # HvyAlcoholConsump
        random.randint(0, 1),           # AnyHealthcare
        random.randint(0, 1),           # NoDocbcCost
        random.randint(3, 5),           # GenHlth: regular/mala/muy mala
        random.randint(10, 30),         # MentHlth: muchos días
        random.randint(10, 30),         # PhysHlth: muchos días
        random.randint(0, 1),           # DiffWalk
        random.randint(0, 1),           # Sex
        random.randint(9, 13),          # Age: 60-80+
        random.randint(1, 3),           # Education: baja
        random.randint(1, 3),           # Income: baja
    ]


def _perfil_produccion() -> list:
    """
    Muestrea de las distribuciones reales del dataset BRFSS 2015 (n=253.680).

    Cada variable usa las proporciones exactas medidas en el dataset:
      - Variables binarias: random.choices con los pesos reales
      - BMI: normal(mean=28.4, std=5.5) truncada en [12, 98]
      - MentHlth/PhysHlth: distribución zero-inflated (mediana=0, p90~10-20)
      - Age/Education/Income/GenHlth: categorical weighted con pesos reales

    El resultado produce un drift score <20% porque las distribuciones son
    estadísticamente similares a la referencia de entrenamiento.
    """
    def _binary(p_yes: float) -> int:
        return random.choices([0, 1], weights=[1 - p_yes, p_yes])[0]

    def _zero_inflated(p_zero: float, max_val: int = 30) -> int:
        if random.random() < p_zero:
            return 0
        # Exponencial discreta sesgada hacia valores bajos
        return min(max_val, int(random.expovariate(1 / 7)) + 1)

    return [
        _binary(0.429),           # HighBP:               43% sí
        _binary(0.424),           # HighChol:             42% sí
        _binary(0.963),           # CholCheck:            96% sí
        round(max(12.0, min(98.0, random.gauss(28.4, 5.5))), 1),  # BMI
        _binary(0.443),           # Smoker:               44% sí
        _binary(0.041),           # Stroke:                4% sí
        _binary(0.094),           # HeartDiseaseorAttack:  9% sí
        _binary(0.757),           # PhysActivity:         76% sí
        _binary(0.634),           # Fruits:               63% sí
        _binary(0.811),           # Veggies:              81% sí
        _binary(0.056),           # HvyAlcoholConsump:     6% sí
        _binary(0.951),           # AnyHealthcare:        95% sí
        _binary(0.084),           # NoDocbcCost:           8% sí
        # GenHlth: 1=excelente(18%) 2=muy buena(35%) 3=buena(30%) 4=regular(12%) 5=mala(5%)
        random.choices([1, 2, 3, 4, 5], weights=[0.179, 0.351, 0.298, 0.124, 0.048])[0],
        _zero_inflated(0.73),     # MentHlth: 73% son 0 días
        _zero_inflated(0.68),     # PhysHlth: 68% son 0 días
        _binary(0.168),           # DiffWalk:             17% sí
        _binary(0.440),           # Sex:                  44% hombre
        # Age: categorías 1-13 (18-24 hasta 80+), distribución real BRFSS
        random.choices(
            list(range(1, 14)),
            weights=[0.022, 0.030, 0.044, 0.054, 0.064, 0.078,
                     0.104, 0.122, 0.131, 0.127, 0.093, 0.063, 0.068]
        )[0],
        # Education: 1-6, mayoría nivel 5-6 (bachillerato/universidad)
        random.choices([1, 2, 3, 4, 5, 6],
                       weights=[0.001, 0.016, 0.037, 0.247, 0.276, 0.423])[0],
        # Income: 1-8, mayoría niveles 7-8 (>$50k)
        random.choices([1, 2, 3, 4, 5, 6, 7, 8],
                       weights=[0.039, 0.046, 0.063, 0.079, 0.102, 0.144, 0.170, 0.356])[0],
    ]


_PERFIL_FN = {
    "bajo":    _perfil_bajo_riesgo,
    "moderado": _perfil_moderado_riesgo,
    "alto":    _perfil_alto_riesgo,
}

# ── Estadísticas ──────────────────────────────────────────────────────────────

@dataclass
class Stats:
    sent: int = 0
    ok: int = 0
    errors: int = 0
    bajo: int = 0
    moderado: int = 0
    alto: int = 0
    total_latency: float = 0.0

    def record(self, nivel: Optional[str], latency: float) -> None:
        self.sent += 1
        if nivel is None:
            self.errors += 1
            return
        self.ok += 1
        self.total_latency += latency
        nivel_lower = nivel.lower()
        if "bajo" in nivel_lower:
            self.bajo += 1
        elif "alto" in nivel_lower:
            self.alto += 1
        else:
            self.moderado += 1

    def avg_latency(self) -> float:
        return (self.total_latency / self.ok * 1000) if self.ok else 0.0

    def print_summary(self) -> None:
        total = self.ok or 1
        print(
            f"\r  Enviadas: {self.sent:4d} | "
            f"OK: {self.ok} | "
            f"Err: {self.errors} | "
            f"Bajo: {self.bajo} ({self.bajo/total:.0%}) "
            f"Mod: {self.moderado} ({self.moderado/total:.0%}) "
            f"Alto: {self.alto} ({self.alto/total:.0%}) | "
            f"Lat: {self.avg_latency():.0f}ms",
            end="", flush=True,
        )


# ── Envío ─────────────────────────────────────────────────────────────────────

def _send(base_url: str, features: list) -> tuple[Optional[str], float]:
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            base_url.rstrip("/") + ENDPOINT,
            json={"inputs": [{"name": "paciente_features", "shape": [1, 21], "datatype": "FP32", "data": features}]},
            timeout=5,
        )
        resp.raise_for_status()
        nivel = resp.json()["outputs"][2]["data"][0]
        return nivel, time.perf_counter() - t0
    except Exception:
        return None, time.perf_counter() - t0


def _pick_perfil(mix: tuple[int, int, int]) -> list:
    bajo_pct, mod_pct, _ = mix
    r = random.randint(1, 100)
    if r <= bajo_pct:
        return _perfil_bajo_riesgo()
    elif r <= bajo_pct + mod_pct:
        return _perfil_moderado_riesgo()
    else:
        return _perfil_alto_riesgo()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DIRA – Generador de tráfico para dashboards")
    parser.add_argument("--url",      required=True, help="URL base de la API")
    parser.add_argument("--duration", type=int,   default=0,
                        help="Segundos de ejecución (0 = infinito hasta Ctrl+C)")
    parser.add_argument("--rps",      type=float, default=2.0,
                        help="Peticiones por segundo base (default: 2)")
    parser.add_argument("--mode",     default="production", choices=["production", "synthetic"],
                        help="production: distribuciones reales BRFSS, drift bajo; "
                             "synthetic: perfiles exagerados, drift alto (default: production)")
    parser.add_argument("--mix",      default="50:30:20",
                        help="Solo en modo synthetic. BAJO:MODERADO:ALTO (default: 50:30:20)")
    parser.add_argument("--wave",     action="store_true", default=True,
                        help="Variar el ritmo en onda senoidal (activo por defecto)")
    parser.add_argument("--no-wave",  dest="wave", action="store_false",
                        help="Ritmo constante sin variación")
    args = parser.parse_args()

    # Parsear mix (solo relevante en modo synthetic)
    try:
        parts = [int(x) for x in args.mix.split(":")]
        assert len(parts) == 3 and sum(parts) == 100
        mix = tuple(parts)
    except Exception:
        print("ERROR: --mix debe tener formato BAJO:MODERADO:ALTO y sumar 100 (ej: 50:30:20)")
        sys.exit(1)

    stats = Stats()
    running = True

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    deadline = time.time() + args.duration if args.duration > 0 else None

    print(f"DIRA Traffic Generator")
    print(f"  URL      : {args.url}")
    print(f"  Modo     : {args.mode}")
    print(f"  RPS base : {args.rps}")
    if args.mode == "synthetic":
        print(f"  Mix      : bajo={mix[0]}% mod={mix[1]}% alto={mix[2]}%")
    print(f"  Duración : {'∞ (Ctrl+C para parar)' if not deadline else f'{args.duration}s'}")
    print(f"  Onda     : {'sí' if args.wave else 'no'}")
    print()

    t_start = time.time()
    req_idx = 0

    while running:
        if deadline and time.time() >= deadline:
            break

        # Ritmo variable en onda senoidal: rps * (0.5 + 0.5 * |sin(t/30)|)
        elapsed = time.time() - t_start
        if args.wave:
            factor = 0.5 + 0.5 * abs(math.sin(elapsed / 30.0))
        else:
            factor = 1.0
        current_rps = max(0.2, args.rps * factor)
        delay = 1.0 / current_rps

        if args.mode == "production":
            features = _perfil_produccion()
        else:
            features = _pick_perfil(mix)
        nivel, latency = _send(args.url, features)
        stats.record(nivel, latency)
        req_idx += 1

        if req_idx % 10 == 0:
            stats.print_summary()

        time.sleep(delay)

    stats.print_summary()
    print(f"\n\nFinalizado. {stats.sent} peticiones en {time.time()-t_start:.0f}s.")


if __name__ == "__main__":
    main()
