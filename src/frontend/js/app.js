// ── app.js ────────────────────────────────────────────────────────────────────
//
// Lógica del formulario de evaluación de riesgo de diabetes de DIRA.
// Recoge los valores del formulario HTML, construye la petición en formato
// Open Inference V2 y muestra el resultado con código de color.
//
// Depende de config.js (cargado antes en index.html) que define API_URL.

// Endpoint de inferencia completo: API_URL viene de config.js, sustituido
// por envsubst al arrancar el contenedor nginx con la URL real del clúster.
const INFER_ENDPOINT = API_URL + '/v2/models/modelo_diabetes_DIRA/infer';

// Helpers para leer valores del formulario:
//   v(id) → valor numérico de un <input type="number"> o <select>
//   b(id) → 1.0 si un <input type="checkbox"> está marcado, 0.0 si no
const v = id => parseFloat(document.getElementById(id).value);
const b = id => document.getElementById(id).checked ? 1.0 : 0.0;

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('form').addEventListener('submit', async (e) => {
    e.preventDefault();   // evitar recarga de página al enviar el formulario

    const btn      = document.getElementById('submitBtn');
    const errorDiv = document.getElementById('error');
    const result   = document.getElementById('result');

    // Deshabilitar el botón mientras se procesa para evitar dobles envíos
    btn.disabled = true;
    btn.textContent = 'Evaluando...';
    errorDiv.style.display = 'none';
    result.style.display = 'none';

    // ── Construir el tensor de entrada ────────────────────────────────────
    // Las 21 features deben ir en el MISMO ORDEN que el dataset de entrenamiento
    // (definido en features.py → FEATURE_COLS). Cualquier cambio de orden
    // produciría predicciones incorrectas sin generar ningún error visible.
    const data = [
      b('HighBP'), b('HighChol'), b('CholCheck'), v('BMI'),
      b('Smoker'), b('Stroke'), b('HeartDiseaseorAttack'),
      b('PhysActivity'), b('Fruits'), b('Veggies'), b('HvyAlcoholConsump'),
      b('AnyHealthcare'), b('NoDocbcCost'),
      v('GenHlth'), v('MentHlth'), v('PhysHlth'),
      b('DiffWalk'), v('Sex'), v('Age'), v('Education'), v('Income')
    ];

    try {
      // ── Petición al endpoint de inferencia ───────────────────────────────
      // Formato Open Inference V2: tensor con nombre 'paciente_features',
      // shape [1, 21] (1 paciente, 21 features) y datos como lista plana.
      const resp = await fetch(INFER_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputs: [{ name: 'paciente_features', shape: [1, 21], datatype: 'FP32', data }]
        })
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }

      // ── Extraer resultados de la respuesta ────────────────────────────────
      // La API devuelve cuatro outputs: prob_diabetes, prediccion, nivel_riesgo, accion.
      // Se accede por nombre para no depender del orden en el array outputs.
      const json    = await resp.json();
      const outputs = json.outputs;
      const prob    = outputs.find(o => o.name === 'prob_diabetes').data[0];
      const nivel   = outputs.find(o => o.name === 'nivel_riesgo').data[0];
      const accion  = outputs.find(o => o.name === 'accion').data[0];

      // ── Código de color según probabilidad ───────────────────────────────
      // Verde  (< 30%): bajo riesgo
      // Naranja (30–70%): riesgo moderado
      // Rojo   (> 70%): alto riesgo / prioritario
      // Estos umbrales deben coincidir con UMBRAL_BAJO y UMBRAL_ALTO en infer.py
      const color = prob < 0.3 ? 'green' : prob < 0.7 ? 'orange' : 'red';

      // Mostrar el resultado con el color correspondiente
      result.className = `result ${color}`;
      document.getElementById('res-prob').textContent  = (prob * 100).toFixed(1) + '%';
      document.getElementById('res-nivel').textContent  = nivel;
      document.getElementById('res-accion').textContent = accion;
      result.style.display = 'block';

    } catch (err) {
      // Mostrar el error al usuario sin ocultar el formulario para que pueda reintentar
      errorDiv.textContent = 'Error: ' + err.message;
      errorDiv.style.display = 'block';
    } finally {
      // Rehabilitar el botón siempre, tanto si hubo éxito como si hubo error
      btn.disabled = false;
      btn.textContent = 'Evaluar Riesgo de Diabetes';
    }
  });
});
