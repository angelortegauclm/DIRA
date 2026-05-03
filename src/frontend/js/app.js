const INFER_ENDPOINT = API_URL + '/v2/models/modelo_diabetes_DIRA/infer';

const v = id => parseFloat(document.getElementById(id).value);
const b = id => document.getElementById(id).checked ? 1.0 : 0.0;

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const btn      = document.getElementById('submitBtn');
    const errorDiv = document.getElementById('error');
    const result   = document.getElementById('result');

    btn.disabled = true;
    btn.textContent = 'Evaluando...';
    errorDiv.style.display = 'none';
    result.style.display = 'none';

    // Orden exacto del dataset de entrenamiento (21 features)
    const data = [
      b('HighBP'), b('HighChol'), b('CholCheck'), v('BMI'),
      b('Smoker'), b('Stroke'), b('HeartDiseaseorAttack'),
      b('PhysActivity'), b('Fruits'), b('Veggies'), b('HvyAlcoholConsump'),
      b('AnyHealthcare'), b('NoDocbcCost'),
      v('GenHlth'), v('MentHlth'), v('PhysHlth'),
      b('DiffWalk'), v('Sex'), v('Age'), v('Education'), v('Income')
    ];

    try {
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

      const json    = await resp.json();
      const outputs = json.outputs;
      const prob    = outputs.find(o => o.name === 'prob_diabetes').data[0];
      const nivel   = outputs.find(o => o.name === 'nivel_riesgo').data[0];
      const accion  = outputs.find(o => o.name === 'accion').data[0];

      const color = prob < 0.3 ? 'green' : prob < 0.7 ? 'orange' : 'red';

      result.className = `result ${color}`;
      document.getElementById('res-prob').textContent  = (prob * 100).toFixed(1) + '%';
      document.getElementById('res-nivel').textContent  = nivel;
      document.getElementById('res-accion').textContent = accion;
      result.style.display = 'block';

    } catch (err) {
      errorDiv.textContent = 'Error: ' + err.message;
      errorDiv.style.display = 'block';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Evaluar Riesgo de Diabetes';
    }
  });
});
