"""
Servicio web Flask que sirve el modelo de riesgo de diabetes.

Endpoints:
    GET  /         -> formulario HTML para probar el modelo en el navegador
    POST /predict  -> API JSON (application/json) o formulario HTML
    GET  /health   -> verificación de estado + metadatos del modelo

Uso:
    python predict.py
    # luego abre http://localhost:9696
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "diabetes_pipeline.joblib"

# (etiqueta, valor por defecto, mínimo, máximo, paso, texto de ayuda)
FIELD_INFO = {
    "Pregnancies": ("Número de embarazos", 2, 0, 20, 1, "Cuántas veces ha estado embarazada la paciente"),
    "Glucose": ("Glucosa (mg/dL)", 120, 0, 300, 1, "Glucosa en plasma (prueba de tolerancia oral de 2 h). 0 = no medido"),
    "BloodPressure": ("Presión arterial (mm Hg)", 70, 0, 200, 1, "Presión arterial diastólica. 0 = no medido"),
    "SkinThickness": ("Grosor del pliegue cutáneo (mm)", 25, 0, 100, 1, "Pliegue cutáneo del tríceps. 0 = no medido"),
    "Insulin": ("Insulina (mu U/ml)", 100, 0, 900, 1, "Insulina sérica a las 2 h. 0 = no medido"),
    "BMI": ("IMC", 28.0, 0.0, 70.0, 0.1, "Índice de masa corporal (kg / m²). 0 = no medido"),
    "DiabetesPedigreeFunction": ("Función de pedigrí diabético", 0.5, 0.0, 3.0, 0.01, "Puntaje de antecedentes familiares de diabetes"),
    "Age": ("Edad (años)", 35, 1, 120, 1, "Edad de la paciente"),
}

# Misma convención que train.py: un cero en estas columnas significa "no medido".
ZERO_AS_MISSING = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}

if MODEL_PATH.exists():
    BUNDLE = joblib.load(MODEL_PATH)
    MODEL = BUNDLE["model"]
    FEATURES = BUNDLE["features"]
    THRESHOLD = BUNDLE.get("threshold", 0.5)
    MODEL_NAME = BUNDLE.get("model_name", "desconocido")
else:  # pragma: no cover - solo si el modelo aún no se entrenó
    BUNDLE = None
    MODEL = None
    FEATURES = list(FIELD_INFO)
    THRESHOLD = 0.5
    MODEL_NAME = "modelo no entrenado"

app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Predictor de Riesgo de Diabetes</title>
  <style>
    :root { --bg:#0f172a; --card:#1e293b; --accent:#38bdf8; --text:#e2e8f0; --muted:#94a3b8; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
           background: radial-gradient(circle at top, #1e3a8a 0%, var(--bg) 55%);
           font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:var(--text); padding:24px; }
    .card { width:100%; max-width:760px; background:rgba(30,41,59,.92); border:1px solid #334155;
            border-radius:18px; padding:28px; box-shadow:0 20px 50px rgba(0,0,0,.45); }
    h1 { margin:0 0 6px; font-size:1.6rem; }
    p.sub { margin:0 0 22px; color:var(--muted); font-size:.95rem; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:14px; }
    label { display:block; font-size:.85rem; margin-bottom:5px; color:var(--muted); }
    input { width:100%; padding:10px 12px; border-radius:9px; border:1px solid #475569;
            background:#0f172a; color:var(--text); font-size:1rem; }
    input:focus { outline:2px solid var(--accent); border-color:transparent; }
    button { margin-top:20px; width:100%; padding:13px; border:0; border-radius:11px;
             background:linear-gradient(90deg,#0ea5e9,#22d3ee); color:#06202f; font-weight:700;
             font-size:1rem; cursor:pointer; }
    button:hover { filter:brightness(1.08); }
    .result { margin-top:22px; padding:18px; border-radius:13px; border:1px solid #334155; background:#0b1220; }
    .pill { display:inline-block; padding:5px 12px; border-radius:999px; font-weight:700; font-size:.85rem; }
    .high { background:#7f1d1d; color:#fecaca; }
    .low  { background:#14532d; color:#bbf7d0; }
    .prob { font-size:2rem; font-weight:800; margin:10px 0 4px; }
    .bar { height:10px; border-radius:999px; background:#1e293b; overflow:hidden; margin:10px 0 6px; }
    .bar span { display:block; height:100%; background:linear-gradient(90deg,#22c55e,#eab308,#ef4444); }
    .error { margin-top:18px; padding:12px; border-radius:10px; background:#7f1d1d; color:#fecaca; }
    code { color:var(--accent); }
    footer { margin-top:20px; color:var(--muted); font-size:.8rem; text-align:center; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Predictor de Riesgo de Diabetes</h1>
    <p class="sub">Ingresa los valores clínicos del paciente y el modelo estimará la probabilidad de diabetes.</p>
    <form method="post" action="/predict">
      <div class="grid">
        {% for name, info in fields.items() %}
        <div>
          <label for="{{ name }}" title="{{ info[5] }}">{{ info[0] }}</label>
          <input type="number" step="{{ info[4] }}" min="{{ info[2] }}" max="{{ info[3] }}"
                 id="{{ name }}" name="{{ name }}" value="{{ values[name] }}" required>
        </div>
        {% endfor %}
      </div>
      <button type="submit">Predecir riesgo</button>
    </form>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    {% if result %}
    <div class="result">
      <span class="pill {{ 'high' if result.prediction == 1 else 'low' }}">
        {{ 'Riesgo alto' if result.prediction == 1 else 'Riesgo bajo' }}
      </span>
      <div class="prob">{{ '%.1f'|format(result.risk_probability * 100) }}%</div>
      <div class="bar"><span style="width: {{ (result.risk_probability * 100)|round(1) }}%"></span></div>
      <div class="sub" style="margin:0">Probabilidad de diabetes (umbral {{ result.threshold }}). Modelo: {{ result.model }}.</div>
    </div>
    {% endif %}
    <footer>Envía JSON a <code>/predict</code> &middot; estado en <code>/health</code></footer>
  </div>
</body>
</html>
"""


def parse_payload(payload: dict) -> dict:
    """Valida y convierte los valores recibidos a números."""
    values = {}
    missing = [name for name in FEATURES if name not in payload or str(payload[name]).strip() == ""]
    if missing:
        raise ValueError(f"Faltan campos: {', '.join(missing)}")

    for name in FEATURES:
        try:
            value = float(payload[name])
        except (TypeError, ValueError):
            raise ValueError(f"El campo '{name}' debe ser un número, se recibió {payload[name]!r}")
        _, _, low, high, _, _ = FIELD_INFO[name]
        if not (low <= value <= high):
            raise ValueError(f"El campo '{name}' debe estar entre {low} y {high}, se recibió {value}")
        if name in ZERO_AS_MISSING and value == 0:
            value = float("nan")  # el imputador lo rellena con la mediana del entrenamiento
        values[name] = value
    return values


def make_prediction(values: dict) -> dict:
    row = pd.DataFrame([values], columns=FEATURES)
    probability = float(MODEL.predict_proba(row)[0, 1])
    prediction = int(probability >= THRESHOLD)
    return {
        "prediction": prediction,
        "risk_label": "riesgo alto" if prediction == 1 else "riesgo bajo",
        "risk_probability": round(probability, 4),
        "threshold": THRESHOLD,
        "model": MODEL_NAME,
    }


@app.get("/")
def index():
    values = {name: info[1] for name, info in FIELD_INFO.items()}
    return render_template_string(PAGE_TEMPLATE, fields=FIELD_INFO, values=values, result=None, error=None)


@app.post("/predict")
def predict():
    if MODEL is None:
        return jsonify({"error": "No se encontró el modelo. Ejecuta 'python train.py' primero."}), 503

    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if payload is None:
        return jsonify({"error": "Envía un cuerpo JSON o un formulario."}), 400

    try:
        values = parse_payload(payload)
    except ValueError as exc:
        if request.is_json:
            return jsonify({"error": str(exc)}), 400
        defaults = {name: info[1] for name, info in FIELD_INFO.items()}
        return render_template_string(
            PAGE_TEMPLATE, fields=FIELD_INFO, values=defaults, result=None, error=str(exc)
        ), 400

    result = make_prediction(values)

    if request.is_json:
        return jsonify(result)

    filled = {name: payload.get(name, FIELD_INFO[name][1]) for name in FEATURES}
    return render_template_string(
        PAGE_TEMPLATE, fields=FIELD_INFO, values=filled, result=result, error=None
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL is not None,
            "model": MODEL_NAME,
            "features": FEATURES,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=False)
