"""
Flask web service that serves the diabetes risk model.

Endpoints:
    GET  /         -> HTML form to try the model in a browser
    POST /predict  -> JSON API (application/json) or HTML form
    GET  /health   -> health check + model metadata

Usage:
    python predict.py
    # then open http://localhost:9696
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "diabetes_pipeline.joblib"

# (label, default, min, max, step, help text)
FIELD_INFO = {
    "Pregnancies": ("Number of pregnancies", 2, 0, 20, 1, "How many times the patient has been pregnant"),
    "Glucose": ("Glucose (mg/dL)", 120, 0, 300, 1, "Plasma glucose concentration (2-hour OGTT). 0 = not measured"),
    "BloodPressure": ("Blood pressure (mm Hg)", 70, 0, 200, 1, "Diastolic blood pressure. 0 = not measured"),
    "SkinThickness": ("Skin thickness (mm)", 25, 0, 100, 1, "Triceps skin fold thickness. 0 = not measured"),
    "Insulin": ("Insulin (mu U/ml)", 100, 0, 900, 1, "2-hour serum insulin. 0 = not measured"),
    "BMI": ("BMI", 28.0, 0.0, 70.0, 0.1, "Body mass index (kg / m^2). 0 = not measured"),
    "DiabetesPedigreeFunction": ("Diabetes pedigree function", 0.5, 0.0, 3.0, 0.01, "Family-history score of diabetes"),
    "Age": ("Age (years)", 35, 1, 120, 1, "Age of the patient"),
}

# Same convention as train.py: a zero in these columns means "not measured".
ZERO_AS_MISSING = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}


if MODEL_PATH.exists():
    BUNDLE = joblib.load(MODEL_PATH)
    MODEL = BUNDLE["model"]
    FEATURES = BUNDLE["features"]
    THRESHOLD = BUNDLE.get("threshold", 0.5)
    MODEL_NAME = BUNDLE.get("model_name", "unknown")
else:  # pragma: no cover - only when the model was not trained yet
    BUNDLE = None
    MODEL = None
    FEATURES = list(FIELD_INFO)
    THRESHOLD = 0.5
    MODEL_NAME = "model not trained"

app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Diabetes Risk Predictor</title>
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
    <h1>Diabetes Risk Predictor</h1>
    <p class="sub">Enter the patient's clinical values and the model will estimate the probability of diabetes.</p>
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
      <button type="submit">Predict risk</button>
    </form>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    {% if result %}
    <div class="result">
      <span class="pill {{ 'high' if result.prediction == 1 else 'low' }}">
        {{ 'High risk' if result.prediction == 1 else 'Low risk' }}
      </span>
      <div class="prob">{{ '%.1f'|format(result.risk_probability * 100) }}%</div>
      <div class="bar"><span style="width: {{ (result.risk_probability * 100)|round(1) }}%"></span></div>
      <div class="sub" style="margin:0">Probability of diabetes (threshold {{ result.threshold }}). Model: {{ result.model }}.</div>
    </div>
    {% endif %}
    <footer>POST JSON to <code>/predict</code> &middot; health check at <code>/health</code></footer>
  </div>
</body>
</html>
"""


def parse_payload(payload: dict) -> dict:
    """Validate and convert the incoming values to floats."""
    values = {}
    missing = [name for name in FEATURES if name not in payload or str(payload[name]).strip() == ""]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    for name in FEATURES:
        try:
            value = float(payload[name])
        except (TypeError, ValueError):
            raise ValueError(f"Field '{name}' must be a number, got {payload[name]!r}")
        _, _, low, high, _, _ = FIELD_INFO[name]
        if not (low <= value <= high):
            raise ValueError(f"Field '{name}' must be between {low} and {high}, got {value}")
        if name in ZERO_AS_MISSING and value == 0:
            value = float("nan")  # the imputer will fill it with the training median
        values[name] = value
    return values


def make_prediction(values: dict) -> dict:
    row = pd.DataFrame([values], columns=FEATURES)
    probability = float(MODEL.predict_proba(row)[0, 1])
    prediction = int(probability >= THRESHOLD)
    return {
        "prediction": prediction,
        "risk_label": "high risk" if prediction == 1 else "low risk",
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
        return jsonify({"error": "Model not found. Run 'python train.py' first."}), 503

    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if payload is None:
        return jsonify({"error": "Send a JSON body or a form."}), 400

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
