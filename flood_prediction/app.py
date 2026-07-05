"""
app.py
------
Flask web application for the Flood Prediction System.
Loads the trained model + scaler produced by train_model.py and exposes:

    GET  /            -> input form (HTML UI)
    POST /predict      -> returns prediction rendered back into the page
    POST /api/predict   -> JSON API endpoint (for programmatic / mobile use)

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = "model/flood_model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURES_PATH = "model/feature_columns.pkl"
BEST_NAME_PATH = "model/best_model_name.pkl"

# --- Load trained artifacts at startup ------------------------------------
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    raise RuntimeError(
        "Model files not found. Run `python generate_dataset.py` and then "
        "`python train_model.py` before starting the Flask app."
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURE_COLUMNS = joblib.load(FEATURES_PATH)
BEST_MODEL_NAME = joblib.load(BEST_NAME_PATH)


def make_prediction(values: dict):
    """values: dict of feature_name -> float"""
    ordered = [values[col] for col in FEATURE_COLUMNS]
    X = np.array(ordered).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_scaled)[0][1])  # P(flood)

    return {
        "flood_predicted": bool(pred),
        "probability": proba,
        "model_used": BEST_MODEL_NAME,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        feature_columns=FEATURE_COLUMNS,
        model_name=BEST_MODEL_NAME,
        result=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {col: float(request.form[col]) for col in FEATURE_COLUMNS}
    except (KeyError, ValueError):
        return render_template(
            "index.html",
            feature_columns=FEATURE_COLUMNS,
            model_name=BEST_MODEL_NAME,
            result={"error": "Please enter valid numeric values for all fields."},
        )

    result = make_prediction(values)
    return render_template(
        "index.html",
        feature_columns=FEATURE_COLUMNS,
        model_name=BEST_MODEL_NAME,
        result=result,
        submitted_values=values,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: POST a JSON body with the required feature keys, e.g.
    {
      "annual_rainfall": 2200,
      "seasonal_rainfall": 1400,
      "cloud_visibility": 2.5,
      "temperature": 27.5,
      "humidity": 88
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        values = {col: float(data[col]) for col in FEATURE_COLUMNS}
    except ValueError:
        return jsonify({"error": "All feature values must be numeric"}), 400

    return jsonify(make_prediction(values))


if __name__ == "__main__":
    # host="0.0.0.0" makes it reachable on your local network / when
    # containerized (e.g. for IBM Cloud deployment)
    app.run(host="0.0.0.0", port=5000, debug=True)
