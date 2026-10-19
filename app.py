"""
app.py  —  Task 2: prediction service with logging around predictions.
Endpoints:
  POST /predict  -> {"features": [f1, f2, f3, f4]}
  GET  /stats    -> monitoring summary (Task 4)
  GET  /drift    -> runs the drift check on recent live inputs (Task 5)
"""
import json
import time
from collections import deque

import joblib
import numpy as np
from flask import Flask, jsonify, request

from logger_config import get_logger
from monitoring import drift_check, record, stats, summary

logger = get_logger()
app = Flask(__name__)

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Load model and training reference stats at startup
try:
    model = joblib.load("model.pkl")
    with open("train_stats.json") as f:
        TRAIN_MEANS = json.load(f)["feature_means"]
    logger.info("model loaded successfully")
except Exception as e:
    # Without a model the service cannot do its job -> CRITICAL
    logger.critical(f"service cannot start, model failed to load: {e}")
    raise

# Keep the last 200 live inputs so we can check them for drift
live_inputs = deque(maxlen=200)
SUMMARY_EVERY = 10  # log a monitoring summary every N requests


@app.post("/predict")
def predict():
    start = time.perf_counter()
    try:
        data = request.get_json()
        logger.debug(f"request features: {data}")  # detailed — only visible at DEBUG

        x = np.array(data["features"], dtype=float).reshape(1, -1)
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {x.shape[1]}")

        # Unusual but valid input: handled, but worth a WARNING
        if (x < 0).any():
            logger.warning(f"negative feature value received: {x.tolist()}")

        pred = int(model.predict(x)[0])
        ms = (time.perf_counter() - start) * 1000
        logger.info(f"prediction={pred} latency_ms={ms:.1f}")

        live_inputs.append(x[0])
        record(ms, ok=True, prediction=pred)
        return jsonify({"prediction": pred, "latency_ms": round(ms, 2)})

    except Exception as e:
        ms = (time.perf_counter() - start) * 1000
        logger.error(f"prediction failed: {e!r} latency_ms={ms:.1f}")
        record(ms, ok=False)
        return jsonify({"error": str(e)}), 400

    finally:
        # Periodic monitoring summary every SUMMARY_EVERY requests
        if stats["requests"] % SUMMARY_EVERY == 0:
            summary()


@app.get("/stats")
def get_stats():
    return jsonify(summary())


@app.get("/drift")
def get_drift():
    if len(live_inputs) < 10:
        logger.warning(f"not enough live data for drift check ({len(live_inputs)} samples)")
        return jsonify({"error": "need at least 10 samples"}), 400
    arr = np.array(live_inputs)
    result = {}
    for i, name in enumerate(FEATURE_NAMES):
        rel = drift_check(TRAIN_MEANS[i], arr[:, i], tol=0.25, feature_name=name)
        result[name] = {"relative_change": round(rel, 3), "drift": rel > 0.25}
    return jsonify(result)


if __name__ == "__main__":
    logger.info("starting ml-service on port 5000")
    app.run(port=5000)
