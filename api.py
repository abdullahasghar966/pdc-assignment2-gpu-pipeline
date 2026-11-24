"""Web API for the digit model.

GET  /accuracy  -> accuracy of the currently loaded model
POST /predict   -> body {"features": [64 numbers]} returns the predicted digit
POST /reload    -> reloads models/model.pt after a retrain (called by Jenkins)
"""
import json
import os

import torch
import torch.nn as nn
from flask import Flask, jsonify, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "model.pt")
METRICS_PATH = os.path.join(BASE, "models", "metrics.json")

app = Flask(__name__)
model = None


def build_model():
    return nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


def load_model():
    global model
    net = build_model()
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()
    model = net


def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.route("/accuracy", methods=["GET"])
def accuracy():
    return jsonify(load_metrics())


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    features = body.get("features")
    if not isinstance(features, list) or len(features) != 64:
        return jsonify({"error": "features must be a list of 64 numbers"}), 400
    try:
        x = torch.tensor([[float(v) / 16.0 for v in features]])
    except (TypeError, ValueError):
        return jsonify({"error": "features must be numbers"}), 400
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    digit = int(probs.argmax().item())
    return jsonify({"digit": digit, "confidence": float(probs[digit])})


@app.route("/reload", methods=["POST"])
def reload_model():
    load_model()
    return jsonify({"status": "reloaded", **load_metrics()})


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
