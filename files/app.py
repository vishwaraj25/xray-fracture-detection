"""
app.py — Flask backend
Serves the web UI and exposes prediction + metrics endpoints.

Usage:
    python app.py
    Open http://localhost:5000
"""

import os, json, base64, tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import predict

UI_DIR      = os.path.join(os.path.dirname(__file__), "..", "ui")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")

app = Flask(__name__, static_folder=UI_DIR, static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "No metrics yet — run train_svm.py first."}), 404
    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    tmp = None
    try:
        if "file" in request.files:
            f = request.files["file"]
            suffix = os.path.splitext(f.filename)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
                f.save(t.name); tmp = t.name
        elif request.is_json and "image" in request.json:
            data = request.json["image"]
            if "," in data: data = data.split(",", 1)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t:
                t.write(base64.b64decode(data)); tmp = t.name
        else:
            return jsonify({"error": "Send a file or base64 image."}), 400

        return jsonify(predict(tmp))
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


if __name__ == "__main__":
    print("BoneAI running → http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
