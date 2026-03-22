"""
app.py — Flask backend
Works locally and on Render (reads PORT from environment).
"""

import os, json, base64, tempfile, sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(BASE)
UI_DIR   = os.path.join(ROOT, "ui")
MODELS_DIR = os.path.join(ROOT, "models")

# Add src to path so imports work
sys.path.insert(0, BASE)
from predict import predict

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
    path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(path):
        return jsonify({"error": "No metrics yet."}), 404
    with open(path) as f:
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
    port = int(os.environ.get("PORT", 5000))
    print(f"BoneAI running → http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
