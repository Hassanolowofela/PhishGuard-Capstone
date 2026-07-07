"""PhishGuard web application (Phase 4) - Flask server.

Serves an interactive single-page interface and a JSON analysis endpoint that
calls the Phase 3 model through the inference module. Email content is processed
in memory only and is never stored, matching the privacy design.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""
import os

from flask import Flask, jsonify, render_template, request

from inference import get_multiclass_model

MODELS_DIR = os.environ.get("PHISHGUARD_MODELS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models")
MAX_BODY = 50000
MAX_SUBJECT = 2000

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    sender = (data.get("sender") or "").strip()[:MAX_SUBJECT]
    subject = (data.get("subject") or "").strip()[:MAX_SUBJECT]
    body = (data.get("body") or "").strip()[:MAX_BODY]

    if not subject and not body and not sender:
        return jsonify({"empty": True})

    try:
        model = get_multiclass_model(MODELS_DIR)
        result = model.predict(body=body, subject=subject, sender=sender)
        result["empty"] = False
        return jsonify(result)
    except Exception as exc:  # never leak a stack trace to the browser
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
