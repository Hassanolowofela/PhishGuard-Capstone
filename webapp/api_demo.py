"""End-to-end API demonstration for PhishGuard (reproduces Figure 4.3).

Run from the webapp/ directory so the model artifacts are found under ../models:

    cd webapp
    python api_demo.py

It exercises the two HTTP endpoints through Flask's test client (no separate
server needed) and prints the health check and a live analysis request, so the
output can be captured as a screenshot for the report.
"""
import json
import warnings

warnings.filterwarnings("ignore")  # keep the console output clean for the screenshot

from app import app

# A realistic scam sample. Edit these three fields to analyze a different email.
SAMPLE = {
    "sender": "security@paypa1-verify.com",
    "subject": "Urgent: verify your account now",
    "body": (
        "Your account will be suspended in 24 hours. "
        "Click http://paypa1-verify.com to confirm your password now."
    ),
}


def main():
    client = app.test_client()

    print("end to end: health check and a live API request\n")

    h = client.get("/health")
    print(f"GET /health   -> {h.status_code} "
          f"{json.dumps(h.get_json(), sort_keys=True)}")

    r = client.post("/api/analyze", json=SAMPLE)
    data = r.get_json() or {}
    print(f"POST /api/analyze -> {r.status_code}")
    if "verdict" in data:
        print(f"   verdict      : {data['verdict']} | confidence: {data['confidence']}")
        print(f"   probabilities: {data['probabilities']}")
        print(f"   flags        : {len(data['flags'])} red flags returned")
    else:
        print(f"   response     : {data}")


if __name__ == "__main__":
    main()
