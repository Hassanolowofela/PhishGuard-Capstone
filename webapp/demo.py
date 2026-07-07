"""Command-line smoke test for the PhishGuard inference module.

Run from the webapp/ directory (so the model artifacts are found under ../models):

    python demo.py

It classifies a few example emails and prints the verdict, the calibrated
phishing probability, and the top contributing features for each.
"""
import json
import os

from inference import PhishGuardModel

SAMPLES = [
    {
        "name": "Obvious phishing lure",
        "subject": "URGENT: Your account will be suspended",
        "body": (
            "Dear customer, we detected unusual activity. Verify your password "
            "now at http://secure-login-update.example.com or your account will "
            "be permanently suspended within 24 hours. Act now to claim your refund."
        ),
    },
    {
        "name": "Ordinary legitimate note",
        "subject": "Re: lunch on Thursday",
        "body": (
            "Hi Sam, Thursday works for me. Let us meet at the usual place around "
            "noon. I will bring the notes from the last meeting. See you then."
        ),
    },
]


def main():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    model = PhishGuardModel(models_dir)
    print(f"Loaded model with {len(model.feature_names)} features "
          f"({model.n_struct} structural + {len(model.feature_names) - model.n_struct} content)\n")
    for s in SAMPLES:
        r = model.predict(s["body"], s["subject"])
        print("=" * 70)
        print(f"{s['name']}")
        print(f"  subject: {s['subject']}")
        print(f"  verdict: {r['label'].upper()}  (phishing probability {r['probability']})")
        print("  top contributing features:")
        for f in r["top_features"]:
            sign = "+" if f["contribution"] > 0 else "-"
            print(f"    {sign} {f['feature']:<24} {f['contribution']:+.4f}  "
                  f"[{f['kind']}, toward {f['pushes']}]")
        print()


if __name__ == "__main__":
    main()
