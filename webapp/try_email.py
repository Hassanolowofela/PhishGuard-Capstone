"""Quick way to classify one email with PhishGuard.

Edit BODY and SUBJECT below, then run from the webapp/ folder:

    python try_email.py
"""
from inference import PhishGuardModel

# ---- edit these two ----
SUBJECT = "Action required"
BODY = "Your mailbox is full. Click http://reset-now.example.com to keep your account active."
# ------------------------

model = PhishGuardModel("../models")
result = model.predict(body=BODY, subject=SUBJECT)

print(f"verdict: {result['label'].upper()}  (phishing probability {result['probability']})")
print("top contributing features:")
for f in result["top_features"]:
    print(f"  {f['contribution']:+.4f}  {f['feature']:<24} [{f['kind']}, toward {f['pushes']}]")
