# PhishGuard Web Application (Phase 4)

This folder holds the Phase 4 web application that turns the Phase 3 model into
the actual PhishGuard tool: a user submits an email and receives a phishing
verdict, a confidence score, and a plain-language explanation of the decision.

Phase 4 is being built in layers. This first layer is the **inference module**,
which is the engine the web interface will call. The web routes and templates
are the next layer.

## Files

| File | Purpose |
|------|---------|
| `features.py` | Reproduces the exact training preprocessing (stage 02) and the 18 structural features (stage 03), and assembles the 5,018-dimension feature row for one email. |
| `inference.py` | `PhishGuardModel`: loads the fitted vectorizer and the calibrated scaled classifier, predicts label and probability, and returns the top contributing features as an exact linear attribution. |
| `app.py` | Flask server: serves the interactive page and a JSON `/api/analyze` endpoint that calls the inference module. |
| `templates/index.html` | The interactive single-page interface (live analysis, confidence meter, sensitivity slider, feature bars, highlighted email). |
| `demo.py` | Command-line smoke test over a couple of example emails. |
| `try_email.py` | Edit-and-run script to classify one email of your own. |
| `requirements.txt` | Serving dependencies (scikit-learn pinned to the model's version). |

## Why the features must match training

The model was trained on a specific feature layout: 18 interpretable structural
features followed by a 5,000-term TF-IDF vector, for 5,018 features in total.
`features.py` deliberately copies the training-time cleaning and feature logic so
that an email submitted at serving time becomes exactly the same kind of vector.
Any drift here would silently reduce accuracy, so the two must stay in sync.

## How the explanation works

The final model is linear, so each feature's contribution to a decision is
`coefficient * scaled_value`. `inference.py` reads the coefficients and the
per-feature scales from the calibrated model's underlying MaxAbsScaler and
LogisticRegression sub-estimators and reports the features with the largest
absolute contribution. This is the same quantity SHAP reports for a linear
model, computed exactly and instantly, with no extra dependency.

## Requirements

The module needs the Phase 3 artifacts, which are expected one level up in
`../models/`:

- `tfidf_vectorizer.joblib`
- `09_classifier_scaled.joblib`

## Run the smoke test

```bash
# from the project root, in your virtual environment
pip install -r webapp/requirements.txt
cd webapp
python demo.py
```

Expected output is a verdict, a calibrated phishing probability, and the top
contributing features for each sample email.

## Using it from your own code

```python
from inference import PhishGuardModel

model = PhishGuardModel("../models")
result = model.predict(
    body="Verify your account now or it will be suspended...",
    subject="URGENT: action required",
)
print(result["label"], result["probability"])
for f in result["top_features"]:
    print(f["feature"], f["contribution"], f["pushes"])
```

## The model: Safe / Scam / Malware

The app serves a three-class model (Safe, Scam, Malware) trained by
`20_train_multiclass.py` on the Zenodo multiclass corpus (single source, to avoid a
dataset artifact). Generate the model once (the artifacts are not committed):

```bash
# from the project root, in your virtual environment
python 20_train_multiclass.py
```

This writes `models/20_multiclass_model.joblib` and
`models/20_multiclass_vectorizer.joblib` and prints honest, leakage-controlled
per-class metrics (macro-F1 about 0.93). The Malware class is the weakest and is a
small-data proof of concept that reads malware-lure wording, not attachments.

## Run the web app

```bash
pip install -r webapp/requirements.txt
cd webapp
python app.py
```

Then open http://127.0.0.1:5000 in a browser. The interface lets you:

- enter the sender address, subject, and body, with a required consent checkbox;
- get an instant Safe / Scam / Malware verdict with a confidence score and the full
  per-class probabilities (analysis also runs automatically a moment after you stop typing);
- read plain-language red flags (urgency, links, money, shouting, sender domain
  issues, and the wording that drove the verdict) and a recommended action;
- load a Safe, Scam, or Malware example with one click.

Email content is processed in memory only and is not stored, matching the privacy
design in the software design document. The Malware verdict carries a clear note that
it is a small-data proof of concept based on wording, not attachment scanning.
