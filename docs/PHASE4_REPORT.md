# PhishGuard - Phase 4: Web Application

Phase 4 turns the Phase 3 model into the actual PhishGuard tool: a web application
where a user pastes an email and receives a verdict, a confidence score, and a
plain-language explanation of the decision. The application lives in the `webapp/`
folder and is built with Flask and only open source libraries, matching the design
in the software design document.

## What the application does

A user enters an email (sender, subject, and body) and gives consent. The app
returns one of three verdicts, **Safe**, **Scam**, or **Malware**, with a confidence
score, a short list of plain-language red flags that explain the decision, a few
transparent sender checks, and a recommended action. Email content is processed in
memory only and is never stored, which is the process-and-discard privacy design
carried over from the SDD.

## Architecture

The application follows the layered client and server design specified earlier. The
browser holds the interface and performs no classification. It sends the email to a
Flask API endpoint (`/api/analyze`), which validates the input, calls the inference
module, and returns a JSON result that the interface renders. The persistence layer
holds only the model artifacts; no email content or logs are written.

- `features.py` reproduces the exact training preprocessing (stage 02) and the 18
  interpretable structural features (stage 03), so a submitted email becomes exactly
  the feature vector the model was trained on.
- `inference.py` loads the model and vectorizer, predicts the verdict and the
  per-class probabilities, and produces the explanation as an exact linear
  attribution over the model coefficients (the same quantity SHAP reports for a
  linear model), plus a few transparent sender heuristics.
- `app.py` is the Flask server and JSON endpoint.
- `templates/index.html` is the interactive single-page interface.

## From two classes to three: Safe, Scam, Malware

Phase 3 produced a binary phishing-versus-legitimate model. Phase 4 extends this to a
three-class model so the tool can separate ordinary mail, social-engineering scams,
and malware-delivery lures. The model is trained by `20_train_multiclass.py` on the
Zenodo Multiclass NLP Dataset for Phishing and Social Engineering Threat Detection
(CC BY 4.0), mapped to three classes: NOT-Malicious becomes Safe, the phishing and
social-engineering categories become Scam, and Malware stays Malware.

The dataset is used on its own rather than merged with CEAS_08. Because it contains
all three classes from a single source, training on it alone avoids the artifact
where a model learns which dataset a message came from instead of a real threat
signal. After cleaning and de-duplication the corpus is 606 messages (Safe 171,
Scam 358, Malware 77).

Honest, leakage-controlled (near-duplicate group-aware) held-out results:

| Class | Precision | Recall | F1 |
|-------|----------:|-------:|---:|
| Safe | 0.98 | 1.00 | 0.99 |
| Scam | 0.96 | 0.98 | 0.97 |
| Malware | 0.92 | 0.75 | 0.83 |
| Macro-F1 | | | **0.93** |

The most important safety property holds: there is no Safe-Malware confusion in
testing, so the model never labels a malware email as Safe. The Malware class is the
weakest and is a small-data proof of concept. It detects the **wording** of a
malware-delivery lure, for example attachment and macro prompts, and it does **not**
scan attachments. The interface states this clearly on every Malware verdict.

## The interface

The interface is designed for a non-technical user and is deliberately interactive:
an animated probability donut and a count-up confidence score, plain-language red
flags, a recommended action, per-class probabilities, one-click Safe, Scam, and
Malware examples, a light and dark theme, an in-memory session history that clears
when the tab closes, and copy and download of the result. Additional sections explain
how the tool works, what each verdict means, and how to spot phishing without a tool.

## Requirements coverage and two deferred decisions

The user-facing functional requirements are met: the app accepts an email through a
web interface (FR1), requires consent (FR2), classifies the email (FR3), returns a
confidence score (FR4), and displays the top contributing signals (FR5).

Two non-core requirements were deliberately not implemented in this prototype, and
the reasons are recorded here for honesty and traceability:

- **Administrative access (FR6).** An authenticated admin panel for model management
  and metrics was scoped out of the prototype. The model is versioned and retrained
  through `20_train_multiclass.py` under version control, which serves the same need
  at prototype scale without adding an authentication surface. It is noted as future
  work rather than built.
- **Operational logging and monitoring (FR7).** This was intentionally not
  implemented because it is in tension with the process-and-discard privacy promise
  that the whole design is built on. The app writes nothing. If monitoring is added
  later, it would record only de-identified aggregate metrics, such as counts of
  verdicts, and never email content. Choosing privacy over monitoring at this stage
  is a deliberate design decision, not an omission.

## How to run

```bash
# from the project root, in your virtual environment
python 20_train_multiclass.py          # generate models/20_multiclass_*.joblib
pip install -r webapp/requirements.txt
cd webapp
python app.py                          # open http://127.0.0.1:5000
```

The model artifacts and the dataset are not committed; they are regenerated by the
script, consistent with the rest of the repository.

## Limitations and next steps

The Malware class is a small-data proof of concept, and all training data comes from
a single corpus, so cross-corpus generalization remains the priority robustness test.
The next phases are Phase 5, automated testing and evaluation of the application and a
cross-dataset generalization test, and deployment to a free-tier cloud service, after
which Phase 4 and Phase 5 are folded into the formal capstone documents.
