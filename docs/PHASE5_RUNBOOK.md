# PhishGuard Phase 5: Local Run Book

A step by step guide to running everything on your own machine, seeing the web app,
reproducing the Phase 5 evaluations, applying the generalization fix, and viewing the
results as charts. Commands are shown for Windows PowerShell from the project root.

## Prerequisites

You need Python 3.12 and Git. Check with:

```powershell
python --version
git --version
```

Open PowerShell in the project folder (the one that contains `webapp`, `tests`, and
the numbered scripts).

## 1. Set up the environment

Create an isolated virtual environment and install the dependencies once.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r webapp\requirements.txt
pip install pytest matplotlib
```

You should see `(.venv)` at the start of your prompt. Everything below assumes it is
active. To leave it later, run `deactivate`.

## 2. See it work: run the web app

This is the best way to understand the system. If the model files are missing, build
the three class model first (a one time step), then start the server.

```powershell
python 20_train_multiclass.py          # only needed if models\ is empty
cd webapp
python app.py
```

Open http://127.0.0.1:5000 in your browser. Paste a sender, subject, and body, or use
the built in Safe, Scam, and Malware example buttons, then click Analyze. You will see
the verdict, a confidence donut, the three class probabilities, the red flags, and the
recommended action. Press Ctrl+C in the terminal to stop, then `cd ..` to return to the
project root.

## 3. Run the Phase 5 evaluations

Each script writes a JSON report into `data\`. Run them from the project root.

```powershell
python 30_cross_corpus_eval.py     # cross-corpus generalization of the binary model
python 32_robustness_probes.py     # adversarial and evasion probes on the live model
```

`30` writes `data\30_cross_corpus_report.json`. Open it and compare the out of corpus
F1 to the in corpus 0.9957 to read the generalization gap. `32` writes
`data\32_robustness_report.json`; the key fields are `evasion_rate_to_safe` and the per
seed confidence changes.

## 4. Apply the fix and build the generalized model

```powershell
python 33_generalize_binary.py
```

This trains the improved model and writes `models\33_binary_generalized.joblib` plus
`data\33_generalization_report.json`. On your machine you can raise the two knobs at
the top of the script for tighter numbers: set `CEAS_SAMPLE` higher (up to about
34000, the full corpus) and `CHAR_MAX_FEATURES` higher (for example 12000). It will
take longer but the conclusion stays the same.

## 5. Visualize the results

Turn the JSON reports into charts.

```powershell
python 34_visualize_phase5.py
```

This writes three PNGs into `screenshots\phase5\`: the cross-corpus gap, the
generalization fix (word baseline vs improved features vs the union model), and the
robustness chart. Open the folder and view them.

## 6. Run the tests

```powershell
python -m pytest tests\ -v
```

You should see the unit tests and the Flask integration tests pass. The three model
path tests run for real here because your environment has the pinned scikit-learn.

## Quick reference: what each output means

| File | What it tells you |
|------|-------------------|
| `data\30_cross_corpus_report.json` | How the CEAS-trained binary model does on an unseen corpus (the gap) |
| `data\32_robustness_report.json` | Whether evasion tricks flip a threat to Safe, and how confidence moves |
| `data\33_generalization_report.json` | Word baseline vs improved features, and the union model that fixes over-flagging |
| `models\33_binary_generalized.joblib` | The saved generalized model (bundle: model, vectorizer, threshold, columns) |
| `screenshots\phase5\*.png` | The three result charts |
| `docs\PHASE5_REPORT.md` | The written synthesis of all four Phase 5 tests |
| `docs\PHASE5_GENERALIZATION.md` | The written explanation of the generalization fix |

## Using the generalized model in your own code

```python
import joblib, scipy.sparse as sp, numpy as np
b = joblib.load("models/33_binary_generalized.joblib")
model, vec, thr, cols = b["model"], b["vectorizer"], b["threshold"], b["struct_cols"]
# build features the same way the script does (structural[cols] then char TF-IDF),
# then: proba = model.predict_proba(X)[:, 1]; verdict = "phishing" if proba >= thr else "legitimate"
```

## Troubleshooting

If `python app.py` cannot find a model, run `python 20_train_multiclass.py` first. If a
script cannot find `data\02_clean.csv` or `data\phishing_nlp_dataset.csv`, those inputs
must be present in `data\` (they are produced by the Phase 2 pipeline and the dataset
acquisition step). If activation is blocked, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in the same window.
