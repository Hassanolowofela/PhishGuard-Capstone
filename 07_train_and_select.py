#!/usr/bin/env python3
"""Phase 3, Step 3: train and select a model on the clean (leakage-controlled) split.

Trains three candidates that are well suited to high-dimensional sparse text
features, tunes each with cross-validation on the TRAINING set, evaluates them on
the VALIDATION set, selects the best by F1, calibrates its probabilities, and
saves the model. The held-out TEST set (06 split) is NOT touched here; it is kept
for the final evaluation step.

Candidates:
  - Logistic Regression (linear, strong baseline for TF-IDF)
  - Linear SVM (LinearSVC; calibrated for probabilities)
  - Complement Naive Bayes (a classic, fast text classifier)

Outputs: models/07_classifier.joblib (calibrated best model) and
data/07_model_selection_report.json.
"""
import os, json, time
import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")
os.makedirs(MODELS, exist_ok=True)
SEED = 42

X = sp.load_npz(os.path.join(DATA, "03_combined.npz")).tocsr()
y = np.load(os.path.join(DATA, "03_labels.npy"))
idx = np.load(os.path.join(DATA, "06_split_indices.npz"))   # clean split
tr, va = idx["train"], idx["val"]
Xtr, ytr = X[tr], y[tr]
Xva, yva = X[va], y[va]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
candidates = {
    "logreg": (LogisticRegression(max_iter=2000, solver="liblinear"), {"C": [0.1, 1, 10]}),
    "linear_svm": (LinearSVC(dual=False, max_iter=5000), {"C": [0.1, 1, 10]}),
    "complement_nb": (ComplementNB(), {"alpha": [0.1, 0.5, 1.0]}),
}

def val_scores(model):
    yp = model.predict(Xva)
    if hasattr(model, "predict_proba"):
        ys = model.predict_proba(Xva)[:, 1]
    elif hasattr(model, "decision_function"):
        ys = model.decision_function(Xva)
    else:
        ys = yp
    return {
        "val_accuracy": round(accuracy_score(yva, yp), 4),
        "val_precision": round(precision_score(yva, yp), 4),
        "val_recall": round(recall_score(yva, yp), 4),
        "val_f1": round(f1_score(yva, yp), 4),
        "val_roc_auc": round(roc_auc_score(yva, ys), 4),
    }

results, best_name, best_est, best_f1 = {}, None, None, -1.0
for name, (est, grid) in candidates.items():
    t0 = time.time()
    gs = GridSearchCV(est, grid, scoring="f1", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr)
    m = {"best_params": gs.best_params_, "cv_f1": round(float(gs.best_score_), 4)}
    m.update(val_scores(gs.best_estimator_))
    m["fit_seconds"] = round(time.time() - t0, 1)
    results[name] = m
    print(f"  {name}: val_f1={m['val_f1']} val_auc={m['val_roc_auc']} ({m['fit_seconds']}s)", flush=True)
    if m["val_f1"] > best_f1:
        best_f1, best_name, best_est = m["val_f1"], name, gs.best_estimator_

# calibrate the winner so the confidence score is meaningful, then save
calibrated = CalibratedClassifierCV(best_est, method="sigmoid", cv=5)
calibrated.fit(Xtr, ytr)
joblib.dump(calibrated, os.path.join(MODELS, "07_classifier.joblib"))

report = {
    "seed": SEED,
    "split": "06 (clean, leakage-controlled)",
    "train_n": int(len(tr)), "val_n": int(len(va)),
    "candidates": results,
    "selected_model": best_name,
    "selected_val_f1": best_f1,
    "calibration": "sigmoid, cv=5",
    "note": "Test set is untouched; final evaluation happens in the next step.",
}
with open(os.path.join(DATA, "07_model_selection_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
