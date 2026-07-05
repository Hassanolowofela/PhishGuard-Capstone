#!/usr/bin/env python3
"""Phase 3, Step 4b: fix feature scaling, retrain, and re-check the ablation.

The Step 4 ablation showed that TF-IDF alone reached ~0.995 F1 while the combined
matrix only reached ~0.95. The cause is scale: the 18 structural features have very
large, unscaled magnitudes next to the 0-to-1 TF-IDF values, so they dominate the
regularized linear model and drag it down.

This script wraps the model in a scaler so every feature is on a comparable scale:
  - Combined and TF-IDF: MaxAbsScaler (suits sparse data, preserves sparsity).
  - Structural only: StandardScaler (dense).

It retrains and tunes Logistic Regression on the clean 06 split, evaluates on
validation and the untouched test set (default and tuned thresholds), calibrates
and saves the improved model, and re-runs the ablation with scaling so the
before/after is explicit.

Outputs: models/09_classifier_scaled.joblib and data/09_rescale_report.json.
"""
import os, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, MODELS = os.path.join(HERE, "data"), os.path.join(HERE, "models")
SEED = 42

X = sp.load_npz(os.path.join(DATA, "03_combined.npz")).tocsr()
y = np.load(os.path.join(DATA, "03_labels.npy"))
c = np.load(os.path.join(DATA, "06_split_indices.npz"))
tr, va, te = c["train"], c["val"], c["test"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def fit_lr(scaler, Xf, itr):
    gs = GridSearchCV(make_pipeline(scaler, LogisticRegression(max_iter=2000, solver="liblinear")),
                      {"logisticregression__C": [0.1, 1, 10]}, scoring="f1", cv=cv, n_jobs=-1)
    gs.fit(Xf[itr], y[itr])
    return gs.best_estimator_, gs.best_params_["logisticregression__C"]

def score(model, Xf, ite):
    yp = model.predict(Xf[ite])
    sc = model.decision_function(Xf[ite]) if hasattr(model, "decision_function") else model.predict_proba(Xf[ite])[:, 1]
    return {"accuracy": round(accuracy_score(y[ite], yp), 4), "f1": round(f1_score(y[ite], yp), 4),
            "roc_auc": round(roc_auc_score(y[ite], sc), 4)}

def metrics_at(yt, proba, thr):
    yp = (proba >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    return {"threshold": round(float(thr), 3), "accuracy": round(accuracy_score(yt, yp), 4),
            "precision": round(precision_score(yt, yp), 4), "recall": round(recall_score(yt, yp), 4),
            "f1": round(f1_score(yt, yp), 4), "false_positive_rate": round(fp/(fp+tn), 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}

# 1) scaled combined model (the new production candidate)
best, bestC = fit_lr(MaxAbsScaler(), X, tr)
proba_va = best.predict_proba(X[va])[:, 1]
proba_te = best.predict_proba(X[te])[:, 1]
grid = np.linspace(0.05, 0.95, 181)
thr_bf1 = float(grid[int(np.argmax([f1_score(y[va], (proba_va >= t).astype(int)) for t in grid]))])

# calibrate and save
cal = CalibratedClassifierCV(best, method="sigmoid", cv=5)
cal.fit(X[tr], y[tr])
os.makedirs(MODELS, exist_ok=True)
joblib.dump(cal, os.path.join(MODELS, "09_classifier_scaled.joblib"))

# 2) ablation with scaling
struct = pd.read_csv(os.path.join(DATA, "03_structural.csv"))
Xs = struct[[k for k in struct.columns if k != "label"]].to_numpy(dtype=np.float64)
Xt = sp.load_npz(os.path.join(DATA, "03_tfidf.npz")).tocsr()
m_struct, _ = fit_lr(StandardScaler(), Xs, tr)
m_tfidf, _ = fit_lr(MaxAbsScaler(), Xt, tr)
# unscaled combined for direct before/after
m_unscaled = LogisticRegression(max_iter=2000, solver="liblinear", C=1).fit(X[tr], y[tr])

report = {
    "split": "06 clean", "test_n": int(len(te)),
    "scaled_combined": {"best_C": bestC,
        "val": score(best, X, va),
        "test_default_0.5": metrics_at(y[te], proba_te, 0.5),
        "test_tuned_best_f1": metrics_at(y[te], proba_te, thr_bf1),
        "roc_auc": round(roc_auc_score(y[te], proba_te), 4),
        "pr_auc": round(average_precision_score(y[te], proba_te), 4)},
    "ablation_scaled_test": {
        "structural_only_18": score(m_struct, Xs, te),
        "tfidf_only_5000": score(m_tfidf, Xt, te),
        "combined_5018_scaled": score(best, X, te)},
    "before_after_combined": {
        "unscaled_test": score(m_unscaled, X, te),
        "scaled_test": score(best, X, te)},
}
with open(os.path.join(DATA, "09_rescale_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
