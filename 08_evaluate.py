#!/usr/bin/env python3
"""Phase 3, Step 4: honest evaluation on the untouched TEST set.

Loads the calibrated model selected in Step 3 and evaluates it on the held-out
test set of the clean (leakage-controlled) 06 split. It reports:

1. Test metrics at the default 0.5 threshold: accuracy, precision, recall, F1,
   ROC-AUC, PR-AUC, the confusion matrix, and the false-positive / false-negative
   rates.
2. Threshold tuning: a threshold chosen on the VALIDATION set (best F1, and a
   precision-oriented threshold) applied to the test set, to trade a little recall
   for higher precision (fewer false alarms on legitimate mail).
3. Leakage comparison: the same model configuration trained and tested on the OLD
   leaky 04 split versus the clean 06 split, to quantify how much the near
   duplicates were inflating the score.
4. Feature ablation: structural-only vs TF-IDF-only vs combined, on the clean
   split, to see which features carry the signal.

Output: data/08_evaluation_report.json (and a printed summary).
"""
import os, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")
SEED = 42

X = sp.load_npz(os.path.join(DATA, "03_combined.npz")).tocsr()
y = np.load(os.path.join(DATA, "03_labels.npy"))
clean = np.load(os.path.join(DATA, "06_split_indices.npz"))
tr, va, te = clean["train"], clean["val"], clean["test"]
model = joblib.load(os.path.join(MODELS, "07_classifier.joblib"))   # calibrated best (Step 3)

def metrics_at(yt, proba, thr):
    yp = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    return {
        "threshold": round(float(thr), 3),
        "accuracy": round(accuracy_score(yt, yp), 4),
        "precision": round(precision_score(yt, yp), 4),
        "recall": round(recall_score(yt, yp), 4),
        "f1": round(f1_score(yt, yp), 4),
        "false_positive_rate": round(fp / (fp + tn), 4),
        "false_negative_rate": round(fn / (fn + tp), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

proba_te = model.predict_proba(X[te])[:, 1]
proba_va = model.predict_proba(X[va])[:, 1]
yte, yva = y[te], y[va]

report = {"split": "06 clean", "test_n": int(len(te)), "selected_model": "logistic regression (calibrated)"}
report["threshold_independent"] = {
    "roc_auc": round(roc_auc_score(yte, proba_te), 4),
    "pr_auc": round(average_precision_score(yte, proba_te), 4),
}
report["test_default_0.5"] = metrics_at(yte, proba_te, 0.5)

# choose thresholds on validation, apply to test
grid = np.linspace(0.05, 0.95, 181)
vf1 = [f1_score(yva, (proba_va >= t).astype(int)) for t in grid]
thr_bestf1 = float(grid[int(np.argmax(vf1))])
vprec = np.array([precision_score(yva, (proba_va >= t).astype(int), zero_division=0) for t in grid])
ok = np.where(vprec >= 0.97)[0]
thr_p97 = float(grid[ok[0]]) if len(ok) else 0.95
report["test_tuned_best_f1"] = metrics_at(yte, proba_te, thr_bestf1)
report["test_tuned_precision_0.97"] = metrics_at(yte, proba_te, thr_p97)

# leakage comparison: same config on leaky 04 vs clean 06
def lr(): return LogisticRegression(max_iter=2000, solver="liblinear", C=1)
def quick(Xf, itr, ite):
    m = lr().fit(Xf[itr], y[itr]); yp = m.predict(Xf[ite])
    return {"accuracy": round(accuracy_score(y[ite], yp), 4), "f1": round(f1_score(y[ite], yp), 4),
            "roc_auc": round(roc_auc_score(y[ite], m.decision_function(Xf[ite])), 4)}
leaky = np.load(os.path.join(DATA, "04_split_indices.npz"))
clean_res = quick(X, tr, te)
leaky_res = quick(X, leaky["train"], leaky["test"])
report["leakage_comparison"] = {
    "clean_06_test": clean_res, "leaky_04_test": leaky_res,
    "f1_inflation": round(leaky_res["f1"] - clean_res["f1"], 4),
    "accuracy_inflation": round(leaky_res["accuracy"] - clean_res["accuracy"], 4),
}

# ablation on the clean split: structural-only vs tfidf-only vs combined
struct = pd.read_csv(os.path.join(DATA, "03_structural.csv"))
Xs = struct[[c for c in struct.columns if c != "label"]].to_numpy(dtype=np.float64)
Xt = sp.load_npz(os.path.join(DATA, "03_tfidf.npz")).tocsr()
def quick_struct(itr, ite):
    m = make_pipeline(StandardScaler(), lr()).fit(Xs[itr], y[itr]); yp = m.predict(Xs[ite])
    sc = m.decision_function(Xs[ite])
    return {"accuracy": round(accuracy_score(y[ite], yp), 4), "f1": round(f1_score(y[ite], yp), 4),
            "roc_auc": round(roc_auc_score(y[ite], sc), 4)}
report["ablation_clean_test"] = {
    "structural_only_18": quick_struct(tr, te),
    "tfidf_only_5000": quick(Xt, tr, te),
    "combined_5018": clean_res,
}

with open(os.path.join(DATA, "08_evaluation_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
