#!/usr/bin/env python3
"""Phase 3, Step 5: explain the scaled model with SHAP.

The Step 4b model (MaxAbsScaler + Logistic Regression on the 5,018 combined
features) is a linear model, so we use shap.LinearExplainer, which gives EXACT
Shapley values for a linear model at a fraction of the cost of KernelSHAP.

SHAP values are expressed in the model's log-odds (decision_function) space:
a positive value pushes the email toward "phishing", a negative value toward
"legitimate". Calibration (done in step 09) only rescales the probability, so
the sign and ranking of the SHAP attributions carry straight over.

What it produces
  data/10_shap_report.json      global top features + worked examples
  data/10_shap_top_features.png  bar chart of the top-20 mean |SHAP|
  data/10_shap_beeswarm.png      SHAP beeswarm over the test set

The point of the step: because we kept the 18 interpretable structural features
(and fixed their scaling in 09), the model can say WHY an email is flagged in
human terms - url counts, urgency words, money symbols - not just a TF-IDF blob.

Run:
  pip install shap        # one-time; pulls in numba/llvmlite
  python 10_explain.py
"""
import os, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, MODELS = os.path.join(HERE, "data"), os.path.join(HERE, "models")
SEED = 42
rng = np.random.default_rng(SEED)

# ---- 1) load features, labels, clean split, and feature names ----
X = sp.load_npz(os.path.join(DATA, "03_combined.npz")).tocsr()
y = np.load(os.path.join(DATA, "03_labels.npy"))
c = np.load(os.path.join(DATA, "06_split_indices.npz"))
tr, te = c["train"], c["test"]

struct_cols = [k for k in pd.read_csv(os.path.join(DATA, "03_structural.csv"),
                                      nrows=1).columns if k != "label"]  # 18, in matrix order
tfidf_names = list(joblib.load(os.path.join(MODELS, "tfidf_vectorizer.joblib")
                               ).get_feature_names_out())               # 5000
feature_names = np.array(struct_cols + tfidf_names)
n_struct = len(struct_cols)
assert len(feature_names) == X.shape[1], (len(feature_names), X.shape[1])
is_struct = np.arange(X.shape[1]) < n_struct

# ---- 2) refit the exact step-09 scaled pipeline (deterministic) ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
gs = GridSearchCV(
    make_pipeline(MaxAbsScaler(), LogisticRegression(max_iter=2000, solver="liblinear")),
    {"logisticregression__C": [0.1, 1, 10]}, scoring="f1", cv=cv, n_jobs=-1)
gs.fit(X[tr], y[tr])
pipe = gs.best_estimator_
scaler = pipe.named_steps["maxabsscaler"]
clf = pipe.named_steps["logisticregression"]
print(f"Refit scaled model, best C = {gs.best_params_['logisticregression__C']}")

# ---- 3) SHAP LinearExplainer (interventional => exact, uses a background sample)
Xtr_s = scaler.transform(X[tr])
Xte_s = scaler.transform(X[te])
bg_idx = rng.choice(Xtr_s.shape[0], size=min(1000, Xtr_s.shape[0]), replace=False)
explainer = shap.LinearExplainer(clf, Xtr_s[bg_idx],
                                 feature_perturbation="interventional")
shap_te = explainer.shap_values(Xte_s)          # (n_test, n_features), log-odds
shap_te = np.asarray(shap_te)
base_value = float(np.ravel(explainer.expected_value)[0])

# ---- 4) global importance = mean |SHAP| over the test set ----
mean_abs = np.abs(shap_te).mean(axis=0)
order = np.argsort(mean_abs)[::-1]

def top(indices, k):
    return [{"feature": str(feature_names[i]),
             "kind": "structural" if is_struct[i] else "tfidf",
             "mean_abs_shap": round(float(mean_abs[i]), 5),
             "mean_signed_shap": round(float(shap_te[:, i].mean()), 5)}
            for i in indices[:k]]

top20 = top(order, 20)
top_struct = top(order[is_struct[order]], 10)
top_tfidf = top(order[~is_struct[order]], 10)

# ---- 5) three worked examples (per-email top contributions) ----
proba = pipe.predict_proba(X[te])[:, 1]
def example(row):
    contrib = shap_te[row]
    o = np.argsort(np.abs(contrib))[::-1][:8]
    return {"test_row": int(row), "true_label": int(y[te][row]),
            "phishing_probability": round(float(proba[row]), 4),
            "base_value_logodds": round(base_value, 4),
            "top_contributions": [
                {"feature": str(feature_names[i]),
                 "kind": "structural" if is_struct[i] else "tfidf",
                 "shap_logodds": round(float(contrib[i]), 4),
                 "pushes": "phishing" if contrib[i] > 0 else "legitimate"}
                for i in o]}

phish = np.where(y[te] == 1)[0]
legit = np.where(y[te] == 0)[0]
ex_conf_phish = phish[np.argmax(proba[phish])]                 # confident phishing
ex_conf_legit = legit[np.argmin(proba[legit])]                 # confident legit
ex_border = int(np.argmin(np.abs(proba - 0.5)))                # most uncertain
examples = [example(ex_conf_phish), example(ex_conf_legit), example(ex_border)]

# ---- 6) charts ----
plt.figure(figsize=(8, 6))
sel = order[:20][::-1]
colors = ["#c0392b" if is_struct[i] else "#2c7fb8" for i in sel]
plt.barh([feature_names[i] for i in sel], [mean_abs[i] for i in sel], color=colors)
plt.xlabel("mean |SHAP| (log-odds)")
plt.title("PhishGuard - top 20 features by mean |SHAP| (red = structural)")
plt.tight_layout()
plt.savefig(os.path.join(DATA, "10_shap_top_features.png"), dpi=160)
plt.close()

plt.figure()
shap.summary_plot(shap_te, features=Xte_s.toarray() if sp.issparse(Xte_s) else Xte_s,
                  feature_names=feature_names, max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(DATA, "10_shap_beeswarm.png"), dpi=160, bbox_inches="tight")
plt.close()

# ---- 7) report ----
report = {
    "model": "step09 scaled MaxAbsScaler + LogisticRegression",
    "explainer": "shap.LinearExplainer (interventional, exact for linear model)",
    "space": "log-odds (decision_function); positive => phishing",
    "test_n": int(len(te)), "n_features": int(X.shape[1]),
    "n_structural": int(n_struct), "n_tfidf": int(len(tfidf_names)),
    "base_value_logodds": round(base_value, 4),
    "top20_overall": top20,
    "top10_structural": top_struct,
    "top10_tfidf": top_tfidf,
    "structural_share_of_total_importance": round(
        float(mean_abs[is_struct].sum() / mean_abs.sum()), 4),
    "worked_examples": examples,
}
with open(os.path.join(DATA, "10_shap_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps({k: report[k] for k in
                  ["base_value_logodds", "top10_structural",
                   "structural_share_of_total_importance"]}, indent=2))
print("Saved: data/10_shap_report.json, data/10_shap_top_features.png, "
      "data/10_shap_beeswarm.png")
