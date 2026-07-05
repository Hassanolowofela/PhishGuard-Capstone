#!/usr/bin/env python3
"""Phase 3, Step 2: leakage audit.

Before any model is trained, this script checks two things that could make the
Phase 2 baseline look better than it really is:

1. Near-duplicate emails that landed across the train/test (and train/val)
   boundary. For each held-out email it finds the cosine similarity to its most
   similar training email (using the TF-IDF vectors). Many near-identical pairs
   across the boundary would mean the test score is partly memorization.

2. Shortcut features: single structural features that separate phishing from
   legitimate almost perfectly on the training set (measured by univariate
   ROC-AUC). A feature with an AUC near 1.0 deserves a second look.

Inputs are read only. Output: data/05_leakage_report.json (and a printed summary).
"""
import os, sys, json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

y = np.load(os.path.join(DATA, "03_labels.npy"))
sp_idx = np.load(os.path.join(DATA, "04_split_indices.npz"))
tr, va, te = sp_idx["train"], sp_idx["val"], sp_idx["test"]
Xt = sp.load_npz(os.path.join(DATA, "03_tfidf.npz")).tocsr().astype(np.float32)
struct = pd.read_csv(os.path.join(DATA, "03_structural.csv"))

assert Xt.shape[0] == len(y) == len(struct), "row-count mismatch across artifacts"
assert int((struct["label"].to_numpy() != y).sum()) == 0, "structural labels do not match y (ordering problem)"

Xn = normalize(Xt, norm="l2", axis=1, copy=True)   # cosine == dot product
Xtr_T = Xn[tr].T.tocsr()                             # (d, n_train)

def nearest_train_similarity(query_rows, chunk=512):
    Q = Xn[query_rows]
    out = np.empty(len(query_rows), dtype=np.float32)
    for s in range(0, len(query_rows), chunk):
        e = min(s + chunk, len(query_rows))
        S = (Q[s:e] @ Xtr_T).toarray()   # (<=chunk, n_train), bounded memory
        out[s:e] = S.max(axis=1)
        print(f"  ...{e}/{len(query_rows)}", file=sys.stderr, flush=True)
    return out

def dup_summary(sims):
    return {
        "n": int(len(sims)),
        "count_ge_0.99": int((sims >= 0.99).sum()),
        "count_ge_0.95": int((sims >= 0.95).sum()),
        "count_ge_0.90": int((sims >= 0.90).sum()),
        "pct_ge_0.95": round(100 * float((sims >= 0.95).mean()), 2),
        "median_similarity": round(float(np.median(sims)), 4),
        "max_similarity": round(float(sims.max()), 4),
    }

report = {"split_sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))}}

try:
    ct = pd.read_csv(os.path.join(DATA, "02_clean.csv"), usecols=["clean_text"])
    report["exact_duplicate_clean_text"] = int(ct["clean_text"].duplicated().sum()) if len(ct) == len(y) else None
    del ct
except Exception as e:
    report["exact_duplicate_clean_text"] = f"skipped ({e})"

print("near-duplicate: test vs train", file=sys.stderr, flush=True)
report["near_duplicate_test_vs_train"] = dup_summary(nearest_train_similarity(te))
print("near-duplicate: val vs train", file=sys.stderr, flush=True)
report["near_duplicate_val_vs_train"] = dup_summary(nearest_train_similarity(va))

feat_cols = [c for c in struct.columns if c != "label"]
Xs = struct[feat_cols].to_numpy(dtype=np.float64)
ytr = y[tr]
aucs = []
for j, c in enumerate(feat_cols):
    try:
        a = roc_auc_score(ytr, Xs[tr, j])
    except Exception:
        a = 0.5
    aucs.append([c, round(max(a, 1 - a), 4)])
aucs.sort(key=lambda x: -x[1])
report["top_structural_features_by_univariate_auc"] = aucs[:8]
report["shortcut_flag"] = bool(any(a >= 0.98 for _, a in aucs))

with open(os.path.join(DATA, "05_leakage_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
