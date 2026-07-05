#!/usr/bin/env python3
"""Phase 3, Step 1: build a reproducible, stratified train/validation/test split.

Loads the Phase 2 feature matrix (data/03_combined.npz) and labels
(data/03_labels.npy), creates a fixed 70/15/15 stratified split, saves the
indices (data/04_split_indices.npz) so every later Phase 3 script uses the same
split, and writes a short report (data/04_split_report.json).
"""
import os, json
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEED = 42
TEST_FRAC = 0.15   # held-out test set, untouched until final evaluation
VAL_FRAC = 0.15    # validation set (fraction of the whole)

X = sp.load_npz(os.path.join(DATA, "03_combined.npz"))
y = np.load(os.path.join(DATA, "03_labels.npy"))
assert X.shape[0] == y.shape[0], "feature/label row mismatch"
n = X.shape[0]
idx = np.arange(n)

# 1) hold out the test set first, stratified by label
idx_trainval, idx_test = train_test_split(
    idx, test_size=TEST_FRAC, stratify=y, random_state=SEED)
# 2) split validation out of the remainder so val is VAL_FRAC of the whole
val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=val_rel, stratify=y[idx_trainval], random_state=SEED)

np.savez(os.path.join(DATA, "04_split_indices.npz"),
         train=idx_train, val=idx_val, test=idx_test)

def bal(ii):
    c = np.bincount(y[ii], minlength=2)
    return {"n": int(len(ii)), "legit": int(c[0]), "phish": int(c[1]),
            "phish_pct": round(100 * c[1] / len(ii), 2)}

s_tr, s_va, s_te = set(idx_train.tolist()), set(idx_val.tolist()), set(idx_test.tolist())
report = {
    "seed": SEED,
    "n_total": int(n),
    "n_features": int(X.shape[1]),
    "overall_phish_pct": round(100 * int(y.sum()) / n, 2),
    "splits": {"train": bal(idx_train), "val": bal(idx_val), "test": bal(idx_test)},
    "integrity": {
        "train_val_overlap": len(s_tr & s_va),
        "train_test_overlap": len(s_tr & s_te),
        "val_test_overlap": len(s_va & s_te),
        "covers_all_rows": (len(s_tr | s_va | s_te) == n),
    },
}
with open(os.path.join(DATA, "04_split_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
