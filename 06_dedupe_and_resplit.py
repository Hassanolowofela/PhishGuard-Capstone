#!/usr/bin/env python3
"""Phase 3, Step 2b: remove near-duplicate leakage, then re-split.

The Step 2 audit showed near-identical emails straddling the train/test boundary.
This fixes that so the reported metrics are trustworthy:

1. Build a near-duplicate graph on the TF-IDF vectors. Two emails are linked when
   their cosine similarity is >= THRESHOLD (0.95). Each email keeps at most its
   top-K nearest neighbours, which stays fast and memory-safe even when a phishing
   template repeats many times and is still enough to connect a cluster.
2. Find the connected components (clusters of near-duplicate emails).
3. Make a GROUP-AWARE 70/15/15 split so every cluster stays entirely on one side.
4. Save the clean split (data/06_split_indices.npz) and verify the fix.

Later Phase 3 scripts load 06_split_indices.npz instead of the original 04 split.
"""
import os, sys, json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
THRESHOLD, K, SEED = 0.95, 25, 42

y = np.load(os.path.join(DATA, "03_labels.npy"))
Xt = sp.load_npz(os.path.join(DATA, "03_tfidf.npz")).tocsr().astype(np.float32)
n = Xt.shape[0]
Xn = normalize(Xt, norm="l2", axis=1, copy=True)
XT = Xn.T.tocsr()

ii_all, jj_all = [], []
for s in range(0, n, 512):
    e = min(s + 512, n)
    S = (Xn[s:e] @ XT).toarray()
    for local in range(e - s):
        i = s + local
        row = S[local]
        cand = np.nonzero(row >= THRESHOLD)[0]
        cand = cand[cand != i]
        if cand.size == 0:
            continue
        if cand.size > K:
            cand = cand[np.argpartition(row[cand], -K)[-K:]]
        ii_all.append(np.full(cand.size, i, dtype=np.int32))
        jj_all.append(cand.astype(np.int32))
    print(f"  edges ...{e}/{n}", file=sys.stderr, flush=True)

ii = np.concatenate(ii_all) if ii_all else np.array([], np.int32)
jj = np.concatenate(jj_all) if jj_all else np.array([], np.int32)
G = sp.csr_matrix((np.ones(ii.size, np.int8), (ii, jj)), shape=(n, n))
G = G + G.T
ncomp, groups = connected_components(G, directed=False)
_, counts = np.unique(groups, return_counts=True)

idx = np.arange(n)
# StratifiedGroupKFold keeps each cluster whole AND balances the labels.
# 7 folds -> ~14.3% each: fold 0 = test, fold 1 = val, folds 2-6 = train (~71%).
folds = [te for _, te in StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED).split(idx, y, groups)]
test, val = folds[0], folds[1]
used = np.zeros(n, bool); used[test] = True; used[val] = True
train = idx[~used]
np.savez(os.path.join(DATA, "06_split_indices.npz"), train=train, val=val, test=test)

def bal(a):
    c = np.bincount(y[a], minlength=2)
    return {"n": int(len(a)), "legit": int(c[0]), "phish": int(c[1]), "phish_pct": round(100*c[1]/len(a), 2)}

XtrT = Xn[train].T.tocsr()
def max_sim(q):
    out = np.empty(len(q), np.float32)
    for s in range(0, len(q), 512):
        e = min(s + 512, len(q))
        out[s:e] = (Xn[q[s:e]] @ XtrT).toarray().max(axis=1)
    return out
te_sim, va_sim = max_sim(test), max_sim(val)
gtr = set(groups[train].tolist())

report = {
    "threshold": THRESHOLD, "top_k": K, "seed": SEED, "n_total": int(n),
    "near_duplicate_clusters_total": int(ncomp),
    "multi_member_clusters": int((counts > 1).sum()),
    "emails_in_multi_member_clusters": int(counts[counts > 1].sum()),
    "largest_cluster_size": int(counts.max()),
    "clean_split": {"train": bal(train), "val": bal(val), "test": bal(test)},
    "group_leakage_after_split": {
        "train_test_shared_groups": len(gtr & set(groups[test].tolist())),
        "train_val_shared_groups": len(gtr & set(groups[val].tolist())),
    },
    "verify_test_vs_train": {"count_ge_0.95": int((te_sim >= 0.95).sum()),
                             "count_ge_0.90": int((te_sim >= 0.90).sum()),
                             "max_similarity": round(float(te_sim.max()), 4)},
    "verify_val_vs_train": {"count_ge_0.95": int((va_sim >= 0.95).sum()),
                            "count_ge_0.90": int((va_sim >= 0.90).sum()),
                            "max_similarity": round(float(va_sim.max()), 4)},
}
with open(os.path.join(DATA, "06_dedupe_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
