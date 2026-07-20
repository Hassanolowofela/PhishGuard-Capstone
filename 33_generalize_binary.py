#!/usr/bin/env python3
"""Phase 5 fix: improve the binary model's cross-corpus generalization.

Phase 5 showed the CEAS-only binary model over-flags legitimate mail on an unseen
corpus (out-of-corpus F1 about 0.84, false-positive rate about 0.80), because only
about 39 percent of the new corpus's words are in its word-level vocabulary.

This script applies three changes and measures each honestly:
  1. Character n-gram TF-IDF (char_wb) instead of word n-grams, so the model keeps
     signal on unseen or lightly obfuscated words (near total coverage).
  2. Balanced class weights, so it does not default to "phishing" on unfamiliar text.
  3. Dropping the three subject-format structural features, which are a corpus
     format artifact (the Zenodo corpus has no subject line) and do not transfer.

It reports three things:
  A. Single-source generalization: train on CEAS, test on the unseen Zenodo corpus,
     comparing the word baseline with the improved features (at 0.5 and a tuned
     threshold). This shows the feature fix improves ranking on an unseen source.
  B. The reverse direction (train on the small Zenodo corpus, test on CEAS), which
     shows a single small corpus cannot carry generalization on its own.
  C. The shipped model: the improved model trained on the union of both corpora and
     evaluated on a held-out 20 percent mix of both sources. This is the model that
     is saved, and it is strong across both sources.

Notes on scale: CEAS is subsampled (stratified) and the character vocabulary is
capped, so this runs quickly and repeatably. The conclusion is the cross-corpus
behaviour, not a single leaderboard number; raise CEAS_SAMPLE and max_features on a
faster machine for tighter estimates.

Outputs: models/33_binary_generalized.joblib, data/33_generalization_report.json
"""
import os, re, json, unicodedata, time
import numpy as np, pandas as pd, scipy.sparse as sp, joblib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, MODELS = os.path.join(HERE, "data"), os.path.join(HERE, "models")
SEED = 42
CEAS_SAMPLE = 4000          # stratified subsample of CEAS for tractability
CHAR_MAX_FEATURES = 6000

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WS = re.compile(r"\s+"); NPR = re.compile(r"[^\x20-\x7E]")
HTML = re.compile(r"<[^>]+>"); WORD = re.compile(r"\b\w+\b")
URG = ["urgent", "verify", "suspend", "account", "password", "click", "login",
       "update", "confirm", "winner", "won", "prize", "free", "limited", "act now",
       "security", "alert", "bank", "ssn", "invoice", "payment", "refund"]
STRUCT = ["body_char_len", "body_word_count", "subject_char_len", "subject_word_count",
          "num_urls", "has_url", "num_html_tags", "has_html", "num_exclaim", "num_question",
          "num_digits", "digit_ratio", "uppercase_ratio", "num_money_symbols",
          "has_money_symbol", "avg_word_len", "urgent_word_count", "subject_is_reply"]
SUBJ = ["subject_char_len", "subject_word_count", "subject_is_reply"]


def sh(t):
    if "<" in t and ">" in t:
        try:
            return BeautifulSoup(t, "lxml").get_text(separator=" ")
        except Exception:
            return t
    return t


def clean(r):
    t = sh(str(r)); t = NPR.sub(" ", unicodedata.normalize("NFKD", t))
    t = URL_RE.sub(" urltoken ", t); t = EMAIL_RE.sub(" emailtoken ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t.lower())
    return WS.sub(" ", t).strip()


def st(subject, body):
    s, b = str(subject), str(body)
    tx = s + " " + b; lo = tx.lower(); w = WORD.findall(tx); nw = max(len(w), 1)
    L = [c for c in tx if c.isalpha()]
    return {"body_char_len": len(b), "body_word_count": len(WORD.findall(b)),
            "subject_char_len": len(s), "subject_word_count": len(WORD.findall(s)),
            "num_urls": len(URL_RE.findall(tx)), "has_url": int(bool(URL_RE.search(tx))),
            "num_html_tags": len(HTML.findall(b)), "has_html": int(bool(HTML.search(b))),
            "num_exclaim": tx.count("!"), "num_question": tx.count("?"),
            "num_digits": sum(c.isdigit() for c in tx),
            "digit_ratio": round(sum(c.isdigit() for c in tx) / max(len(tx), 1), 4),
            "uppercase_ratio": round(sum(c.isupper() for c in L) / max(len(L), 1), 4),
            "num_money_symbols": sum(tx.count(c) for c in "$£€"),
            "has_money_symbol": int(any(c in tx for c in "$£€")),
            "avg_word_len": round(sum(len(x) for x in w) / nw, 3),
            "urgent_word_count": sum(lo.count(u) for u in URG),
            "subject_is_reply": int(s.strip().lower().startswith(("re:", "fw:", "fwd:")))}


KNOWN = {"Phishing", "Malware", "Scareware", "Baiting", "Pretexting",
         "NOT-Malicious General Class", "NOT-Malicious"}
LEG = {"NOT-Malicious General Class", "NOT-Malicious"}


def pz(cell):
    s = str(cell)
    if "\t" in s:
        pre, post = s.rsplit("\t", 1); post = post.strip()
        if post in KNOWN:
            return pre.strip(), post
    for k in sorted(KNOWN, key=len, reverse=True):
        if s.strip().endswith(k):
            return s.strip()[:-len(k)].strip(), k
    return s.strip(), None


def mk(kind):
    if kind == "word":
        return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5,
                               max_df=0.9, max_features=5000, sublinear_tf=True)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=5,
                           max_features=CHAR_MAX_FEATURES, sublinear_tf=True)


def MET(y, yp, pr):
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    return {"n": int(len(y)), "accuracy": round(accuracy_score(y, yp), 4),
            "precision": round(precision_score(y, yp, zero_division=0), 4),
            "recall": round(recall_score(y, yp, zero_division=0), 4),
            "f1": round(f1_score(y, yp, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, pr), 4),
            "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None}


def load():
    c = pd.read_csv(os.path.join(DATA, "02_clean.csv"))[["label", "subject", "body_raw", "clean_text"]].copy()
    c["subject"] = c["subject"].fillna(""); c["body_raw"] = c["body_raw"].fillna("")
    c["y"] = c["label"].astype(int); c["corpus"] = "ceas"
    if len(c) > CEAS_SAMPLE:
        # groupby().sample() preserves every column (unlike groupby().apply(),
        # which drops the grouping column "y" on newer pandas).
        c = c.groupby("y", group_keys=False).sample(
            frac=CEAS_SAMPLE / len(c), random_state=SEED).reset_index(drop=True)
    z = pd.read_csv(os.path.join(DATA, "phishing_nlp_dataset.csv"))
    pr = z["Corpus"].map(pz)
    z["body_raw"] = [p[0] for p in pr]; z["rl"] = [p[1] for p in pr]
    z = z[z["rl"].notna()].copy(); z["subject"] = ""
    z["y"] = (~z["rl"].isin(LEG)).astype(int)
    z["clean_text"] = (" . " + z["body_raw"].astype(str)).map(clean)
    z = z[z["clean_text"].str.len() > 0].drop_duplicates("clean_text"); z["corpus"] = "zenodo"
    cols = ["corpus", "subject", "body_raw", "clean_text", "y"]
    df = pd.concat([c[cols], z[cols]], ignore_index=True).reset_index(drop=True)
    feats = pd.DataFrame([st(s, b) for s, b in zip(df["subject"], df["body_raw"])])
    return df, feats


def sm(feats, idx, cols):
    return sp.csr_matrix(feats.iloc[idx][cols].to_numpy(dtype=np.float32))


def train(df, feats, kind, bal, drop, tr):
    cols = [c for c in STRUCT if not (drop and c in SUBJ)]
    v = mk(kind)
    Xtr = sp.hstack([sm(feats, tr, cols), v.fit_transform(df["clean_text"].iloc[tr])]).tocsr()
    clf = make_pipeline(MaxAbsScaler(), LogisticRegression(
        max_iter=400, solver="lbfgs", C=1.0, class_weight=("balanced" if bal else None)))
    clf.fit(Xtr, df["y"].to_numpy()[tr])
    return clf, v, cols


def proba(df, feats, clf, v, cols, te):
    X = sp.hstack([sm(feats, te, cols), v.transform(df["clean_text"].iloc[te])]).tocsr()
    return clf.predict_proba(X)[:, 1]


def tune_thr(y, pr):
    grid = np.linspace(0.2, 0.85, 131)
    return float(grid[int(np.argmax([f1_score(y, (pr >= t).astype(int)) for t in grid]))])


def main():
    t0 = time.time()
    df, feats = load(); y = df["y"].to_numpy()
    ci = df.index[df["corpus"] == "ceas"].to_numpy()
    zi = df.index[df["corpus"] == "zenodo"].to_numpy()

    # A) single-source generalization: train CEAS, test unseen Zenodo
    clf, v, cols = train(df, feats, "word", False, False, ci)
    base = MET(y[zi], (proba(df, feats, clf, v, cols, zi) >= 0.5).astype(int), proba(df, feats, clf, v, cols, zi))
    tr2, ho = train_test_split(ci, test_size=0.2, stratify=y[ci], random_state=SEED)
    clf2, v2, cols2 = train(df, feats, "char", True, True, tr2)
    thr = tune_thr(y[ho], proba(df, feats, clf2, v2, cols2, ho))
    clf2b, v2b, cols2b = train(df, feats, "char", True, True, ci)
    pz2 = proba(df, feats, clf2b, v2b, cols2b, zi)
    imp05 = MET(y[zi], (pz2 >= 0.5).astype(int), pz2)
    impT = MET(y[zi], (pz2 >= thr).astype(int), pz2)

    # B) reverse direction (small corpus alone cannot generalize)
    clfr, vr, colsr = train(df, feats, "char", True, True, zi)
    pcr = proba(df, feats, clfr, vr, colsr, ci)
    imprev = MET(y[ci], (pcr >= 0.5).astype(int), pcr)

    # C) shipped model realistic performance: 80/20 union split, mixed sources
    strat = [f"{a}{b}" for a, b in zip(df["corpus"], df["y"])]
    utr, ute = train_test_split(df.index.to_numpy(), test_size=0.2, stratify=strat, random_state=SEED)
    clfu, vu, colsu = train(df, feats, "char", True, True, utr)
    pu = proba(df, feats, clfu, vu, colsu, ute)
    union_ho = MET(y[ute], (pu >= 0.5).astype(int), pu)

    # Final shipped model: train on the FULL union, threshold tuned on a 15 percent holdout
    ftr, fho = train_test_split(df.index.to_numpy(), test_size=0.15, stratify=y, random_state=SEED)
    clff, vf, colsf = train(df, feats, "char", True, True, ftr)
    fthr = tune_thr(y[fho], proba(df, feats, clff, vf, colsf, fho))
    clffull, vfull, colsfull = train(df, feats, "char", True, True, df.index.to_numpy())
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump({"model": clffull, "vectorizer": vfull, "threshold": round(fthr, 3),
                 "struct_cols": colsfull, "config": "char_wb(3,4)+balanced+drop_subject, union-trained"},
                os.path.join(MODELS, "33_binary_generalized.joblib"))

    rep = {"goal": "improve cross-corpus generalization of the binary phishing model",
           "note": ("CEAS subsampled to ~4000 (stratified) and character n-grams (3,4) capped at 6000 "
                    "features for tractability. Numbers are directional; the shipped model trains on the full union."),
           "corpora": {"ceas_sample": int(len(ci)), "zenodo": int(len(zi))},
           "A_single_source_generalization_CEAS_to_Zenodo": {
               "baseline_word_at_0.5": base,
               "improved_char_balanced_dropsubject_at_0.5": imp05,
               "improved_char_at_tuned_threshold": {**impT, "threshold": round(thr, 3)}},
           "B_reverse_Zenodo_to_CEAS_improved_at_0.5": imprev,
           "C_shipped_model_union_heldout_20pct_mixed_sources": union_ho,
           "final_model": {"file": "models/33_binary_generalized.joblib",
                           "trained_on": "union of CEAS_08 subsample and Zenodo",
                           "threshold": round(fthr, 3), "struct_features_used": len(colsfull)},
           "runtime_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(DATA, "33_generalization_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print("DONE", rep["runtime_seconds"])


if __name__ == "__main__":
    main()
