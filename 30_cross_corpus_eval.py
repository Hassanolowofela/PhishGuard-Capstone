#!/usr/bin/env python3
"""Phase 5, Test 1: cross-corpus generalization of the binary phishing model.

The headline binary model (09_classifier_scaled) was trained and tested on a single
corpus (CEAS_08), where it reached about 0.9957 test F1. A high in-corpus score can
hide poor generalization, so this script asks the harder question: does that model
still work on email from a completely different source and format that it never saw?

Out-of-distribution corpus: the Zenodo multiclass NLP dataset (the source used for
the Phase 4 three-class model, and never seen by the binary model). Its threat
classes map to phishing (1) and its NOT-Malicious class maps to legitimate (0).

The features are rebuilt exactly as the binary training pipeline did them:
  structural (18) computed from raw subject+body, then TF-IDF (5000) from the same
  cleaning used in 02_clean, combined structural-first into a 5018-wide matrix.

Output: data/30_cross_corpus_report.json
"""
import os, re, json, unicodedata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from bs4 import BeautifulSoup
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, MODELS = os.path.join(HERE, "data"), os.path.join(HERE, "models")

# ---- binary cleaning + structural features (identical to 02_clean / 03_features) ----
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WS_RE = re.compile(r"\s+")
NONPRINT_RE = re.compile(r"[^\x20-\x7E]")
HTML_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"\b\w+\b")
URGENT_WORDS = ["urgent", "verify", "suspend", "account", "password", "click", "login",
                "update", "confirm", "winner", "won", "prize", "free", "limited", "act now",
                "security", "alert", "bank", "ssn", "invoice", "payment", "refund"]
STRUCT_ORDER = ["body_char_len", "body_word_count", "subject_char_len", "subject_word_count",
                "num_urls", "has_url", "num_html_tags", "has_html", "num_exclaim", "num_question",
                "num_digits", "digit_ratio", "uppercase_ratio", "num_money_symbols",
                "has_money_symbol", "avg_word_len", "urgent_word_count", "subject_is_reply"]


def strip_html(text):
    if "<" in text and ">" in text:
        try:
            return BeautifulSoup(text, "lxml").get_text(separator=" ")
        except Exception:
            return text
    return text


def clean_text(raw):
    t = str(raw)
    t = strip_html(t)
    t = unicodedata.normalize("NFKD", t)
    t = NONPRINT_RE.sub(" ", t)
    t = URL_RE.sub(" urltoken ", t)
    t = EMAIL_RE.sub(" emailtoken ", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return WS_RE.sub(" ", t).strip()


def structural(subject, body):
    s, b = str(subject), str(body)
    text = s + " " + b
    low = text.lower()
    w = WORD_RE.findall(text)
    nw = max(len(w), 1)
    L = [c for c in text if c.isalpha()]
    d = {"body_char_len": len(b), "body_word_count": len(WORD_RE.findall(b)),
         "subject_char_len": len(s), "subject_word_count": len(WORD_RE.findall(s)),
         "num_urls": len(URL_RE.findall(text)), "has_url": int(bool(URL_RE.search(text))),
         "num_html_tags": len(HTML_RE.findall(b)), "has_html": int(bool(HTML_RE.search(b))),
         "num_exclaim": text.count("!"), "num_question": text.count("?"),
         "num_digits": sum(c.isdigit() for c in text),
         "digit_ratio": round(sum(c.isdigit() for c in text) / max(len(text), 1), 4),
         "uppercase_ratio": round(sum(c.isupper() for c in L) / max(len(L), 1), 4),
         "num_money_symbols": sum(text.count(c) for c in "$£€"),
         "has_money_symbol": int(any(c in text for c in "$£€")),
         "avg_word_len": round(sum(len(x) for x in w) / nw, 3),
         "urgent_word_count": sum(low.count(u) for u in URGENT_WORDS),
         "subject_is_reply": int(s.strip().lower().startswith(("re:", "fw:", "fwd:")))}
    return [d[k] for k in STRUCT_ORDER]


# ---- parse the Zenodo corpus (label is tab-appended in the Corpus field) ----
KNOWN = {"Phishing", "Malware", "Scareware", "Baiting", "Pretexting",
         "NOT-Malicious General Class", "NOT-Malicious"}
LEGIT = {"NOT-Malicious General Class", "NOT-Malicious"}


def parse(cell):
    s = str(cell)
    if "\t" in s:
        pre, post = s.rsplit("\t", 1)
        post = post.strip()
        if post in KNOWN:
            return pre.strip(), post
    for k in sorted(KNOWN, key=len, reverse=True):
        if s.strip().endswith(k):
            return s.strip()[:-len(k)].strip(), k
    return s.strip(), None


def metrics(y, yp, proba):
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    return {"n": int(len(y)),
            "accuracy": round(accuracy_score(y, yp), 4),
            "precision": round(precision_score(y, yp, zero_division=0), 4),
            "recall": round(recall_score(y, yp, zero_division=0), 4),
            "f1": round(f1_score(y, yp, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, proba), 4),
            "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
            "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) else None,
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def main():
    model = joblib.load(os.path.join(MODELS, "09_classifier_scaled.joblib"))
    vec = joblib.load(os.path.join(MODELS, "tfidf_vectorizer.joblib"))

    df = pd.read_csv(os.path.join(DATA, "phishing_nlp_dataset.csv"))
    pr = df["Corpus"].map(parse)
    df["text"] = [p[0] for p in pr]
    df["rl"] = [p[1] for p in pr]
    df = df[df["rl"].notna()].copy()
    df["y"] = (~df["rl"].isin(LEGIT)).astype(int)

    # de-duplicate on cleaned text, mirroring the training cleaning step
    df["clean"] = (" . " + df["text"].astype(str)).map(clean_text)
    df = df[df["clean"].str.len() > 0].drop_duplicates("clean").reset_index(drop=True)

    y = df["y"].to_numpy()
    Xs = sp.csr_matrix(np.array([structural("", b) for b in df["text"]], dtype=np.float32))
    Xt = vec.transform(df["clean"])
    X = sp.hstack([Xs, Xt]).tocsr()   # structural first, then TF-IDF (matches training)

    proba = model.predict_proba(X)[:, 1]
    yp_default = (proba >= 0.5).astype(int)

    # TF-IDF vocabulary coverage: how much of the OOD text the model can even see
    vocab = set(vec.vocabulary_.keys())
    toks = df["clean"].str.split()
    covered = toks.map(lambda ws: np.mean([w in vocab for w in ws]) if ws else 0.0)

    report = {
        "test": "cross_corpus_generalization",
        "model": "09_classifier_scaled (binary, trained on CEAS_08 only)",
        "ood_corpus": "Zenodo multiclass NLP dataset (unseen source and format)",
        "ood_class_balance": {"legit_0": int((y == 0).sum()), "phish_1": int((y == 1).sum())},
        "in_corpus_reference_test_f1": 0.9957,
        "ood_default_threshold_0.5": metrics(y, yp_default, proba),
        "tfidf_vocab_coverage": {
            "mean_fraction_tokens_in_vocab": round(float(covered.mean()), 4),
            "note": "Fraction of OOD tokens present in the CEAS-trained vocabulary."},
        "interpretation_hint": ("Compare OOD f1 to the in-corpus 0.9957 to read the "
                                "generalization gap. Subject features are zero because "
                                "the OOD corpus has no separate subject line."),
    }
    with open(os.path.join(DATA, "30_cross_corpus_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
