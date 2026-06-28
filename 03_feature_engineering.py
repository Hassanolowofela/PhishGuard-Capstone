"""
PhishGuard Capstone - Phase 2, Stage 3: Feature Engineering / Extraction
=======================================================================

Two complementary feature families are produced:

A) STRUCTURAL / HANDCRAFTED features (interpretable)
   Computed from the RAW body+subject so casing, punctuation and URLs are
   intact. These are human-readable signals (link counts, urgency words,
   uppercase ratio, money symbols, ...) that directly support the project's
   explainability goal (SHAP/LIME) in Phase 3.

B) TF-IDF text features (high-signal, sparse)
   Bag-of-words over the cleaned text with unigrams+bigrams. Captures the
   lexical content that separates phishing from legitimate mail.

Outputs
-------
  data/03_structural.csv        interpretable features + label (dense)
  data/03_tfidf.npz             sparse TF-IDF matrix
  data/03_combined.npz          structural + TF-IDF combined sparse matrix
  data/03_labels.npy            label vector aligned to the matrices
  models/tfidf_vectorizer.joblib  fitted vectorizer (reuse in Phase 3)
  data/03_feature_dictionary.csv  description of every structural feature
"""
import os
import re

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

IN_CSV = "data/02_clean.csv"
os.makedirs("models", exist_ok=True)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"\b\w+\b")
FREEMAIL = ("gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com")

# Words commonly used to manufacture urgency / lure clicks in phishing.
URGENT_WORDS = [
    "urgent", "verify", "suspend", "account", "password", "click", "login",
    "update", "confirm", "winner", "won", "prize", "free", "limited", "act now",
    "security", "alert", "bank", "ssn", "invoice", "payment", "refund",
]


def structural_features(subject: str, body: str) -> dict:
    s, b = str(subject), str(body)
    text = s + " " + b
    low = text.lower()
    words = WORD_RE.findall(text)
    n_words = max(len(words), 1)
    letters = [c for c in text if c.isalpha()]

    return {
        "body_char_len": len(b),
        "body_word_count": len(WORD_RE.findall(b)),
        "subject_char_len": len(s),
        "subject_word_count": len(WORD_RE.findall(s)),
        "num_urls": len(URL_RE.findall(text)),
        "has_url": int(bool(URL_RE.search(text))),
        "num_html_tags": len(HTML_RE.findall(b)),
        "has_html": int(bool(HTML_RE.search(b))),
        "num_exclaim": text.count("!"),
        "num_question": text.count("?"),
        "num_digits": sum(c.isdigit() for c in text),
        "digit_ratio": round(sum(c.isdigit() for c in text) / max(len(text), 1), 4),
        "uppercase_ratio": round(
            sum(c.isupper() for c in letters) / max(len(letters), 1), 4),
        "num_money_symbols": sum(text.count(c) for c in "$\u00a3\u20ac"),
        "has_money_symbol": int(any(c in text for c in "$\u00a3\u20ac")),
        "avg_word_len": round(sum(len(w) for w in words) / n_words, 3),
        "urgent_word_count": sum(low.count(w) for w in URGENT_WORDS),
        "subject_is_reply": int(s.strip().lower().startswith(("re:", "fw:", "fwd:"))),
    }


FEATURE_DESCRIPTIONS = {
    "body_char_len": "Length of the email body in characters",
    "body_word_count": "Number of words in the body",
    "subject_char_len": "Length of the subject line in characters",
    "subject_word_count": "Number of words in the subject",
    "num_urls": "Count of URLs in subject+body",
    "has_url": "1 if at least one URL is present",
    "num_html_tags": "Count of HTML tags in the raw body",
    "has_html": "1 if the body contains HTML markup",
    "num_exclaim": "Count of '!' characters (urgency/excitement signal)",
    "num_question": "Count of '?' characters",
    "num_digits": "Count of digit characters",
    "digit_ratio": "Digits as a fraction of all characters",
    "uppercase_ratio": "Uppercase letters as a fraction of all letters (shouting)",
    "num_money_symbols": "Count of $, GBP, EUR symbols (financial lure)",
    "has_money_symbol": "1 if any currency symbol is present",
    "avg_word_len": "Average word length",
    "urgent_word_count": "Count of known urgency/lure keywords",
    "subject_is_reply": "1 if subject starts with Re:/Fw:/Fwd: (spoofed thread)",
}


def main():
    df = pd.read_csv(IN_CSV)
    df["subject"] = df["subject"].fillna("")
    df["body_raw"] = df["body_raw"].fillna("")
    df["clean_text"] = df["clean_text"].fillna("")

    # ---- A) Structural features ----
    print("Engineering structural features ...")
    feats = pd.DataFrame(
        [structural_features(s, b)
         for s, b in zip(df["subject"], df["body_raw"])]
    )
    struct = pd.concat([df[["label"]].reset_index(drop=True), feats], axis=1)
    struct.to_csv("data/03_structural.csv", index=False)

    # Correlation of each structural feature with the phishing label.
    corr = feats.corrwith(df["label"]).sort_values(key=abs, ascending=False)

    # ---- B) TF-IDF text features ----
    print("Fitting TF-IDF vectorizer (unigrams + bigrams) ...")
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2),
        min_df=5, max_df=0.9, max_features=5000, sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(df["clean_text"])
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    sparse.save_npz("data/03_tfidf.npz", X_tfidf)

    # ---- Combine structural + TF-IDF into one matrix for modeling ----
    X_struct = sparse.csr_matrix(feats.values.astype(np.float32))
    X_combined = sparse.hstack([X_struct, X_tfidf]).tocsr()
    sparse.save_npz("data/03_combined.npz", X_combined)
    np.save("data/03_labels.npy", df["label"].values)

    # ---- Feature dictionary ----
    pd.DataFrame(
        [{"feature": k, "description": v} for k, v in FEATURE_DESCRIPTIONS.items()]
    ).to_csv("data/03_feature_dictionary.csv", index=False)

    print("\n================ FEATURE SUMMARY ================")
    print(f"Rows:                    {X_combined.shape[0]:,}")
    print(f"Structural features:     {X_struct.shape[1]}")
    print(f"TF-IDF features:         {X_tfidf.shape[1]:,}")
    print(f"Combined feature width:  {X_combined.shape[1]:,}")
    print("\nTop structural signals (|correlation| with phishing label):")
    for name, val in corr.head(8).items():
        print(f"  {name:<20} {val:+.3f}")
    print("=================================================\n")
    print("Saved: data/03_structural.csv, data/03_tfidf.npz, "
          "data/03_combined.npz,\n       data/03_labels.npy, "
          "data/03_feature_dictionary.csv, models/tfidf_vectorizer.joblib")


if __name__ == "__main__":
    main()
