"""PhishGuard serving: feature construction.

This module reproduces the exact preprocessing (pipeline stage 02) and the 18
interpretable structural features (stage 03) used during training, so an email
submitted to the web app is turned into precisely the same 5,018-dimension
feature vector the model was trained on (18 structural, then 5,000 TF-IDF).

Keeping this logic identical to training is essential: any drift between the
training features and the serving features would silently degrade accuracy.
"""
import re
import unicodedata

import numpy as np
import scipy.sparse as sp

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except Exception:  # defensive fallback if bs4 is unavailable
    _HAS_BS4 = False

# ---- regexes shared with stages 02 and 03 ----
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WS_RE = re.compile(r"\s+")
NONPRINT_RE = re.compile(r"[^\x20-\x7E]")
HTML_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"\b\w+\b")

URGENT_WORDS = [
    "urgent", "verify", "suspend", "account", "password", "click", "login",
    "update", "confirm", "winner", "won", "prize", "free", "limited", "act now",
    "security", "alert", "bank", "ssn", "invoice", "payment", "refund",
]

# 18 structural features, in the exact order used to build the training matrix.
STRUCT_ORDER = [
    "body_char_len", "body_word_count", "subject_char_len", "subject_word_count",
    "num_urls", "has_url", "num_html_tags", "has_html", "num_exclaim",
    "num_question", "num_digits", "digit_ratio", "uppercase_ratio",
    "num_money_symbols", "has_money_symbol", "avg_word_len", "urgent_word_count",
    "subject_is_reply",
]


def strip_html(text: str) -> str:
    if "<" in text and ">" in text:
        if _HAS_BS4:
            try:
                return BeautifulSoup(text, "lxml").get_text(separator=" ")
            except Exception:
                return text
        return HTML_RE.sub(" ", text)
    return text


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return NONPRINT_RE.sub(" ", text)


def clean_text(raw: str) -> str:
    """Stage 02 cleaning: produces the model-ready NLP text for TF-IDF."""
    t = str(raw)
    t = strip_html(t)
    t = normalize_unicode(t)
    t = URL_RE.sub(" urltoken ", t)
    t = EMAIL_RE.sub(" emailtoken ", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def structural_features(subject: str, body: str) -> dict:
    """Stage 03 structural features, computed from the RAW subject and body."""
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
        "num_money_symbols": sum(text.count(c) for c in "$£€"),
        "has_money_symbol": int(any(c in text for c in "$£€")),
        "avg_word_len": round(sum(len(w) for w in words) / n_words, 3),
        "urgent_word_count": sum(low.count(w) for w in URGENT_WORDS),
        "subject_is_reply": int(s.strip().lower().startswith(("re:", "fw:", "fwd:"))),
    }


def build_combined(subject: str, body: str, vectorizer):
    """Return (X, clean, feats): the 1x5018 sparse feature row for one email.

    The TF-IDF half is built from ``subject + " . " + body`` after cleaning,
    matching how the training text field was assembled in stage 02.
    """
    combined_text = str(subject) + " . " + str(body)
    clean = clean_text(combined_text)
    Xt = vectorizer.transform([clean])                       # 1 x 5000
    feats = structural_features(subject, body)
    Xs = sp.csr_matrix(
        np.array([[feats[k] for k in STRUCT_ORDER]], dtype=np.float32))  # 1 x 18
    X = sp.hstack([Xs, Xt]).tocsr()                          # 1 x 5018
    return X, clean, feats
