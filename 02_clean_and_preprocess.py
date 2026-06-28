"""
PhishGuard Capstone - Phase 2, Stage 2: Cleaning & Preprocessing
================================================================

Purpose
-------
Transform raw, messy email text into a normalized form suitable for feature
extraction, while preserving the raw body for structural feature engineering
(Stage 3 needs the original casing, punctuation, and URLs).

Pipeline (applied to subject + body)
------------------------------------
  1. Fill missing subject/receiver with empty strings.
  2. Strip HTML markup (BeautifulSoup) -> visible text only.
  3. Normalize unicode and remove non-printable/control characters.
  4. Replace URLs and email addresses with placeholder tokens so the model
     learns "a link is present" rather than memorizing specific domains.
  5. Collapse repeated whitespace/newlines.
  6. Lowercase to produce the NLP text field 'clean_text'.
  7. Drop rows that become empty after cleaning, then de-duplicate on
     clean_text (cleaning can collapse near-duplicates into exact ones).

Outputs
-------
  data/02_clean.csv   columns: label, subject, body_raw, clean_text
"""
import os
import re
import unicodedata

import pandas as pd
from bs4 import BeautifulSoup

IN_CSV = "data/01_validated.csv"
OUT_CSV = "data/02_clean.csv"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WS_RE = re.compile(r"\s+")
NONPRINT_RE = re.compile(r"[^\x20-\x7E]")  # keep printable ASCII after norm


def strip_html(text: str) -> str:
    if "<" in text and ">" in text:
        try:
            return BeautifulSoup(text, "lxml").get_text(separator=" ")
        except Exception:
            return text
    return text


def normalize_unicode(text: str) -> str:
    # Convert accented/smart characters to closest ASCII, drop the rest.
    text = unicodedata.normalize("NFKD", text)
    return NONPRINT_RE.sub(" ", text)


def clean_text(raw: str) -> str:
    """Full cleaning pipeline producing the model-ready text field."""
    t = str(raw)
    t = strip_html(t)
    t = normalize_unicode(t)
    t = URL_RE.sub(" urltoken ", t)
    t = EMAIL_RE.sub(" emailtoken ", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)   # keep alphanumerics + spaces
    t = WS_RE.sub(" ", t).strip()
    return t


def main():
    df = pd.read_csv(IN_CSV)
    n0 = len(df)

    df["subject"] = df["subject"].fillna("")
    df["receiver"] = df["receiver"].fillna("")
    df["body"] = df["body"].fillna("")

    # Preserve the raw body+subject for Stage 3 structural features.
    df["body_raw"] = df["body"].astype(str)
    combined = (df["subject"].astype(str) + " . " + df["body"].astype(str))

    print("Cleaning text (HTML strip, tokenize URLs/emails, normalize) ...")
    df["clean_text"] = combined.map(clean_text)

    # Drop rows that are empty after cleaning.
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].copy()
    dropped_empty = before - len(df)

    # De-duplicate on cleaned text (cleaning can reveal exact duplicates).
    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).copy()
    dropped_dups = before - len(df)

    out = df[["label", "subject", "body_raw", "clean_text"]].reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    print("\n================ CLEANING SUMMARY ================")
    print(f"Input rows:            {n0:,}")
    print(f"Dropped (empty text):  {dropped_empty:,}")
    print(f"Dropped (duplicates):  {dropped_dups:,}")
    print(f"Final rows:            {len(out):,}")
    print(f"Class balance now:     {out['label'].value_counts().to_dict()} "
          f"({100*out['label'].mean():.1f}% phishing)")
    avg_tokens = out["clean_text"].str.split().map(len).mean()
    print(f"Avg tokens/email:      {avg_tokens:.1f}")
    print("Example cleaned text:")
    print("  ", out["clean_text"].iloc[0][:160])
    print("==================================================\n")
    print(f"Saved cleaned data -> {OUT_CSV}")


if __name__ == "__main__":
    main()
