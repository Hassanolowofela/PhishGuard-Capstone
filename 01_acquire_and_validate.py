"""
PhishGuard Capstone - Phase 2, Stage 1: Data Acquisition & Validation
=====================================================================

Purpose
-------
Acquire a credible, labeled public phishing-email corpus and validate its
integrity before any modeling work begins. Validation guards against silent
data problems (wrong schema, label leakage, severe imbalance, missing text)
that would otherwise corrupt every downstream stage.

Dataset
-------
CEAS_08 component of the curated "Phishing Email Dataset" (Al-Subaiey et al.,
2024), mirrored on GitHub. Each record is a single email with:
    sender, receiver, date, subject, body, label, urls
where label = 1 (phishing) or 0 (legitimate), and urls = 1 if the body
contains at least one URL else 0.

License: CC BY-SA 4.0.  Cite: Al-Subaiey, A., Al-Thani, M., Alam, N. A.,
Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024). Novel interpretable
and robust web-based AI platform for phishing email detection. arXiv:2405.11619.

Outputs
-------
    data/01_validated.csv     validated raw data (schema-checked)
    data/01_quality_report.json   machine-readable data-quality summary
"""
import json
import os
import urllib.request

import numpy as np
import pandas as pd

RAW_URL = ("https://raw.githubusercontent.com/rokibulroni/"
           "Phishing-Email-Dataset/main/CEAS_08.csv")
DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "CEAS_08.csv")
OUT_CSV = os.path.join(DATA_DIR, "01_validated.csv")
OUT_REPORT = os.path.join(DATA_DIR, "01_quality_report.json")

EXPECTED_COLS = ["sender", "receiver", "date", "subject", "body", "label", "urls"]


def acquire():
    """Download the dataset once; reuse the local copy on later runs."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_PATH):
        print(f"Downloading dataset from {RAW_URL} ...")
        urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    size_mb = os.path.getsize(RAW_PATH) / 1e6
    print(f"Local dataset: {RAW_PATH} ({size_mb:.1f} MB)")
    return pd.read_csv(RAW_PATH)


def validate(df: pd.DataFrame) -> dict:
    """Run integrity checks and return a structured quality report."""
    report = {}

    # 1. Schema check
    missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
    assert not missing_cols, f"Missing expected columns: {missing_cols}"
    report["n_rows"], report["n_cols"] = df.shape
    report["columns"] = list(df.columns)

    # 2. Label validity and balance
    labels = sorted(df["label"].dropna().unique().tolist())
    assert set(labels).issubset({0, 1}), f"Unexpected label values: {labels}"
    counts = df["label"].value_counts().to_dict()
    report["label_counts"] = {int(k): int(v) for k, v in counts.items()}
    report["phishing_pct"] = round(100 * df["label"].mean(), 2)

    # 3. Missingness
    report["missing_per_column"] = {c: int(df[c].isna().sum()) for c in df.columns}

    # 4. Duplicates
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["duplicate_bodies"] = int(df.duplicated(subset=["body"]).sum())

    # 5. Empty / trivial bodies (text the model must actually learn from)
    body_len = df["body"].astype(str).str.strip().str.len()
    report["empty_body_rows"] = int((body_len == 0).sum())
    report["body_char_len"] = {
        "min": int(body_len.min()), "median": int(body_len.median()),
        "mean": round(float(body_len.mean()), 1), "max": int(body_len.max()),
    }

    # 6. Leakage sanity check: 'urls' flag should NOT perfectly predict label
    leak = df.groupby("urls")["label"].mean().to_dict()
    report["label_rate_by_url_flag"] = {int(k): round(float(v), 3) for k, v in leak.items()}

    return report


def main():
    df = acquire()
    report = validate(df)

    print("\n================ DATA QUALITY REPORT ================")
    print(f"Rows: {report['n_rows']:,}   Columns: {report['n_cols']}")
    print(f"Phishing: {report['label_counts'].get(1,0):,} "
          f"({report['phishing_pct']}%)   "
          f"Legitimate: {report['label_counts'].get(0,0):,}")
    print(f"Duplicate rows: {report['duplicate_rows']}   "
          f"Duplicate bodies: {report['duplicate_bodies']}")
    print(f"Empty bodies: {report['empty_body_rows']}")
    print(f"Missing subjects: {report['missing_per_column']['subject']}   "
          f"Missing receivers: {report['missing_per_column']['receiver']}")
    print(f"Body length (chars): {report['body_char_len']}")
    print(f"Label rate by URL flag: {report['label_rate_by_url_flag']}")
    print("====================================================\n")

    # Persist validated data and report
    df.to_csv(OUT_CSV, index=False)
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved validated data -> {OUT_CSV}")
    print(f"Saved quality report -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
