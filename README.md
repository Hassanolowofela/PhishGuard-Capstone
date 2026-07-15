# PhishGuard - Phase 2: Data Acquisition & Preparation

[![CI](https://github.com/Hassanolowofela/PhishGuard-Capstone/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/Hassanolowofela/PhishGuard-Capstone/actions/workflows/ci.yml)

A reproducible data pipeline for **PhishGuard**, a web-based machine-learning tool
for phishing email detection, built as part of an MSIT capstone project. This
repository covers **Phase 2** of the project: acquiring a public labeled email
dataset, validating it, cleaning and preprocessing the text, and engineering the
features used for model training in Phase 3.

> A full screenshot-by-screenshot account of running this pipeline is in
> **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)**. The Phase 1 project proposal -
> problem statement, scope, SMART goals, ethics & security review, and project plan -
> is in **[docs/PROPOSAL.md](docs/PROPOSAL.md)**. The system design, architecture, and
> ethics/security-by-design specification are in **[docs/SDD.md](docs/SDD.md)**. The
> Phase 3 model development and honest evaluation writeup is in
> **[docs/PHASE3_REPORT.md](docs/PHASE3_REPORT.md)**, with a hands-on run-it-yourself
> guide in **[docs/PHASE3_WALKTHROUGH.md](docs/PHASE3_WALKTHROUGH.md)**. The Phase 4
> web application and its Safe/Scam/Malware classifier are described in
> **[docs/PHASE4_REPORT.md](docs/PHASE4_REPORT.md)** (the app itself is in **[webapp/](webapp/)**).

---

## Overview

PhishGuard is an MSIT capstone building a web-based, machine-learning tool that lets
non-technical users check whether an email is phishing. This repository is **Phase 2**:
the data pipeline that turns a raw public email corpus into model-ready features. It
runs as three sequential stages, each reading the previous stage's output:

1. **Acquire & validate** (`01_acquire_and_validate.py`) - downloads and caches the
   CEAS_08 dataset (39,154 labeled emails), then profiles it for schema correctness,
   class balance, duplicates, missing values, and label leakage, writing a
   data-quality report before anything downstream runs.
2. **Clean & preprocess** (`02_clean_and_preprocess.py`) - strips HTML to visible text,
   replaces URLs and email addresses with placeholder tokens, lowercases and normalizes
   the text, collapses whitespace, and drops duplicate messages - leaving ~34,123
   deduplicated emails (~49.5% phishing).
3. **Feature engineering** (`03_feature_engineering.py`) - derives 18 interpretable
   structural features from the original text (body/subject length, URL and HTML
   counts, punctuation, digit/uppercase ratios, currency-symbol and urgency-keyword
   signals, reply-prefix spoofing) plus a 5,000-term TF-IDF representation - 5,018
   features in total - and saves the fitted vectorizer for reuse in Phase 3.

Together these stages produce the labeled feature matrix and supporting artifacts
consumed in Phase 3. An early in-pipeline baseline on the combined features reached
roughly 0.97, but that number was optimistic: it predated leakage control and a
feature-scaling fix. Phase 3 rebuilds the evaluation rigorously (near-duplicate
leakage audit, group-aware re-split, and a truly held-out test set) and, after
correcting a feature-scaling flaw the early baseline had hidden, the final model
reaches **0.996 F1** on the clean held-out test set. The full story is in
**[docs/PHASE3_REPORT.md](docs/PHASE3_REPORT.md)**.

---

## Results at a glance

| Item | Result |
|------|--------|
| Source dataset | CEAS_08, 39,154 labeled emails (Al-Subaiey et al., 2024) |
| After cleaning & de-duplication | 34,123 emails, ~49.5% phishing |
| Features engineered | 18 interpretable + 5,000 TF-IDF = 5,018 total |
| Phase 3 final model (held-out test) | 0.9957 accuracy / 0.9957 F1 / 0.9997 ROC-AUC |

---

## Phase 3: model development (honest evaluation)

Phase 3 trains and evaluates the classifier with an emphasis on results that hold up
on unseen data. The short version of the arc:

| Stage | Test F1 | What it represents |
|-------|--------:|--------------------|
| Early in-pipeline baseline | ~0.97 | Optimistic: no leakage control, unscaled features |
| Leakage-controlled combined | 0.9484 | Honest, but exposed a feature-scaling flaw |
| Scaled combined (final) | **0.9957** | Honest and correct, exceeds the original number |

Highlights: a near-duplicate leakage audit found 18.7% of test emails had a training
near-twin, so the data was de-duplicated and re-split by group; a feature-scaling flaw
that made the combined model underperform TF-IDF alone was found and fixed with a
`MaxAbsScaler`; SHAP explanations show the model reasons mainly from email content;
and a dominant structural cue (`subject_is_reply`) was verified to be a corpus artifact
the model does not depend on. Full details, tables, and per-step scripts are in
**[docs/PHASE3_REPORT.md](docs/PHASE3_REPORT.md)**.

---

## Repository structure

```
.
├── 01_acquire_and_validate.py     # Stage 1: download + validate the dataset
├── 02_clean_and_preprocess.py     # Stage 2: clean and normalize the text
├── 03_feature_engineering.py      # Stage 3: build structural + TF-IDF features
├── requirements.txt               # Python dependencies
├── README.md                      # this file
├── Phase2_Report.md               # written report of methods and findings
├── docs/
│   ├── PROPOSAL.md                # Phase 1 capstone proposal
│   ├── SDD.md                     # software design doc (architecture + ethics/security)
│   ├── PHASE3_REPORT.md           # Phase 3 model development + honest evaluation
│   ├── PHASE3_WALKTHROUGH.md      # Phase 3 step-by-step run guide
│   └── WALKTHROUGH.md             # step-by-step run log with screenshots
├── screenshots/                   # images used in the walkthrough
├── data/                          # small reviewable artifacts (see .gitignore)
│   ├── 01_quality_report.json
│   ├── 03_structural.csv
│   ├── 03_feature_dictionary.csv
│   └── 02_clean_sample.csv
└── models/
    └── tfidf_vectorizer.joblib    # fitted vectorizer for reuse in Phase 3
```

Large intermediate files (the 65 MB raw dataset and the full feature matrices) are
**not** committed; they are regenerated by running the scripts. See `.gitignore`.

---

## Quick start

```bash
pip install -r requirements.txt
python 01_acquire_and_validate.py
python 02_clean_and_preprocess.py
python 03_feature_engineering.py
```

Stage 1 downloads the dataset (~65 MB) on first run and caches it locally.

---

## Continuous integration and delivery

Every push and pull request runs a GitHub Actions pipeline
(`.github/workflows/ci.yml`) that installs the dependencies, lints the code with
ruff, byte-compiles every Python file so a syntax error cannot reach `main`, runs
the unit tests, and checks the documentation for stray dashes. The unit tests in
`tests/` cover the parts that do not need model artifacts: the text cleaning and
the 18 structural features in `webapp/features.py`, the combined feature vector
width, and the sender heuristics in `webapp/inference.py`.

For delivery, a `Dockerfile` builds a consistent image and serves the app with
gunicorn, so PhishGuard runs the same way on any machine or free tier host. This
keeps the project releasable at all times: small, frequently merged changes are
automatically built and tested before they reach `main`.

---

## Dataset & license

CEAS_08 component of the curated *Phishing Email Dataset* (CC BY-SA 4.0), mirrored on
GitHub. Please cite:

> Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman,
> S. A. U. (2024). *Novel interpretable and robust web-based AI platform for phishing
> email detection.* arXiv. https://arxiv.org/abs/2405.11619

---

## Project phases

- [x] Phase 1 - Initiation & planning ([proposal](docs/PROPOSAL.md), repo)
- [x] **Phase 2 - Data acquisition & preparation** (this repository)
- [x] Phase 3 - Model development ([report](docs/PHASE3_REPORT.md))
- [x] Phase 4 - Web application ([report](docs/PHASE4_REPORT.md), [app](webapp/))
- [ ] Phase 5 - Testing & evaluation
- [ ] Phase 6 - Documentation & delivery
