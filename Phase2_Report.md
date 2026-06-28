# PhishGuard Capstone — Phase 2 Report: Data Acquisition & Preparation

**Project:** PhishGuard — A Web-Based Machine-Learning Tool for Phishing Email Detection
**Phase:** 2 of 6 (Weeks 1–3 on the project Gantt chart)
**Activities covered:** Acquire & validate datasets · Data cleaning & preprocessing · Feature engineering / extraction

---

## 1. Acquire & Validate Datasets

### 1.1 Dataset selected
The project uses the **CEAS_08** component of the curated *Phishing Email Dataset*
(Al-Subaiey et al., 2024), distributed under a CC BY-SA 4.0 license and mirrored on
GitHub. It was chosen over single-source corpora because it is **labeled, balanced,
and content-rich**: every record contains the sender, receiver, date, subject, body,
a URL-presence flag, and a binary label (1 = phishing, 0 = legitimate). This matches
the project scope, which calls for a public labeled corpus of phishing and legitimate
emails, and it avoids the licensing and access friction of gated sources.

### 1.2 Validation results
A schema-and-integrity check (`01_acquire_and_validate.py`) produced the following
data-quality profile:

| Check | Result |
|-------|--------|
| Rows × columns | 39,154 × 7 |
| Class balance | 21,842 phishing (55.8%) / 17,312 legitimate (44.2%) |
| Exact duplicate rows | 0 |
| Duplicate bodies | 0 |
| Empty bodies | 0 |
| Missing values | subject: 28 · receiver: 462 · body: 0 |
| Body length (chars) | min 11 · median 562 · max 143,996 |
| Leakage check (label rate by URL flag) | 0.54 vs 0.57 |

Two findings shaped later decisions. First, the moderate class imbalance (56/44) is
mild enough to model directly without resampling. Second, and more importantly, the
**URL-presence flag does not trivially predict the label** (53.8% vs 56.8% phishing
rate), so the dataset has no obvious leakage that would let a model "cheat." Missing
subjects and receivers are rare and were handled in cleaning rather than by dropping
rows.

---

## 2. Data Cleaning & Preprocessing

Raw email text is noisy: HTML markup, inconsistent encodings, embedded URLs, and
boilerplate. The cleaning pipeline (`02_clean_and_preprocess.py`) normalizes the text
while **preserving the raw body** so that Stage 3 can still measure original casing,
punctuation, and links. Steps applied to the combined subject + body:

1. Fill missing subject/receiver with empty strings.
2. Strip HTML markup to visible text (BeautifulSoup).
3. Normalize unicode and remove non-printable control characters.
4. Replace URLs and email addresses with the placeholder tokens `urltoken` and
   `emailtoken`, so the model learns *that a link is present* rather than memorizing
   specific domains.
5. Collapse repeated whitespace and lowercase the result into `clean_text`.
6. Drop rows that become empty, then de-duplicate on the cleaned text.

### Cleaning results
| Step | Rows |
|------|------|
| Input | 39,154 |
| Dropped — empty after cleaning | 0 |
| Dropped — duplicates revealed by normalization | 5,031 |
| **Final** | **34,123** |

De-duplication was the most consequential step. Normalization exposed **5,031
near-identical messages** (phishing templates that differed only in URLs or
whitespace). Removing them prevents the model from memorizing repeated templates and
prevents the same message leaking across the train/test split. A useful side effect:
the class balance tightened to a near-even **49.5% phishing**, and the average cleaned
email is ~217 tokens.

---

## 3. Feature Engineering / Extraction

Two complementary feature families were produced (`03_feature_engineering.py`) so the
project can serve both accuracy and its explainability goal.

### 3.1 Structural / handcrafted features (interpretable)
Eighteen human-readable features were computed from the **raw** text, including body
and subject length, URL and HTML counts, exclamation/question/digit counts, uppercase
ratio, currency-symbol presence, an urgency-keyword count, and a reply-thread flag. A
full description ships in `data/03_feature_dictionary.csv`. These directly support the
SHAP/LIME explainability component planned for Phase 3.

The strongest individual signals, by absolute correlation with the phishing label:

| Feature | Correlation |
|---------|-------------|
| `subject_is_reply` | −0.60 |
| `subject_char_len` | −0.32 |
| `subject_word_count` | −0.31 |
| `body_char_len` | −0.29 |
| `body_word_count` | −0.29 |
| `num_question` | −0.20 |

The dominant signal is intuitive: legitimate mail in this corpus is far more often part
of an ongoing reply thread (`Re:`/`Fw:`), whereas phishing messages are typically
unsolicited and shorter. These correlations are dataset-specific and are reported as
exploratory signal, not as final model weights.

### 3.2 TF-IDF text features
The cleaned text was vectorized with a TF-IDF representation over unigrams and bigrams
(English stop-words removed, `min_df=5`, `max_df=0.9`, `max_features=5000`, sublinear
term-frequency scaling), yielding 5,000 lexical features. The fitted vectorizer is
saved so the identical transformation can be applied to new emails at prediction time.

### 3.3 Combined feature matrix
The 18 structural features and 5,000 TF-IDF features were combined into a single sparse
matrix of shape **34,123 × 5,018**, saved with its aligned label vector for Phase 3.

---

## 4. Pipeline Sanity Check

To confirm the prepared features are genuinely predictive (not to tune a final model),
a 3-fold cross-validated logistic-regression baseline was run on the combined matrix:

| Metric | Score |
|--------|-------|
| Accuracy | approximately 0.97 (exact value depends on solver settings) |
| F1 | approximately 0.97 (exact value depends on solver settings) |
| ROC-AUC | approximately 0.99 |

This comfortably clears the project's 95% accuracy target with only a baseline model,
indicating the Phase 2 outputs are sound and that the Phase 3 goals are realistic.

---

## 5. Deliverables

**Pipeline (reproducible):** `01_acquire_and_validate.py`, `02_clean_and_preprocess.py`,
`03_feature_engineering.py`, `requirements.txt`, `README.md`.

**Artifacts for Phase 3:** `03_combined.npz` (model input), `03_labels.npy`,
`03_structural.csv` (interpretable features), `models/tfidf_vectorizer.joblib`,
`03_feature_dictionary.csv`, plus `01_quality_report.json` and a 200-row
`02_clean_sample.csv` for review. (The full intermediate CSVs and matrices are
regenerated by running the scripts and are omitted here for size.)

---

## 6. Next Step (Phase 3 — Model Development)

With a validated, balanced, leakage-checked feature set in hand, Phase 3 will build a
baseline classifier, train the primary model, and tune it toward the ≥95% accuracy and
≥0.95 F1 targets, then attach the SHAP/LIME explainability layer over the structural
features prepared here.

---

### Reference
Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman,
S. A. U. (2024). *Novel interpretable and robust web-based AI platform for phishing
email detection.* arXiv. https://arxiv.org/abs/2405.11619
