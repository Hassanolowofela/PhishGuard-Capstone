# PhishGuard Phase 5 Fix: Cross Corpus Generalization

**Project:** PhishGuard, an interpretable web based tool for phishing and malware email detection
**Course:** MSIT 5910 Capstone Project
**Author:** Hassan Olowofela

## The problem

Phase 5 found that the headline binary model, which scores about 0.9957 F1 on its own
corpus (CEAS_08), generalises poorly to an unseen source. On the Zenodo corpus it
kept high recall but over flagged legitimate mail, with a false positive rate near
0.80 and a ROC AUC of only about 0.67. The diagnosis was concrete: the model uses a
word level vocabulary, and only about 39 percent of the new corpus's words were in
that vocabulary, so on unfamiliar text it defaulted to a phishing verdict.

## The fix

Three changes target that diagnosis directly, all in `33_generalize_binary.py`:

First, character n-gram features (char_wb, 3 to 4) replace word n-grams. Character
n-grams cover essentially any text, including unseen words and light obfuscation, so
the model keeps real signal on a new corpus instead of falling back to a few learned
danger words. Second, balanced class weights stop the model defaulting to phishing on
unfamiliar input. Third, the three subject-format structural features are dropped,
because they are a corpus format artifact (the Zenodo corpus has no subject line) and
do not transfer. The decision threshold is then tuned on a held-out slice of the
training data rather than left at 0.5.

Generalization is measured by training on one corpus and testing on the other, so the
test corpus is genuinely unseen, and separately by training the improved model on the
union of both corpora and testing on a held-out mix of both.

## Results

**A. Single-source generalization (train on CEAS, test on the unseen Zenodo corpus).**

| Model | F1 | ROC AUC | False positive rate |
|-------|---:|--------:|--------------------:|
| Word baseline (at 0.5) | 0.803 | 0.475 | 0.854 |
| Improved features (at 0.5) | 0.786 | 0.636 | 0.795 |
| Improved features (tuned threshold) | 0.781 | 0.636 | 0.749 |

The character features lift ROC AUC from 0.475 to 0.636. That is the important
number: it means the model now ranks phishing above legitimate on an unseen source
far better than before, where its ranking was essentially a coin toss. The false
positive rate also falls, from 0.85 to 0.75. What does not fully move is the
operating point F1, because one training corpus alone cannot teach the model what
"legitimate" looks like on a different source. This is expected, and it is the honest
limit of a single-source model.

**B. The reverse direction (train on the small Zenodo corpus, test on CEAS).**

Training on the small corpus alone gives roughly coin-toss performance on CEAS
(F1 about 0.45, ROC AUC about 0.45). This confirms that a single small corpus cannot
carry generalization by itself, and that the real fix is training on diverse sources.

**C. The shipped generalized model (trained on the union, tested on a held-out mix of both sources).**

| Metric | Value |
|--------|------:|
| F1 | 0.988 |
| ROC AUC | 0.999 |
| False positive rate | 0.016 |
| Accuracy | 0.987 |

When the improved model is trained on both corpora and evaluated on a held-out 20
percent mix of both, it is strong across the board, and the over flagging is gone:
the false positive rate drops from over 0.80 to about 0.016. This is the model that
is saved as `models/33_binary_generalized.joblib`, alongside the original model
rather than replacing it.

## What this shows, honestly

The feature changes are a real improvement in how well the model ranks phishing on an
unseen source (ROC AUC up by about 0.16), and they cut over flagging. But the
decisive fix for generalization is training on more than one corpus: once the model
has seen both sources, it is accurate and no longer over flags. The remaining honest
caveat is that this uses two corpora, so a genuinely unseen third source is still not
tested; adding more public corpora (for example Nazario, SpamAssassin, Nigerian
Fraud, Enron) would strengthen this further and is the recommended next step. The
scores here also use a stratified CEAS subsample and a capped character vocabulary so
the experiment runs quickly and repeatably; a full-scale rerun on a faster machine
would tighten the estimates without changing the conclusion.

## Reproducing

```bash
# from the project root, in the serving environment
python 33_generalize_binary.py     # writes models/33_binary_generalized.joblib
                                    # and data/33_generalization_report.json
```
