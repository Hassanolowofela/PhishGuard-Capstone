# PhishGuard Phase 5: Testing and Evaluation

**Project:** PhishGuard, an interpretable web based tool for phishing and malware email detection
**Course:** MSIT 5910 Capstone Project
**Author:** Hassan Olowofela

## Overview

Phase 3 measured PhishGuard on held out, leakage controlled test data from a single corpus, and Phase 4 built the web application around a three class Safe, Scam, and Malware model. Those results are strong, but a strong in corpus score can hide weaknesses that only appear outside the training data, at the boundaries of the interface, or under deliberate evasion. Phase 5 stress tests the system along four independent lines: how well the model generalises to a new email source, whether the web application behaves correctly end to end, how the model holds up against simple adversarial tricks, and whether a non technical person can actually use the tool. The guiding principle throughout is honesty: the goal is to find and document real limits, not to produce flattering numbers.

## Test 1: Cross corpus generalization

**Question.** The headline binary phishing model reached about 0.9957 test F1 on the corpus it was trained on (CEAS_08). Does it still work on email from a completely different source and format that it never saw?

**Method.** The trained model (`models/09_classifier_scaled.joblib`) was evaluated, without any retraining, on the Zenodo multiclass dataset, which is the independent source used for the Phase 4 model and was never seen by the binary model. Its threat categories were mapped to phishing and its NOT-Malicious class to legitimate. Features were rebuilt exactly as in training: the eighteen structural features first, then the 5000 term TF-IDF from the same cleaning pipeline. The script is `30_cross_corpus_eval.py` and the full output is `data/30_cross_corpus_report.json`.

**Results.** On 606 out of distribution emails (171 legitimate, 435 phishing):

| Metric | In corpus (CEAS_08) | Out of corpus (Zenodo) |
|--------|--------------------:|-----------------------:|
| F1 | 0.9957 | 0.8394 |
| Accuracy | about 0.99 | 0.7393 |
| Recall (phishing) | high | 0.9494 |
| Precision (phishing) | high | 0.7523 |
| ROC AUC | about 0.99 | 0.6741 |
| False positive rate | low | 0.7953 |

**Interpretation.** The model generalises only partially. It still catches most phishing on the new source (recall 0.95), but it does so by over flagging: it labels about eight in ten legitimate messages as phishing, so precision falls to 0.75 and the ROC AUC of 0.67 shows the underlying ranking is far weaker than the in corpus 0.99. A large part of the cause is vocabulary. Only about 39 percent of the tokens in the new corpus appear in the CEAS trained TF-IDF vocabulary, so on unfamiliar text the model leans on a few learned danger words and defaults toward a phishing verdict. This is an honest and expected finding for a single corpus model, and it is the clearest evidence that a future version should train on multiple corpora to travel well.

## Test 2: Integration tests for the web application

**Question.** Does the running application handle real requests correctly, including malformed and oversized input, not just the functions in isolation?

**Method.** A new suite, `tests/test_app.py`, drives the Flask application through its HTTP endpoints with the test client. Five tests need no model and always run: the home page renders, `/health` returns valid JSON with the expected keys, an empty submission short circuits to an empty flag without touching the model, a missing body is handled, and oversized input is truncated to the configured limits before any work is done. Three further tests exercise the real classification path (a three class verdict is returned, the class probabilities sum to one, and a suspicious sender produces sender notes). These are gated by a guard that first checks the model can both load and run a prediction, so they run on the pinned serving environment and skip cleanly where the model cannot execute.

**Results.** All applicable tests pass. In the current sandbox, which runs an older scikit-learn than the one the model was saved with, the five endpoint tests pass and the three model path tests skip by design; on the pinned `webapp/requirements.txt` environment the model path tests run for real. The suite is wired into the existing continuous integration pipeline, so these checks now run on every push and pull request alongside the earlier unit tests.

## Test 3: Robustness and adversarial probes

**Question.** PhishGuard reads wording, not attachments, so can an attacker slip a threat past it with light disguises?

**Method.** The script `32_robustness_probes.py` takes five seed emails and applies five realistic evasion tricks to each: leetspeak character swaps, spacing out trigger words, obfuscating links, padding the message with benign text, and swapping letters for unicode lookalikes. For each of the twenty five probes it records whether the verdict flips to Safe (a successful evasion) and how far the confidence moves, using the same serving model as the web app. Full output is `data/32_robustness_report.json`.

**Results.** No probe ever downgraded a threat to Safe; the evasion rate to Safe was zero across all twenty five attempts, and the average confidence drop on threats was only about four points. Scam detection was especially stable, holding above 0.98 confidence under every trick. The weakness is concentrated in the malware class, which is the smallest and lowest scoring class from Phase 4: benign padding and unicode homoglyphs cut the malware confidence sharply, by up to about 0.34, and reclassified the malware lure as a scam. One additional finding worth recording is that one of the five seed scams was scored Safe at baseline, at a confidence of 0.49 just under the decision boundary, a reminder that borderline scams do slip through.

**Interpretation.** The model is more robust than expected to surface level wording attacks, which is reassuring, but the malware class is fragile and the borderline miss shows the model should be treated as a helpful second opinion rather than a gate. Because PhishGuard analyses text and not attachments, this is a documented limitation of scope, not a defect.

## Test 4: Usability check

**Question.** Can a non technical person use PhishGuard, understand the verdict, and act on it?

**Status.** Usability requires real users, so Phase 5 delivers a ready to run instrument rather than invented results. The plan in `docs/PHASE5_USABILITY_PLAN.md` specifies a short moderated test for two to five non technical participants, with four task scenarios, moderator observation metrics, an adapted System Usability Scale survey with its scoring rule, and an analysis template. It is designed to surface the most serious problems in about twenty minutes per participant, and it collects no personal data, consistent with the privacy stance of the project. The scored results will be recorded in that template and folded into the Phase 6 documentation once the sessions are run.

## Summary of findings

| Test | Result | What it tells us |
|------|--------|------------------|
| Cross corpus generalization | F1 0.84, accuracy 0.74, ROC AUC 0.67 out of corpus | Catches phishing but over flags legitimate mail on a new source; needs multi corpus training to generalise |
| Integration tests | All applicable tests pass, wired into CI | The application handles routing, empty, malformed, and oversized input correctly |
| Robustness probes | 0 of 25 evasions to Safe; malware class sensitive | Robust to surface wording tricks; malware is the weak class; borderline scams can slip through |
| Usability | Instrument and plan delivered; sessions pending | A concrete, low cost way to test the non expert usability goal |

## Limitations and honest caveats

The cross corpus test uses one alternative corpus, so it estimates a single generalization gap rather than mapping the full range. The robustness probes cover five common tricks and five seeds, so they are illustrative rather than exhaustive, and a determined attacker with heavier rewriting could do more. The usability results are not yet in. None of these caveats undermine the headline conclusion: PhishGuard is accurate and stable within its training distribution and against light evasion, but it is a single corpus, text only tool whose main risk is over flagging unfamiliar legitimate mail, which is exactly why the interface presents a confidence score and reasons rather than a bare verdict.

## Reproducibility

```bash
# from the project root, in the serving environment
python 30_cross_corpus_eval.py          # writes data/30_cross_corpus_report.json
python 32_robustness_probes.py          # writes data/32_robustness_report.json
python -m pytest tests/                 # runs unit and integration tests
```

Each script writes a JSON report into `data/`, so every number in this report can be regenerated.
