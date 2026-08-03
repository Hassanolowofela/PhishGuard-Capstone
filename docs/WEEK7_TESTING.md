# Week 7: System Testing and Maintenance

This document summarizes the final system testing performed on PhishGuard and the
maintenance plan that keeps the tool trustworthy after release. It complements the
phase reports in this folder and the reproducible scripts in the repository root.

## Testing overview

Testing followed a layered approach, checking the tool at the unit, integration,
system, and acceptance levels. Each level is captured as reproducible evidence so
the results can be regenerated from the repository.

### Unit and integration tests

The automated suite in `tests/test_app.py` runs with `pytest`. It exercises the
individual components (text cleaning, the eighteen structural feature calculations,
and the sender heuristics) and the full request path through the Flask application
(index page, health endpoint, empty and missing input, oversized input, three class
verdict, probabilities summing to one, and suspicious sender flagging).

```
pytest -q
```

All eighteen tests pass. Tests that require the trained model are skipped
automatically when the model artifact is absent, so the suite still runs in a clean
checkout.

### Performance test

`benchmark.py` (repository root) times the served model over 500 predictions and
reports model size, feature width, cold start, peak memory, latency, and throughput.

```
python benchmark.py
```

On a typical development machine the model and vectorizer occupy about 160 KB on
disk, mean inference latency is roughly 1.3 milliseconds, and throughput is several
hundred predictions per second on a single thread. This confirms a lightweight,
interpretable model is fast enough for interactive use without special hardware.

### Robustness test

The robustness probes perturb threatening emails with five evasion techniques
(leetspacing, spacing out trigger words, obfuscating links, benign padding, and
unicode look alike characters). Across 25 probes over 5 random seeds, no perturbation
flipped a threatening message to a Safe verdict, and the mean confidence drop on
threats was about four percentage points. The report is written to
`data/32_robustness_report.json` and visualized in `screenshots/phase5/`.

### Acceptance test

The deployed tool was run on a real, unsolicited job offer scam received during the
project. Pasted in exactly as received, PhishGuard returned a verdict of Scam at
about 94 percent confidence, with six interpretable red flags and a plain recommended
action. This demonstrates the complete system working end to end on genuine, unseen
input and meeting the interpretability goal in practice.

### API demonstration

`webapp/api_demo.py` exercises both HTTP endpoints through the Flask test client, so
the health check and a live analysis request can be reproduced without starting a
separate server.

```
cd webapp
python api_demo.py
```

### Defects found and resolved

- A data handling error raised an exception on messages with a missing field; fixed
  by defaulting absent fields to empty strings.
- A sender flag assertion initially failed; corrected so the heuristic inspects the
  sender domain.
- A probability equality check was made tolerant to floating point rounding.

## Maintenance plan

Maintenance is treated as planned, ongoing work across four types:

- **Corrective:** fix defects reported by users or surfaced by monitoring; reproduce
  with a new test, fix, and release.
- **Adaptive:** keep the tool working as its environment changes (Python, scikit-learn,
  deployment host); dependency versions are pinned so upgrades are deliberate.
- **Perfective:** improve what already works, such as clearer explanations and
  interface refinements gathered from usability feedback.
- **Preventive:** refactor, add tests, and watch for model drift as phishing tactics
  evolve.

A practical schedule pairs these with a cadence: monthly dependency and security
review, quarterly model review against fresh samples, an annual full evaluation, and
immediate response for any security or correctness defect.

### Version control and documentation

Work follows a two branch workflow, with day to day changes on a development branch
and reviewed changes merged into `main` through pull requests. Tagged releases mark
milestones and record version updates, bug fixes, and enhancements in their notes.
Configuration such as dependency versions and model artifacts is kept under the same
version control, so any past state of the tool is reproducible.

### Risk identification and mitigation

- **Model drift** as phishing language changes: scheduled retraining and evaluation
  against new data.
- **Dependency and security vulnerabilities:** pinned versions with a monthly review.
- **Loss of institutional knowledge:** thorough documentation and a clean version
  history.
- **Silent failure in production:** health checks and logging so failures are visible
  quickly.
