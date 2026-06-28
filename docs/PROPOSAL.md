# Capstone Proposal — A Web-Based Machine-Learning Tool for Phishing Email Detection

**Course:** MSIT 5910-01 — Capstone Project
**Instructor:** Prof. Adebayo Abayomi-Alli
**Assignment:** Activity Unit 2 — Capstone Proposal
**Author:** Hassan Olowofela
**Date:** June 25, 2026

> This is the **Phase 1** proposal for **PhishGuard**. Phase 2 (data acquisition and
> preparation) is implemented in this repository; see the [README](../README.md).

---

## Part 1: Problem Statement

### 1.1 Problem Context

Phishing remains one of the most prevalent and damaging cyber-attack vectors in
information technology. Rather than exploiting technical flaws, it exploits human
trust: an attacker disguises a malicious email as a legitimate message to trick the
recipient into revealing credentials, transferring funds, or installing malware. The
scale is striking. The Federal Bureau of Investigation's Internet Crime Complaint
Center reports that phishing and spoofing were the single most-reported category of
cybercrime in 2024, with 193,407 complaints (more than double the next-largest
category) amid total losses exceeding $16.6 billion (Federal Bureau of Investigation
[FBI], 2025). Because the attack targets the user rather than the system, it bypasses
many perimeter defenses and ranks among the leading initial-access methods in
reported breaches.

This project sits within the cybersecurity domain, at the intersection of applied
machine learning and secure web development. Conventional defenses such as
blacklists, whitelists, signature-based filters, and static rules are increasingly
insufficient, because attackers continuously adapt their wording, spoof trusted
senders, and craft zero-day campaigns that no signature recognizes. A 2024 systematic
review concludes that the most pressing weakness in current defenses is the limited
adaptability of models to new phishing behavior (Kyaw et al., 2024), while
Hosseinzadeh et al. (2025) show that a modern hybrid deep-learning architecture can
reach roughly 96–97% accuracy on a public dataset, confirming that learning-based
detection is both feasible and effective.

The motivation is practical. Working in technical support at an HR/EOR technology
company, the proposer regularly encounters the question "is this email safe?" behind
everyday tickets. Non-technical staff and small teams without a security operations
center rarely have an easy way to check a suspicious message before clicking. This
project addresses that gap by packaging an accurate detection model behind a simple,
accessible web interface.

### 1.2 Stakeholders

Solving this problem creates value for several groups who are directly or indirectly
affected by phishing:

- **Non-technical end users and employees:** those most frequently targeted. The tool
  gives them an immediate, self-service way to vet a suspicious message before acting
  on it, reducing credential theft and financial loss.
- **Organizations and IT/Support teams:** small and mid-sized organizations without a
  dedicated security team gain a lightweight layer of defense and fewer successful
  social-engineering incidents, lowering breach-related cost and downtime.
- **Help-desk and support personnel:** frontline staff who field "is this safe?"
  questions gain a consistent, explainable second opinion that reduces ticket volume.
- **Society at large:** lowering the success rate of a common breach entry point
  reduces fraud, identity theft, and the harm that follows large-scale data
  compromises.

### 1.3 Problem Statement

**Problem.** Non-technical users and small teams lack an accessible, on-demand tool to
determine whether an email is a phishing attempt before they act on it, leaving them
exposed to one of the most common causes of security breaches.

**Statement.** This project will design, build, and evaluate PhishGuard, a web-based
application that uses a machine-learning classifier, trained on public labeled email
datasets, to classify a submitted email as phishing or legitimate, present an
explanation of the key features driving each decision, and report standard
performance metrics (accuracy, precision, recall, and F1). The work is completed
within a single eight-week capstone term. The aim is to demonstrate that an accurate,
transparent, and usable phishing-detection tool can be delivered as a self-contained
prototype that meaningfully assists non-expert users.

---

## Part 2: Scope, Goals, and Expected Outcomes

### 2.1 Project Scope

**In scope.** The project deliverable is a self-contained prototype comprising the
following:

- A machine-learning classifier trained on a public, labeled corpus of phishing and
  legitimate emails, with text preprocessing and feature extraction.
- A lightweight web interface through which a user can paste or submit an email and
  receive a phishing / legitimate classification.
- A basic explainability component that surfaces the top features influencing each
  classification (e.g., via SHAP or LIME), so the result is interpretable.
- A documented evaluation of the model using standard metrics on a held-out test set,
  plus a comparison against a simple baseline.

**Out of scope.** To keep the project achievable in eight weeks, the following are
explicitly excluded and treated as future extensions:

- live integration with a corporate mail server or inbox, and real-time interception
  of email traffic;
- malicious-attachment analysis, sandboxed file detonation, and URL / link crawling
  at scale;
- hardened defense against adversarial or AI-generated attacks at production scale;
- multi-language detection and enterprise-grade deployment, scaling, and 24/7
  operations.

**Assumptions.** Suitable public, labeled email datasets remain available under their
licenses; the project is built and evaluated on a single developer machine or
free-tier cloud; and the work is completed within an eight-week term by a single
student.

**Constraints.** Time (eight weeks), budget (free / open-source tools only), compute
(no dedicated GPU cluster), and data (publicly available, ethically usable corpora)
all bound the work.

### 2.2 Project Goals (SMART)

The objectives below follow the SMART framework, meaning each is specific,
measurable, achievable, relevant, and time-bound:

- **Goal 1:** Train a machine-learning classifier that achieves at least 95% accuracy
  and an F1 score of at least 0.95 on a held-out test set drawn from a public labeled
  email dataset, by the end of Week 5.
- **Goal 2:** Deliver a functional web interface that accepts a submitted email and
  returns a phishing / legitimate classification in under three seconds per request,
  by the end of Week 6.
- **Goal 3:** Integrate an explainability feature that displays the top five features
  contributing to each individual classification, by the end of Week 6.
- **Goal 4:** Produce a documented evaluation reporting accuracy, precision, recall,
  F1, and false-positive rate against a baseline, together with a completed
  ethics-and-security review, by the end of Week 7.
- **Goal 5:** Submit the final capstone package (written report, documented
  source-code repository, and working demo) by the end of Week 8.

### 2.3 Expected Outcomes

Success will be demonstrated through the following tangible results:

- **Trained classifier:** a serialized, reproducible model with a documented training
  pipeline.
- **Functional web application:** a working web prototype that classifies submitted
  emails and shows interpretable explanations.
- **Performance evaluation:** a metrics report (accuracy, precision, recall, F1,
  false-positive rate) and a measurable improvement over the baseline detector.
- **Project artifacts:** a documented Git repository, a written capstone report, and a
  recorded or live demonstration of the tool in use.

---

## Part 3: Ethical and Security Implications

### 3.1 Potential Ethical Issues

- **Data privacy.** Emails are inherently personal and often contain personally
  identifiable information. Even public training corpora may include real PII, and any
  email a user submits could contain sensitive content that must not be exposed or
  retained unnecessarily.
- **User consent and transparency.** Users who submit an email must understand what
  happens to that data, whether it is stored, logged, or used to improve the model.
  Transparent notice and meaningful consent are required before any processing.
- **Intellectual property and licensing.** Datasets and open-source libraries carry
  licenses that must be respected, with proper attribution; the project must not
  redistribute data or code in violation of its terms.
- **Algorithmic fairness and bias.** A classifier trained on a skewed corpus may
  perform unevenly, for example flagging legitimate emails from certain writing styles
  or languages more often. False positives wrongly distrust valid senders, while false
  negatives create a false sense of safety; both carry real consequences.

### 3.2 Security Implications

- **Data protection.** Submitted emails may contain credentials, financial details, or
  confidential business information; transmitted or stored insecurely, this data
  becomes an attractive target and a liability.
- **System vulnerabilities.** As a web application, the tool exposes an attack surface
  vulnerable to common threats such as injection, cross-site scripting, and insecure
  dependencies catalogued in the OWASP Top 10 (OWASP Foundation, 2021), and could
  itself be compromised.
- **Access control.** Without proper authentication and least-privilege design,
  unauthorized parties could access the application, its logs, or the underlying model
  and data.
- **Adversarial manipulation.** Attackers may craft emails designed to evade the
  model. While full adversarial robustness is out of scope, the risk must be
  acknowledged and the tool positioned as a decision aid rather than a guarantee.

### 3.3 Proposed Mitigation Measures

- **Encryption and data minimization:** enforce TLS / HTTPS in transit and encrypt any
  stored data at rest (e.g., AES-256). Adopt a process-and-discard model so submitted
  emails are not retained beyond the immediate classification.
- **Secure development:** validate and sanitize all user input, follow OWASP
  secure-coding practices, keep dependencies patched, and apply least-privilege access
  with authentication on administrative functions (OWASP Foundation, 2021).
- **Compliance frameworks:** align data handling with privacy principles such as the
  GDPR and CCPA (lawful basis, purpose limitation, data minimization) and the NIST
  Cybersecurity Framework 2.0 for managing risk (National Institute of Standards and
  Technology [NIST], 2024).
- **Consent, fairness, and transparency:** present a clear privacy notice and obtain
  consent; use de-identified training data; test performance across email types and
  report false-positive and false-negative rates; and frame results as advisory,
  keeping a human in the loop.
- **Professional ethical codes:** adhere to the ACM Code of Ethics and Professional
  Conduct throughout design, development, and evaluation (Association for Computing
  Machinery [ACM], 2018).

---

## Part 4: Project Planning Tools

### 4.1 Gantt Chart

The chart below maps the project across the eight-week capstone term, organized into
six phases. Overlapping bars reflect the realistic concurrency of model development
and web-application work. A filled cell (█) indicates a week in which the activity is
active.

| Phase / Activity | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Phase 1: Initiation & Planning** | █ |  |  |  |  |  |  |  |
| &nbsp;&nbsp;Finalize problem statement & scope | █ |  |  |  |  |  |  |  |
| &nbsp;&nbsp;Literature review & proposal | █ |  |  |  |  |  |  |  |
| **Phase 2: Data Acquisition & Prep** | █ | █ | █ |  |  |  |  |  |
| &nbsp;&nbsp;Acquire & validate datasets | █ | █ |  |  |  |  |  |  |
| &nbsp;&nbsp;Data cleaning & preprocessing |  | █ |  |  |  |  |  |  |
| &nbsp;&nbsp;Feature engineering / extraction |  | █ | █ |  |  |  |  |  |
| **Phase 3: Model Development** |  | █ | █ | █ | █ |  |  |  |
| &nbsp;&nbsp;Baseline model build |  | █ | █ |  |  |  |  |  |
| &nbsp;&nbsp;Primary classifier training |  |  | █ | █ |  |  |  |  |
| &nbsp;&nbsp;Hyperparameter tuning |  |  |  | █ | █ |  |  |  |
| **Phase 4: Web Application** |  |  |  | █ | █ | █ |  |  |
| &nbsp;&nbsp;Front-end design |  |  |  | █ |  |  |  |  |
| &nbsp;&nbsp;Back-end & API integration |  |  |  | █ | █ |  |  |  |
| &nbsp;&nbsp;Explainability module |  |  |  |  | █ | █ |  |  |
| **Phase 5: Testing & Evaluation** |  |  |  |  |  | █ | █ |  |
| &nbsp;&nbsp;Model evaluation (metrics) |  |  |  |  |  | █ |  |  |
| &nbsp;&nbsp;Integration & usability testing |  |  |  |  |  | █ | █ |  |
| &nbsp;&nbsp;Security & ethics review |  |  |  |  |  |  | █ |  |
| **Phase 6: Documentation & Delivery** |  |  |  |  |  |  | █ | █ |
| &nbsp;&nbsp;Final report writing |  |  |  |  |  |  | █ | █ |
| &nbsp;&nbsp;Demo & presentation prep |  |  |  |  |  |  |  | █ |
| &nbsp;&nbsp;Final submission |  |  |  |  |  |  |  | █ |

**Key milestones:**

- Proposal approved (end of Week 1)
- Dataset prepared (end of Week 3)
- Model meets accuracy target (end of Week 5)
- Working web prototype (end of Week 6)
- Evaluation & security review complete (end of Week 7)
- Final submission and demo (end of Week 8)

### 4.2 Resource Justification Table

The following tools, technologies, and datasets are required to execute the project.
All are open-source or free-tier, consistent with the budget constraint.

| Resource / Tool | Category | Justification |
|---|---|---|
| Python 3 | Language | Primary development language; rich ecosystem for ML, NLP, and web work, and the standard for this type of project. |
| scikit-learn | ML library | Provides baseline classifiers, train / test utilities, and the standard metrics (accuracy, precision, recall, F1) used for evaluation. |
| TensorFlow / Keras | Deep learning | Supports the deep-learning classifier; needed to reach the accuracy targets reported in recent literature. |
| pandas & NumPy | Data handling | Efficient loading, cleaning, and manipulation of the email datasets and feature matrices. |
| NLTK / spaCy | NLP toolkit | Tokenization, stop-word removal, and text preprocessing to convert raw email text into model features. |
| SHAP / LIME | Explainability | Generates per-prediction feature attributions, fulfilling the explainability goal and supporting fairness review. |
| Flask / Streamlit | Web framework | Lightweight framework to build the web interface and serve the model with minimal overhead. |
| HTML / CSS / Bootstrap | Front-end | Builds a clean, accessible submission form and results display for non-technical users. |
| Public email datasets | Data | Labeled phishing / legitimate corpora (e.g., SpamAssassin, Nazario phishing corpus, Kaggle phishing-email sets) provide the training and test data, used under their respective licenses. |
| Jupyter / VS Code | IDE / tooling | Interactive experimentation, model prototyping, and code development. |
| Git & GitHub | Version control | Tracks changes, enables reproducibility, and hosts the documented source-code deliverable. |
| Free-tier cloud / local host | Hosting | Hosts the demo prototype (e.g., a free-tier service) for the final demonstration without incurring cost. |

### 4.3 Feasibility Checklist

The checklist evaluates whether the project can be completed successfully across the
technical, operational, and economic dimensions, with schedule and legal / ethical
factors included for completeness.

| Feasibility Criterion | Assessment | Notes |
|---|:--:|---|
| Required ML and web technologies are mature and available | Yes | All tools are open-source, well-documented, and proven for this class of problem. |
| Proposer has the necessary skills (or can acquire them in time) | Yes | Math background plus IT support experience; remaining ML skills are learnable within the term. |
| Suitable labeled datasets exist and are accessible | Yes | Multiple public phishing / legitimate corpora are freely available under usable licenses. |
| Compute resources are sufficient | Yes | Models run on a standard machine or free-tier cloud; no dedicated GPU cluster required. |
| Scope is achievable within the eight-week term | Yes | Scope is deliberately bounded; out-of-scope items are deferred to future work. |
| Schedule is realistic for eight weeks | Partial | The timeline is compressed; overlapping phases require disciplined time management and weekly milestones. |
| Project can run without budget / paid licensing | Yes | Entirely open-source and free-tier stack; economic feasibility is strong. |
| Operational deployment for a demo is practical | Yes | A single-instance demo is simple to host; full production deployment is out of scope. |
| Legal, ethical, and privacy requirements can be met | Yes | Addressed through de-identification, consent notice, encryption, and compliance alignment (Part 3). |
| Outcomes are measurable and demonstrable | Yes | Standard metrics and a working prototype provide clear, objective evidence of success. |

**Conclusion.** Across technical, operational, and economic dimensions, the project is
feasible. The single notable risk, schedule pressure from the compressed eight-week
timeline, is managed through the milestone structure and phased schedule shown in the
Gantt chart.

---

## References

Association for Computing Machinery. (2018). *ACM code of ethics and professional conduct.* https://www.acm.org/code-of-ethics

Federal Bureau of Investigation. (2025). *2024 internet crime report.* U.S. Department of Justice, Internet Crime Complaint Center. https://www.ic3.gov/AnnualReport/Reports/2024_IC3Report.pdf

Hosseinzadeh, M., Ali, U., Ali, S., Abbaszadi, R., Gharehchopogh, F. S., Khoshvaght, P., … Lansky, J. (2025). Improving phishing email detection performance through deep learning with adaptive optimization. *Scientific Reports, 15*(1), 36724. https://doi.org/10.1038/s41598-025-20668-5

Kyaw, P. H., Gutierrez, J., & Ghobakhlou, A. (2024). A systematic review of deep learning techniques for phishing email detection. *Electronics, 13*(19), 3823. https://doi.org/10.3390/electronics13193823

National Institute of Standards and Technology. (2024). *The NIST cybersecurity framework (CSF) 2.0* (NIST CSWP 29). U.S. Department of Commerce. https://doi.org/10.6028/NIST.CSWP.29

OWASP Foundation. (2021). *OWASP top 10:2021.* https://owasp.org/Top10/
