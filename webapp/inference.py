"""PhishGuard serving: model loading, prediction, and explanation.

Loads the artifacts produced by Phase 3 (the fitted TF-IDF vectorizer and the
calibrated, scaled logistic-regression classifier) and exposes a single
``PhishGuardModel.predict`` method that returns, for one submitted email:

  * a label (phishing or legitimate),
  * a calibrated phishing probability, and
  * the top features that drove the decision, in human terms.

Because the final model is linear, the explanation is an exact linear
attribution: contribution_i = coefficient_i * scaled_value_i. The coefficients
and per-feature scales are read from the calibrated model's underlying
MaxAbsScaler + LogisticRegression sub-estimators (averaged across the
calibration folds), which is the same quantity SHAP reports for a linear model.
"""
import os
import re

import joblib
import numpy as np

from features import STRUCT_ORDER, build_combined

DEFAULT_THRESHOLD = 0.5   # favors recall (catching phishing); tune per deployment

# Free email providers: legitimate people use these, but so do impersonators.
FREEMAIL = (
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "icloud.com", "proton.me", "protonmail.com", "mail.com", "gmx.com",
)


def _sender_domain(sender):
    m = re.search(r"[\w.+-]+@([\w.-]+\.[\w-]+)", sender or "")
    return m.group(1).lower() if m else None


def sender_notes(sender):
    """Transparent, non-model heuristics about the From address (advisory context)."""
    notes = []
    if not sender or not sender.strip():
        return notes
    dom = _sender_domain(sender)
    if dom is None:
        notes.append("The sender could not be read as a valid email address, which is itself worth a second look.")
        return notes
    if dom in FREEMAIL:
        notes.append(f"The sender uses a free email provider ({dom}). Real people use these, but so do impersonators posing as a company.")
    if re.search(r"\d", dom.split(".")[0]) or dom.count("-") >= 2:
        notes.append(f"The sender domain ({dom}) has an unusual look. Check it carefully against the organization it claims to be.")
    return notes


class PhishGuardModel:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
        self.clf = joblib.load(os.path.join(models_dir, "09_classifier_scaled.joblib"))
        self.feature_names = list(STRUCT_ORDER) + list(self.vectorizer.get_feature_names_out())
        self.n_struct = len(STRUCT_ORDER)
        self.coef, self.scale = self._extract_linear()

    def _extract_linear(self):
        """Average coefficients and MaxAbsScaler scales across calibration folds."""
        coefs, scales = [], []
        for cc in getattr(self.clf, "calibrated_classifiers_", []):
            est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            steps = getattr(est, "named_steps", {}) if est is not None else {}
            scaler = steps.get("maxabsscaler")
            lr = steps.get("logisticregression")
            if scaler is None or lr is None:
                continue
            coefs.append(np.ravel(lr.coef_))
            scales.append(np.asarray(scaler.scale_, dtype=float))
        if not coefs:
            return None, None
        return np.mean(coefs, axis=0), np.mean(scales, axis=0)

    def _explain(self, X, top_k):
        if self.coef is None:
            return []
        x = X.toarray().ravel()
        safe_scale = np.where(self.scale == 0, 1.0, self.scale)
        contrib = self.coef * (x / safe_scale)
        order = np.argsort(np.abs(contrib))[::-1]
        out = []
        for i in order:
            if contrib[i] == 0:
                continue
            out.append({
                "feature": self.feature_names[i],
                "kind": "structural" if i < self.n_struct else "content",
                "contribution": round(float(contrib[i]), 4),
                "pushes": "phishing" if contrib[i] > 0 else "legitimate",
            })
            if len(out) >= top_k:
                break
        return out

    def predict(self, body, subject="", sender="", threshold=DEFAULT_THRESHOLD, top_k=5):
        """Classify one email and explain the decision.

        Args:
            body: the email body (plain text or HTML).
            subject: the subject line (optional but improves the structural signals).
            sender: the From address (optional). It is folded into the analyzed
                text, exactly as a From line would be if pasted with the email,
                and it also drives a few transparent sender heuristics.
            threshold: probability at or above which the email is flagged.
            top_k: how many contributing features to return.
        """
        body_used = (f"From: {sender}\n{body}" if sender and sender.strip() else body)
        X, clean, feats = build_combined(subject, body_used, self.vectorizer)
        proba = float(self.clf.predict_proba(X)[0, 1])
        return {
            "label": "phishing" if proba >= threshold else "legitimate",
            "probability": round(proba, 4),
            "threshold": threshold,
            "top_features": self._explain(X, top_k),
            "sender_notes": sender_notes(sender),
            "n_features": X.shape[1],
        }


_MODEL = None


def get_model(models_dir=None):
    """Lazily load a single shared model instance (handy for a web server)."""
    global _MODEL
    if _MODEL is None:
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        _MODEL = PhishGuardModel(models_dir)
    return _MODEL


# ============================================================================
# Multiclass model (Safe / Scam / Malware) - Phase 4 extension
# ============================================================================
# Trained by 20_train_multiclass.py on the Zenodo multiclass corpus. This detects
# malware-LURE emails from their wording; it is NOT antivirus and does not scan
# attachments. The Malware class is a small-data proof of concept.

STRUCT_PHRASE = {
    "has_url": "Contains a link",
    "num_urls": "Contains one or more links",
    "urgent_word_count": "Uses urgency or pressure language",
    "has_money_symbol": "Mentions money or payment",
    "num_money_symbols": "Mentions money or payment",
    "num_exclaim": "Heavy use of exclamation marks",
    "uppercase_ratio": "Uses shouting (many capital letters)",
    "digit_ratio": "Unusual amount of numbers",
    "num_digits": "Unusual amount of numbers",
}

RECOMMENDED_ACTION = {
    "safe": "No strong threat signals were found. Still, verify unexpected requests through a channel you already trust.",
    "scam": "Treat this as a likely scam. Do not click links or share personal information, verify with the organization directly, and report it.",
    "malware": "Treat this as a likely malware-delivery email. Do not open attachments or enable content, delete it, and report it to your IT team.",
}

POC_NOTE = ("Malware detection here is a small-data proof of concept based on the email's "
            "wording, not on scanning any attachment.")


class PhishGuardMulticlass:
    def __init__(self, models_dir):
        self.model = joblib.load(os.path.join(models_dir, "20_multiclass_model.joblib"))
        self.vectorizer = joblib.load(os.path.join(models_dir, "20_multiclass_vectorizer.joblib"))
        self.clf = self.model.named_steps["logisticregression"]
        self.scaler = self.model.named_steps["maxabsscaler"]
        self.classes = list(self.clf.classes_)
        self.feature_names = list(STRUCT_ORDER) + list(self.vectorizer.get_feature_names_out())
        self.n_struct = len(STRUCT_ORDER)

    def _flags(self, X, ci, verdict, feats, sender):
        x = X.toarray().ravel()
        scale = np.where(self.scaler.scale_ == 0, 1.0, self.scaler.scale_)
        contrib = self.clf.coef_[ci] * (x / scale)
        order = np.argsort(np.abs(contrib))[::-1]
        flags, seen, terms = [], set(), []
        for i in order:
            if contrib[i] <= 0:
                continue
            if i < self.n_struct:
                phrase = STRUCT_PHRASE.get(self.feature_names[i])
                if phrase and phrase not in seen and feats.get(self.feature_names[i], 0):
                    flags.append({"text": phrase, "kind": "signal"}); seen.add(phrase)
            elif len(terms) < 3:
                terms.append(self.feature_names[i])
            if len(flags) >= 4:
                break
        if verdict != "safe" and terms:
            kind = "malware-delivery" if verdict == "malware" else "scam"
            quoted = ", ".join("'%s'" % t for t in terms)
            flags.append({"text": f"Wording associated with {kind} emails: {quoted}", "kind": "wording"})
        for note in sender_notes(sender):
            flags.append({"text": note, "kind": "sender"})
        if verdict == "safe" and not flags:
            flags.append({"text": "No strong phishing or malware signals were found in the text.", "kind": "ok"})
        return flags

    def predict(self, body, subject="", sender=""):
        body_used = (f"From: {sender}\n{body}" if sender and sender.strip() else body)
        X, clean, feats = build_combined(subject, body_used, self.vectorizer)
        p = self.model.predict_proba(X)[0]
        ci = int(np.argmax(p))
        verdict = self.classes[ci]
        probs = {c: round(float(p[self.classes.index(c)]), 4) for c in ("safe", "scam", "malware")}
        return {
            "verdict": verdict,
            "confidence": round(float(p[ci]), 4),
            "probabilities": probs,
            "flags": self._flags(X, ci, verdict, feats, sender),
            "recommended_action": RECOMMENDED_ACTION[verdict],
            "poc_note": POC_NOTE if verdict == "malware" else "",
            "n_features": X.shape[1],
        }


_MC_MODEL = None


def get_multiclass_model(models_dir=None):
    global _MC_MODEL
    if _MC_MODEL is None:
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        _MC_MODEL = PhishGuardMulticlass(models_dir)
    return _MC_MODEL
