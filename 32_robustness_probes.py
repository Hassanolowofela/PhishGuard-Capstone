#!/usr/bin/env python3
"""Phase 5, Test 3: robustness and adversarial probes on the deployed model.

PhishGuard reads wording, not attachments, so an attacker who lightly disguises a
message might slip past it. This script measures that risk honestly. It takes a set
of emails the model already flags as a threat (scam or malware), applies simple,
realistic evasion tricks an attacker could use, and records whether the verdict
flips to Safe (a successful evasion) and how far the confidence moves.

It probes the real serving model through the same inference module the web app
uses, so the numbers reflect the shipped behaviour. Point it at a models directory
with PHISHGUARD_MODELS, or it defaults to ./models.

Output: data/32_robustness_report.json
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "webapp"))
from inference import get_multiclass_model  # noqa: E402

MODELS_DIR = os.environ.get("PHISHGUARD_MODELS") or os.path.join(HERE, "models")

# Seed threats: each should be flagged scam or malware at baseline.
SEEDS = [
    {"name": "account_suspend_scam",
     "sender": "security@paypa1-verify.com",
     "subject": "Urgent: your account will be suspended",
     "body": "Your account will be suspended within 24 hours. Verify your password now "
             "by clicking http://paypa1-verify.com/login to confirm your identity."},
    {"name": "prize_baiting_scam",
     "sender": "rewards@claim-prize-now.net",
     "subject": "Congratulations, you won a prize",
     "body": "You are our lucky winner. Claim your free gift card now. Click the link and "
             "confirm your bank details to receive your payment immediately."},
    {"name": "invoice_malware_lure",
     "sender": "billing@micros0ft-invoices.com",
     "subject": "Invoice overdue, action required",
     "body": "Please open the attached invoice document and enable content to view the "
             "payment details. Your account will incur charges if not settled today."},
    {"name": "delivery_scam",
     "sender": "no-reply@track-delivery-parcel.com",
     "subject": "Your package could not be delivered",
     "body": "We could not deliver your package. Update your address and pay the small "
             "customs fee now at http://track-delivery-parcel.com to release the parcel."},
    {"name": "security_alert_scam",
     "sender": "alerts@bank-secure-login.com",
     "subject": "Security alert on your account",
     "body": "We detected unusual login activity. Verify your identity and reset your "
             "password immediately or your account access will be permanently limited."},
]

LEET = str.maketrans({"o": "0", "i": "1", "a": "@", "e": "3", "s": "$"})
TRIGGERS = ["verify", "account", "password", "suspended", "click", "payment",
            "bank", "login", "invoice", "prize", "winner", "free", "urgent"]
HOMO = str.maketrans({"a": "а", "e": "е", "o": "о", "c": "с", "p": "р"})


def leetspeak(email):
    e = dict(email); e["body"] = email["body"].translate(LEET); return e


def space_out_triggers(email):
    body = email["body"]
    for w in TRIGGERS:
        spaced = " ".join(w)
        body = body.replace(w, spaced).replace(w.capitalize(), spaced)
    e = dict(email); e["body"] = body; return e


def obfuscate_links(email):
    body = email["body"].replace("http://", "hxxp://").replace("https://", "hxxps://").replace(".", "[.]")
    e = dict(email); e["body"] = body; return e


def benign_padding(email):
    pad = ("Thanks for your continued membership. We appreciate your business and hope you "
           "have a wonderful day. Here is our regular monthly newsletter with helpful tips. ")
    e = dict(email); e["body"] = pad + email["body"] + " " + pad; return e


def homoglyphs(email):
    e = dict(email); e["body"] = email["body"].translate(HOMO); return e


PERTURBATIONS = {
    "leetspeak": leetspeak,
    "space_out_triggers": space_out_triggers,
    "obfuscate_links": obfuscate_links,
    "benign_padding": benign_padding,
    "unicode_homoglyphs": homoglyphs,
}


def verdict_of(model, email):
    r = model.predict(body=email["body"], subject=email["subject"], sender=email["sender"])
    return r["verdict"], round(float(r["confidence"]), 4)


def main():
    model = get_multiclass_model(MODELS_DIR)

    per_seed, rows = [], []
    evasions = 0
    total_probes = 0
    conf_drops = []

    for seed in SEEDS:
        base_v, base_c = verdict_of(model, seed)
        seed_rec = {"name": seed["name"], "baseline_verdict": base_v,
                    "baseline_confidence": base_c, "probes": {}}
        for pname, fn in PERTURBATIONS.items():
            pv, pc = verdict_of(model, fn(seed))
            flipped_to_safe = (base_v in ("scam", "malware")) and (pv == "safe")
            still_threat = pv in ("scam", "malware")
            seed_rec["probes"][pname] = {
                "verdict": pv, "confidence": pc,
                "verdict_changed": pv != base_v,
                "evaded_to_safe": flipped_to_safe,
                "confidence_delta": round(pc - base_c, 4)}
            total_probes += 1
            if flipped_to_safe:
                evasions += 1
            if base_v in ("scam", "malware"):
                conf_drops.append(base_c - pc)
            rows.append((seed["name"], pname, base_v, pv, flipped_to_safe))
        per_seed.append(seed_rec)

    # per-perturbation evasion tally
    by_perturbation = {}
    for pname in PERTURBATIONS:
        flips = sum(1 for r in rows if r[1] == pname and r[4])
        by_perturbation[pname] = {"evasions": flips, "of_seeds": len(SEEDS)}

    report = {
        "test": "robustness_adversarial_probes",
        "model": "multiclass Safe/Scam/Malware (serving model)",
        "n_seeds": len(SEEDS), "n_probes": total_probes,
        "evasion_rate_to_safe": round(evasions / total_probes, 4) if total_probes else None,
        "mean_confidence_drop_on_threats": round(sum(conf_drops) / len(conf_drops), 4) if conf_drops else None,
        "by_perturbation": by_perturbation,
        "per_seed": per_seed,
        "interpretation_hint": ("A high evasion rate or large confidence drop shows the "
                                "model is sensitive to surface wording tricks. Because "
                                "PhishGuard reads text, not attachments, this documents a "
                                "known limitation rather than a defect."),
    }
    with open(os.path.join(HERE, "data", "32_robustness_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ["evasion_rate_to_safe", "mean_confidence_drop_on_threats", "by_perturbation"]}, indent=2))


if __name__ == "__main__":
    main()
