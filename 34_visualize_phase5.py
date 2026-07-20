#!/usr/bin/env python3
"""Phase 5 visualiser: turn the JSON reports into charts you can look at.

Reads the reports written by the Phase 5 scripts and renders three figures:
  1. The cross-corpus gap (in-corpus vs unseen-corpus performance).
  2. The generalization fix (word baseline vs improved features vs the union model).
  3. Robustness under adversarial tricks (confidence held per perturbation).

Run it after the other scripts have produced their JSON reports:
    python 34_visualize_phase5.py
Figures are written to screenshots/phase5/ (override with PHISHGUARD_FIGS).
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.environ.get("PHISHGUARD_FIGS") or os.path.join(HERE, "screenshots", "phase5")
os.makedirs(OUT, exist_ok=True)

NAVY, TEAL, AMBER, RED, GREEN, GREY = "#1E2761", "#1C7293", "#E0A800", "#C0392B", "#1F9E5A", "#8895A7"


def load(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_cross_corpus(r):
    if not r:
        return
    inc = r.get("in_corpus_reference_test_f1", 0.9957)
    ood = r["ood_default_threshold_0.5"]
    labels = ["In corpus\n(CEAS)", "Unseen corpus\n(Zenodo)"]
    f1 = [inc, ood["f1"]]
    auc = [0.99, ood["roc_auc"]]
    fpr = [0.02, ood["false_positive_rate"]]
    x = range(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - w for i in x], f1, w, label="F1", color=TEAL)
    ax.bar([i for i in x], auc, w, label="ROC AUC", color=NAVY)
    ax.bar([i + w for i in x], fpr, w, label="False positive rate", color=RED)
    for i, vals in enumerate(zip(f1, auc, fpr)):
        for j, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#333")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("Cross-corpus gap: strong in corpus, weaker and over-flagging on an unseen source")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "phase5_cross_corpus.png"), dpi=140); plt.close(fig)


def fig_generalization(r):
    if not r:
        return
    A = r["A_single_source_generalization_CEAS_to_Zenodo"]
    C = r["C_shipped_model_union_heldout_20pct_mixed_sources"]
    groups = ["Word baseline\n(unseen)", "Improved features\n(unseen)", "Union model\n(mixed held-out)"]
    f1 = [A["baseline_word_at_0.5"]["f1"], A["improved_char_balanced_dropsubject_at_0.5"]["f1"], C["f1"]]
    auc = [A["baseline_word_at_0.5"]["roc_auc"], A["improved_char_balanced_dropsubject_at_0.5"]["roc_auc"], C["roc_auc"]]
    fpr = [A["baseline_word_at_0.5"]["false_positive_rate"], A["improved_char_balanced_dropsubject_at_0.5"]["false_positive_rate"], C["false_positive_rate"]]
    x = range(len(groups)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar([i - w for i in x], f1, w, label="F1", color=TEAL)
    ax.bar([i for i in x], auc, w, label="ROC AUC", color=NAVY)
    ax.bar([i + w for i in x], fpr, w, label="False positive rate", color=RED)
    for i, vals in enumerate(zip(f1, auc, fpr)):
        for j, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#333")
    ax.set_xticks(list(x)); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("The generalization fix: better ranking on unseen data, and over-flagging solved by union training")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "phase5_generalization.png"), dpi=140); plt.close(fig)


def fig_robustness(r):
    if not r:
        return
    seeds = r["per_seed"]
    perts = list(seeds[0]["probes"].keys())
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = [s["name"].replace("_", " ") for s in seeds]
    base = [s["baseline_confidence"] for s in seeds]
    x = range(len(seeds))
    ax.plot(x, base, "o-", color=NAVY, linewidth=2, label="baseline")
    palette = [TEAL, AMBER, GREEN, RED, GREY]
    for k, pert in enumerate(perts):
        vals = [s["probes"][pert]["confidence"] for s in seeds]
        ax.plot(x, vals, "o--", color=palette[k % len(palette)], alpha=0.8, label=pert.replace("_", " "))
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Model confidence")
    er = r.get("evasion_rate_to_safe", 0.0)
    ax.set_title(f"Robustness: confidence under evasion tricks (evasions to Safe: {er:.0%})")
    ax.legend(loc="lower left", fontsize=7, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "phase5_robustness.png"), dpi=140); plt.close(fig)


def main():
    fig_cross_corpus(load("30_cross_corpus_report.json"))
    fig_generalization(load("33_generalization_report.json"))
    fig_robustness(load("32_robustness_report.json"))
    print("Wrote figures to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
