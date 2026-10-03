"""
Phase 6: Baseline Figures Generator.
Generates publication-quality figures comparing DistilBERT baseline classifier
and DataSentinel sentinel comparator on the benchmark test set.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "baseline")

os.makedirs(FIGURES_DIR, exist_ok=True)

with open(os.path.join(EXP_DIR, "results.json"), "r", encoding="utf-8") as f:
    results = json.load(f)

# Style setup
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 1. Overall Metrics Bar Chart
metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
distilbert_vals = [
    results["baseline_distilbert"]["overall"]["accuracy"] * 100,
    results["baseline_distilbert"]["overall"]["precision"] * 100,
    results["baseline_distilbert"]["overall"]["recall"] * 100,
    results["baseline_distilbert"]["overall"]["f1_score"] * 100
]
datasentinel_vals = [
    results["datasentinel_comparator"]["overall"]["accuracy"] * 100,
    results["datasentinel_comparator"]["overall"]["precision"] * 100,
    results["datasentinel_comparator"]["overall"]["recall"] * 100,
    results["datasentinel_comparator"]["overall"]["f1_score"] * 100
]

x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, distilbert_vals, width, label="Baseline DistilBERT Classifier", color="#2563eb")
rects2 = ax.bar(x + width/2, datasentinel_vals, width, label="DataSentinel Sentinel Comparator", color="#f97316")

ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_title("Overall Performance: DistilBERT Baseline vs. DataSentinel Sentinel", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=11)
ax.set_ylim(80, 105)
ax.legend(loc="lower right")

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "baseline_vs_datasentinel_comparison.png"), dpi=200)
plt.close(fig)

# 2. Same-Task vs Cross-Task Vulnerability (Research Gap G1)
g1_labels = ["Same-Task Attacks\n(Vulnerability Gap G1)", "Cross-Task Attacks"]
g1_distilbert = [
    results["baseline_distilbert"]["same_task"]["recall"] * 100,
    results["baseline_distilbert"]["cross_task"]["recall"] * 100
]
g1_datasentinel = [
    results["datasentinel_comparator"]["same_task"]["recall"] * 100,
    results["datasentinel_comparator"]["cross_task"]["recall"] * 100
]

x = np.arange(len(g1_labels))
fig, ax = plt.subplots(figsize=(7, 5))
rects1 = ax.bar(x - width/2, g1_distilbert, width, label="Baseline DistilBERT", color="#10b981")
rects2 = ax.bar(x + width/2, g1_datasentinel, width, label="DataSentinel Sentinel", color="#ef4444")

ax.set_ylabel("Attack Detection Recall (%)", fontsize=12)
ax.set_title("Detection Recall by Task Relation: Same-Task vs Cross-Task", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(g1_labels, fontsize=11)
ax.set_ylim(40, 110)
ax.legend(loc="lower right")

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "same_vs_cross_task_vulnerability.png"), dpi=200)
plt.close(fig)

# 3. Per-Attack Recall Comparison
attacks = ["naive", "escape", "ignore", "fake_comp", "combine"]
att_distilbert = [results["baseline_distilbert"]["per_attack"][a]["recall"] * 100 for a in attacks]
att_datasentinel = [results["datasentinel_comparator"]["per_attack"][a]["recall"] * 100 for a in attacks]

x = np.arange(len(attacks))
fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(x - width/2, att_distilbert, width, label="Baseline DistilBERT", color="#6366f1")
rects2 = ax.bar(x + width/2, att_datasentinel, width, label="DataSentinel Sentinel", color="#ec4899")

ax.set_ylabel("Recall (%)", fontsize=12)
ax.set_title("Detection Recall Across 5 Core Attack Strategies", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([a.replace("_", " ").title() for a in attacks], fontsize=11)
ax.set_ylim(70, 108)
ax.legend(loc="lower right")

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "per_attack_recall_comparison.png"), dpi=200)
plt.close(fig)

print("Generated all figures successfully in:", FIGURES_DIR)
