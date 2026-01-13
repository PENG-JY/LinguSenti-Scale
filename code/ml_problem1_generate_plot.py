#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate plots for ML Problem 1: Predict High-Engagement Reddit Posts

This script loads:
- metrics CSV
- predictions CSV

And generates:
1. ROC Curve
2. Probability Distribution by True Label
3. Confusion Matrix Heatmap

Plots are saved to:
    data/plots/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ======================
# Paths
# ======================

METRICS_CSV = "data/csv/ml_problem1_high_engagement_metrics.csv"
PRED_CSV = "data/csv/ml_problem1_high_engagement_predictions.csv"
PLOT_DIR = "data/plots/"


# ======================
# Ensure plot dir exists
# ======================
os.makedirs(PLOT_DIR, exist_ok=True)


# =========================================================
# Load Data
# =========================================================
print("[INFO] Loading CSV files...")
metrics = pd.read_csv(METRICS_CSV)
pred_df = pd.read_csv(PRED_CSV)


# =========================================================
# 1. ROC Curve
# =========================================================
def plot_roc():
    y_true = pred_df["label_high_engagement"]
    y_prob = pred_df["probability_positive"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.title("ROC Curve for High-Engagement Prediction")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    out_path = os.path.join(PLOT_DIR, "roc_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] roc_curve.png saved.")


# =========================================================
# 2. Probability Distribution
# =========================================================
def plot_probability_distribution():
    plt.figure(figsize=(8, 6))

    sns.histplot(
        data=pred_df,
        x="probability_positive",
        hue="label_high_engagement",
        bins=30,
        kde=True,
        palette={0: "#1f77b4", 1: "#ff7f0e"},
        alpha=0.6
    )

    plt.title("Prediction Probability Distribution by True Label")
    plt.xlabel("Predicted Probability of High Engagement")
    plt.ylabel("Count")

    out_path = os.path.join(PLOT_DIR, "probability_by_label.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] probability_by_label.png saved.")


# =========================================================
# 3. Confusion Matrix
# =========================================================
def plot_confusion_matrix():
    y_true = pred_df["label_high_engagement"]
    y_pred = pred_df["prediction"]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix (Prediction vs. True Label)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    out_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] confusion_matrix.png saved.")


# ================================
# Run all plots
# ================================
if __name__ == "__main__":
    plot_roc()
    plot_probability_distribution()
    plot_confusion_matrix()

    print("[DONE] All business-question-driven plots generated in data/plots/")


