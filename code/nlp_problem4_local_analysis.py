#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Problem 4 — Local Analysis (from Parquet outputs)

Reads:
    data/nlp_q4/top_words/
    data/nlp_q4/linguistic_stats/
    data/nlp_q4/submission_features/

Outputs:
    data/plots/nlp_problem4_top_words.png
    data/plots/nlp_problem4_linguistic_stats_cleaned.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_parquet_folder(path):
    """Load Spark output folder (multiple parquet part files) into a single pandas DataFrame."""
    return pd.read_parquet(path)

def main():

    # Ensure output directory exists
    os.makedirs("data/plots", exist_ok=True)

    # ============================================================
    # Load all outputs from data/nlp_q4/
    # ============================================================
    df_top = load_parquet_folder("data/nlp_q4/top_words/")
    df_stats = load_parquet_folder("data/nlp_q4/linguistic_stats/")
    df_features = load_parquet_folder("data/nlp_q4/submission_features/")

    print("Loaded:")
    print(" - top_words:", df_top.shape)
    print(" - linguistic_stats:", df_stats.shape)
    print(" - submission_features:", df_features.shape)

    # ============================================================
    # Plot Top 20 Words (High vs Low engagement)
    # ============================================================
    top20_high = (
        df_top[df_top["engagement"]=="high"]
        .sort_values("count", ascending=False)
        .head(20)
    )

    top20_low = (
        df_top[df_top["engagement"]=="low"]
        .sort_values("count", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(14,6))

    # High engagement
    plt.subplot(1,2,1)
    sns.barplot(data=top20_high, x="count", y="word", color="steelblue")
    plt.title("Top 20 Words — High Engagement")

    # Low engagement
    plt.subplot(1,2,2)
    sns.barplot(data=top20_low, x="count", y="word", color="gray")
    plt.title("Top 20 Words — Low Engagement")

    plt.tight_layout()
    plot_path = "data/plots/nlp_problem4_top_words.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Saved] {plot_path}")

    # ============================================================
    # Clean & Save Stats Table
    # ============================================================
    cleaned = df_stats.copy()
    cleaned["avg_word_count"] = cleaned["avg_word_count"].round(2)
    cleaned["avg_unique_ratio"] = cleaned["avg_unique_ratio"].round(3)
    cleaned["avg_question_marks"] = cleaned["avg_question_marks"].round(3)

    stats_csv = "data/csv/nlp_problem4_linguistic_stats_cleaned.csv"
    cleaned.to_csv(stats_csv, index=False)
    print(f"[Saved] {stats_csv}")

    print("\n=== NLP Problem 4 Local Analysis Complete ===")

if __name__ == "__main__":
    main()

