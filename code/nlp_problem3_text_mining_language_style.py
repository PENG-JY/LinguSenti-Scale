#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NLP Problem 3: Text Mining & Language Style
- Compute NRC emotion counts and densities per submission
- Add basic language-style features (length, question marks, readability)
- Join with num_comments from original submissions
- Write per-post features to S3 (parquet)
- Write summary CSV and plot locally for report/website:
    * Graph: emotional vocabulary density vs. num_comments
    * Table: [feature, mean, std, corr_with_num_comments]
"""

import os
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ====== CONFIG ======
NRC_PATH = "s3a://yw1150-dsan6000-project/reddit/nrc/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
CLEAN_SUB_PATH = "s3a://yw1150-dsan6000-project/reddit/clean_submissions/"
SUBMISSIONS_PATH = "s3a://yw1150-dsan6000-project/reddit/submissions/"
OUT_PATH_PARQUET = "s3a://yw1150-dsan6000-project/reddit/nlp/emotion_features/"

LOCAL_CSV_DIR = "data/csv"
LOCAL_PLOTS_DIR = os.path.join(LOCAL_CSV_DIR, "plots")

EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
    "negative",
    "positive",
]

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------- Spark Session ----------------
def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("NLP_Problem3_Text_Mining_Language_Style")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262"
        )
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain"
        )
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created successfully.")
    return spark


# ---------------- Core: emotion + style features ----------------
def compute_emotion_features(spark):
    # 1) Read cleaned submissions
    logger.info(f"Reading cleaned submissions from: {CLEAN_SUB_PATH}")
    clean_sub = spark.read.parquet(CLEAN_SUB_PATH)

    # Expect columns: id, raw_text, clean_text
    clean_sub = clean_sub.select("id", "raw_text", "clean_text")

    # 2) Read original submissions to get num_comments
    logger.info(f"Reading original submissions (for num_comments) from: {SUBMISSIONS_PATH}")
    sub_meta = (
        spark.read.parquet(SUBMISSIONS_PATH)
        .select("id", "num_comments")
    )

    # Join on id
    sub = (
        clean_sub.join(sub_meta, on="id", how="left")
        .withColumn("num_comments", F.coalesce(F.col("num_comments"), F.lit(0)))
    )

    # Basic length and question features on raw_text
    sub = sub.withColumn("post_length_chars", F.length(F.col("raw_text")))

    sub = sub.withColumn(
        "question_count",
        F.length(F.regexp_replace(F.col("raw_text"), "[^?]", "")),
    )

    # Approximate sentence count from ., !, ? and compute readability score later
    sub = sub.withColumn(
        "sentence_punct_count",
        F.length(F.regexp_replace(F.col("raw_text"), "[^\\.!?]", ""))
    )

    # Tokens from clean_text
    sub = sub.withColumn("tokens", F.split(F.col("clean_text"), "@"))
    sub = sub.withColumn("total_tokens", F.size("tokens"))

    # Use token count as a post length measure
    sub = sub.withColumn("post_length_tokens", F.col("total_tokens"))

    sub_tokens = sub.cache()
    logger.info("Submissions with tokens and basic style features prepared.")

    # 3) Read NRC lexicon
    logger.info(f"Reading NRC lexicon from: {NRC_PATH}")
    nrc_raw = (
        spark.read
        .option("sep", "\t")
        .option("header", "false")
        .csv(NRC_PATH)
        .toDF("word", "emotion", "value")
    )
    nrc = nrc_raw.filter(F.col("value") == "1").select("word", "emotion")

    # 4) Explode tokens and join with NRC
    logger.info("Exploding tokens and joining with NRC lexicon...")
    exploded = (
        sub_tokens
        .withColumn("token", F.explode("tokens"))
        .filter(F.col("token").isNotNull() & (F.col("token") != ""))
    )

    joined = (
        exploded.join(
            nrc,
            exploded.token == nrc.word,
            how="inner"
        )
        .select("id", "emotion")
    )

    logger.info("Counting tokens per (id, emotion)...")
    counts = (
        joined
        .groupBy("id", "emotion")
        .agg(F.count("*").alias("emotion_count"))
    )

    pivoted = (
        counts
        .groupBy("id")
        .pivot("emotion", EMOTIONS)
        .agg(F.first("emotion_count"))
    )

    # Fill missing emotion counts with 0
    for emo in EMOTIONS:
        pivoted = pivoted.withColumn(
            f"{emo}_count",
            F.coalesce(F.col(emo), F.lit(0).cast("long"))
        ).drop(emo)

    # 5) Merge and compute densities and emotional vocab density
    logger.info("Merging counts and computing densities and derived features...")

    base_cols = [
        "id",
        "num_comments",
        "total_tokens",
        "post_length_tokens",
        "post_length_chars",
        "question_count",
        "sentence_punct_count",
    ]

    result = sub_tokens.select(*base_cols).join(pivoted, on="id", how="left")

    result = result.withColumn(
        "total_tokens",
        F.when(F.col("total_tokens") == 0, F.lit(1)).otherwise(F.col("total_tokens"))
    )

    # Compute per-emotion density and total emotion tokens
    emotion_sum_expr = None
    for emo in EMOTIONS:
        count_col = f"{emo}_count"
        dens_col = f"{emo}_density"
        result = result.withColumn(
            dens_col,
            F.col(count_col) / F.col("total_tokens")
        )
        if emotion_sum_expr is None:
            emotion_sum_expr = F.col(count_col)
        else:
            emotion_sum_expr = emotion_sum_expr + F.col(count_col)

    # Total emotional tokens and emotional vocabulary density
    result = result.withColumn(
        "emotion_token_count",
        F.coalesce(emotion_sum_expr, F.lit(0).cast("long"))
    )
    result = result.withColumn(
        "emotional_vocab_density",
        F.col("emotion_token_count") / F.col("total_tokens")
    )

    # Readability score: average tokens per "sentence" (approx via punctuation)
    result = result.withColumn(
        "sentence_punct_count",
        F.when(F.col("sentence_punct_count") <= 0, F.lit(1)).otherwise(F.col("sentence_punct_count"))
    )
    result = result.withColumn(
        "readability_score",
        F.col("total_tokens") / F.col("sentence_punct_count")
    )

    logger.info(f"Writing per-post emotion features (parquet) to: {OUT_PATH_PARQUET}")
    (
        result
        .write
        .mode("overwrite")
        .parquet(OUT_PATH_PARQUET)
    )

    return result


# ---------------- Local CSV + Plots ----------------
def save_feature_table_and_plot(result_df):
    os.makedirs(LOCAL_CSV_DIR, exist_ok=True)
    os.makedirs(LOCAL_PLOTS_DIR, exist_ok=True)

    # Required features and their labels
    feature_map = {
        "post_length_tokens": "postlength",
        "emotional_vocab_density": "emotionalworddensity",
        "question_count": "questioncount",
        "readability_score": "readabilityscore",
    }

    rows = []
    logger.info("Computing mean, std, and correlation with num_comments for required features...")

    for col_name, label in feature_map.items():
        agg_row = (
            result_df
            .select(
                F.mean(col_name).alias("mean"),
                F.stddev_samp(col_name).alias("std"),
                F.corr(col_name, "num_comments").alias("corr")
            )
            .first()
        )
        rows.append(
            {
                "feature": label,
                "mean": float(agg_row["mean"]) if agg_row["mean"] is not None else None,
                "std": float(agg_row["std"]) if agg_row["std"] is not None else None,
                "corr_with_num_comments": float(agg_row["corr"]) if agg_row["corr"] is not None else None,
            }
        )

    stats_df = pd.DataFrame(rows, columns=["feature", "mean", "std", "corr_with_num_comments"])
    stats_csv_path = os.path.join(LOCAL_CSV_DIR, "nlp_problem3_feature_stats.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    logger.info(f"Feature stats table written to: {stats_csv_path}")

    # Plot: emotional vocabulary density vs num_comments
    logger.info("Preparing data for emotional_vocab_density vs num_comments plot...")

    pair_df = (
        result_df
        .select("emotional_vocab_density", "num_comments")
        .where(
            (F.col("emotional_vocab_density").isNotNull()) &
            (F.col("num_comments").isNotNull()) &
            (F.col("num_comments") >= 0)
        )
        .sample(False, 0.05, seed=42)  # sample for plotting
        .toPandas()
    )

    if not pair_df.empty:
        # Clip to reasonable ranges
        p99_density = pair_df["emotional_vocab_density"].quantile(0.99)
        p99_comments = pair_df["num_comments"].quantile(0.99)

        pair_df = pair_df[
            (pair_df["emotional_vocab_density"] <= p99_density) &
            (pair_df["num_comments"] <= p99_comments)
        ]

        # Hard cap density at 1
        pair_df = pair_df[pair_df["emotional_vocab_density"] <= 1]

        if not pair_df.empty:
            # Binning for smoothed trend
            bin_width = 0.02
            pair_df["density_bin"] = (pair_df["emotional_vocab_density"] / bin_width).astype(int)

            binned = (
                pair_df.groupby("density_bin")
                .agg(
                    density_avg=("emotional_vocab_density", "mean"),
                    comments_avg=("num_comments", "mean"),
                )
                .reset_index()
                .sort_values("density_avg")
            )

            plt.figure(figsize=(8, 5))
            plt.scatter(
                pair_df["emotional_vocab_density"],
                pair_df["num_comments"],
                s=5,
                alpha=0.25,
            )
            if not binned.empty:
                plt.plot(
                    binned["density_avg"],
                    binned["comments_avg"],
                    linewidth=2.0,
                    label="Binned mean",
                )
                plt.legend()

            plt.xlabel("Emotional vocabulary density")
            plt.ylabel("Number of comments")
            plt.title("Emotional Vocabulary Density vs. Number of Comments")
            plt.tight_layout()

            plot_path = os.path.join(
                LOCAL_PLOTS_DIR,
                "nlp_problem3_emotional_vocab_density_vs_num_comments.png"
            )
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logger.info(f"Plot saved to: {plot_path}")
        else:
            logger.warning("All points removed by clipping; no plot generated.")
    else:
        logger.warning("No data available for plotting emotional_vocab_density vs num_comments.")


# ---------------- Main ----------------
if __name__ == "__main__":
    spark = create_spark_session()

    try:
        result_df = compute_emotion_features(spark)
        save_feature_table_and_plot(result_df)
        logger.info("NLP Problem 3 completed successfully.")
    finally:
        spark.stop()