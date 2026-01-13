#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NLP Q1 — Temporal Emotion Patterns (NRC-only)
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
CLEAN_COMMENTS_PATH = "s3a://jp2132-spark-reddit/reddit/clean/clean_comments*/"

NRC_PATH = "s3a://jp2132-spark-reddit/reddit/nrc/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

OUT_PARQUET = "s3a://jp2132-spark-reddit/reddit/nlp/q1_temporal_nrc/"

LOCAL_OUT = "data/q1_nrc"
PLOT_DIR = os.path.join(LOCAL_OUT, "plots")

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NLP-Q1-NRC")


# ===========================================================================
# Spark Session
# ===========================================================================
def create_spark():
    spark = (
        SparkSession.builder
        .appName("NLP Q1 NRC Temporal Emotion Patterns")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262"
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ===========================================================================
# Core: NRC × Temporal
# ===========================================================================
def compute_nrc_temporal(spark):

    # -------- 1) Read clean comments --------
    logger.info("Reading clean comments...")
    base_df = spark.read.parquet(CLEAN_COMMENTS_PATH).select(
        "id", "clean_text", "created_utc",
        "score", "controversiality"
    )

    # Convert created_utc (BIGINT) → timestamp
    base_df = base_df.withColumn("created_ts", F.to_timestamp(F.col("created_utc")))

    # -------- 2) Tokenize --------
    df = base_df.withColumn("token", F.explode(F.split("clean_text", "@")))
    df = df.filter(F.length("token") > 0)

    # -------- 3) Read NRC lexicon --------
    logger.info("Reading NRC lexicon...")
    nrc_raw = (
        spark.read.option("sep", "\t").csv(NRC_PATH)
        .toDF("word", "emotion", "value")
    )
    nrc = nrc_raw.filter(F.col("value") == "1").select("word", "emotion")

    # -------- 4) Join token with NRC --------
    logger.info("Joining tokens with NRC...")
    joined = (
        df.join(nrc, df.token == nrc.word, "inner")
        .select(df.id, "emotion")
    )

    # Count tokens per emotion
    counts = (
        joined.groupBy("id", "emotion")
        .agg(F.count("*").alias("emotion_count"))
    )

    # Pivot to wide format
    pivoted = (
        counts.groupBy("id")
        .pivot("emotion", EMOTIONS)
        .agg(F.first("emotion_count"))
    )

    # Fill NA and rename columns
    for emo in EMOTIONS:
        pivoted = pivoted.withColumn(f"{emo}_count", F.coalesce(F.col(emo), F.lit(0))).drop(emo)

    # -------- 5) Merge back to original comments --------
    result = base_df.join(pivoted, on="id", how="left")

    # Compute total tokens
    token_total_expr = None
    for emo in EMOTIONS:
        c = F.col(f"{emo}_count")
        token_total_expr = c if token_total_expr is None else (token_total_expr + c)

    result = result.withColumn("emotion_token_count", F.coalesce(token_total_expr, F.lit(0)))
    result = result.withColumn(
        "total_tokens",
        F.when(F.col("emotion_token_count") == 0, 1)
         .otherwise(F.col("emotion_token_count"))
    )

    # density
    for emo in EMOTIONS:
        result = result.withColumn(
            f"{emo}_density",
            F.col(f"{emo}_count") / F.col("total_tokens")
        )

    # -------- 6) Temporal features (FIXED) --------
    result = result.withColumn("hour", F.hour("created_ts"))
    result = result.withColumn("month", F.month("created_ts"))

    # ---- 🔥 FIX: use Spark’s built-in weekday number ----
    # dayofweek(): Sunday=1 ... Saturday=7   (safe for sorting)
    result = result.withColumn("weekday_num", F.dayofweek("created_ts"))

    # Weekday short name (Mon, Tue, ...)
    result = result.withColumn("weekday", F.date_format("created_ts", "E"))

    # -------- 7) Write per-comment parquet --------
    logger.info(f"Writing per-comment NRC emotion features to {OUT_PARQUET}")
    result.write.mode("overwrite").parquet(OUT_PARQUET)

    return result


# ===========================================================================
# Visualization + CSV
# ===========================================================================
def save_summary_and_plots(df):
    os.makedirs(LOCAL_OUT, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ---------------- Hour × emotion ----------------
    hour_stats = (
        df.groupBy("hour")
        .agg(*[F.avg(f"{emo}_density").alias(f"{emo}_avg") for emo in EMOTIONS])
        .orderBy("hour")
        .toPandas()
    )

    hour_stats.to_csv(os.path.join(LOCAL_OUT, "hourly_emotion_density.csv"), index=False)

    # ---------------- Month × emotion ----------------
    month_stats = (
        df.groupBy("month")
        .agg(*[F.avg(f"{emo}_density").alias(f"{emo}_avg") for emo in EMOTIONS])
        .orderBy("month")
        .toPandas()
    )

    month_stats.to_csv(os.path.join(LOCAL_OUT, "monthly_emotion_density.csv"), index=False)

    # ---------------- Weekday × emotion ----------------
    weekday_stats = (
        df.groupBy("weekday_num", "weekday")
        .agg(*[F.avg(f"{emo}_density").alias(f"{emo}_avg") for emo in EMOTIONS])
        .orderBy("weekday_num")
        .toPandas()
    )
    weekday_stats.to_csv(os.path.join(LOCAL_OUT, "weekday_emotion_density.csv"), index=False)

    # ---------------- Histogram  ----------------
    sample_pdf = df.select("anger_density", "joy_density").sample(False, 0.02).toPandas()

    plt.hist(sample_pdf["anger_density"], bins=50, alpha=0.6, label="anger")
    plt.hist(sample_pdf["joy_density"], bins=50, alpha=0.6, label="joy")
    plt.legend()
    plt.title("Distribution of NRC Emotion Density")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "nlp_problem1_emotion_hist.png"))
    plt.close()

    # ---------------- Hourly Emotion Trend ----------------
    hour_pdf = (
        df.groupBy("hour")
        .agg(*[F.avg(f"{emo}_density").alias(f"{emo}_avg") for emo in EMOTIONS])
        .orderBy("hour")
        .toPandas()
    )

    plt.figure(figsize=(12, 6))
    for emo in EMOTIONS:
        plt.plot(hour_pdf["hour"], hour_pdf[f"{emo}_avg"], label=emo)

    plt.title("Emotion Density by Hour")
    plt.xlabel("Hour (0–23)")
    plt.ylabel("Average Emotion Density")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "nlp_problem1_hourly_emotion_trend.png"))
    plt.close()

    # ---------------- Monthly Emotion Trend ----------------
    month_pdf = (
        df.groupBy("month")
        .agg(*[F.avg(f"{emo}_density").alias(f"{emo}_avg") for emo in EMOTIONS])
        .orderBy("month")
        .toPandas()
    )

    plt.figure(figsize=(12, 6))
    for emo in EMOTIONS:
        plt.plot(month_pdf["month"], month_pdf[f"{emo}_avg"], label=emo)

    plt.title("Emotion Density by Month")
    plt.xlabel("Month (1–12)")
    plt.ylabel("Average Emotion Density")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "nlp_problem1_monthly_emotion_trend.png"))
    plt.close()

    # ---------------- Weekday Emotion Trend ----------------
    weekday_pdf = (
        df.groupBy("weekday_num", "weekday")
        .agg(*[F.avg(f"{emo}_density").alias(f"{emo}_avg") for emo in EMOTIONS])
        .orderBy("weekday_num")
        .toPandas()
    )

    plt.figure(figsize=(12, 6))
    for emo in EMOTIONS:
        plt.plot(weekday_pdf["weekday"], weekday_pdf[f"{emo}_avg"], marker='o', label=emo)

    plt.title("Emotion Density by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Average Emotion Density")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "nlp_problem1_weekday_emotion_trend.png"))
    plt.close()


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    spark = create_spark()
    try:
        df = compute_nrc_temporal(spark)
        save_summary_and_plots(df)
        logger.info("NLP Q1 NRC Completed.")
    finally:
        spark.stop()
