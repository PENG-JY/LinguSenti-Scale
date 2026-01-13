#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EDA Problem 2: Post Length Distribution & Its Relationship with Engagement
"""

import sys
import os
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, coalesce, concat_ws, regexp_replace, split, size,
    when, count, avg, corr
)

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Spark Session ----------------
def create_spark_session(master_url: str):
    spark = (
        SparkSession.builder
        .appName("EDA_Problem2_PostLength_Cluster")
        .master(master_url)
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262"
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
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


# ---------------- EDA Core ----------------
def analyze_post_length(spark, submissions_path, output_dir):
    logger.info(f"Loading submissions from: {submissions_path}")

    df = (
        spark.read.parquet(submissions_path)
        .select("title", "selftext", "score", "num_comments")
    ).cache()

    n = df.count()
    logger.info(f"Rows in cleaned submissions: {n:,}")

    # post_length = word_count(title + selftext)
    clean_text = regexp_replace(
        concat_ws(" ", coalesce(col("title"), lit("")), coalesce(col("selftext"), lit(""))),
        r"[^A-Za-z0-9]+", " "
    )
    df2 = (
        df.withColumn("post_length", size(split(clean_text, r"\s+")))
          .withColumn("post_length", when(col("post_length") < 0, lit(0)).otherwise(col("post_length")))
          .withColumn("score", coalesce(col("score"), lit(0)))
          .withColumn("num_comments", coalesce(col("num_comments"), lit(0)))
    ).cache()

    # ---- Summary describe & percentiles
    desc_tbl = df2.select("post_length", "score", "num_comments").describe()
    p50, p90, p95, p99 = df2.approxQuantile("post_length", [0.5, 0.9, 0.95, 0.99], 0.01)
    summary_rows = [
        ("post_length_p50", p50),
        ("post_length_p90", p90),
        ("post_length_p95", p95),
        ("post_length_p99", p99),
    ]
    summary_df = spark.createDataFrame(summary_rows, ["metric", "value"])

    # ---- Table A: length bins
    df_binned = df2.withColumn(
        "length_bin",
        when(col("post_length") < 100, lit("short_<100"))
        .when((col("post_length") >= 100) & (col("post_length") <= 500), lit("medium_100_500"))
        .otherwise(lit("long_>500"))
    )
    bin_stats = (
        df_binned.groupBy("length_bin")
        .agg(
            count(lit(1)).alias("n_posts"),
            avg(col("num_comments")).alias("avg_num_comments"),
            avg(col("score")).alias("avg_score"),
            avg(col("post_length")).alias("avg_post_length")
        )
        .orderBy("length_bin")
    )

    # ---- Table B: correlation matrix
    corr_pl_score = df2.select(corr("post_length", "score").alias("corr")).first()["corr"]
    corr_pl_comments = df2.select(corr("post_length", "num_comments").alias("corr")).first()["corr"]
    corr_score_comments = df2.select(corr("score", "num_comments").alias("corr")).first()["corr"]
    corr_rows = [
        ("post_length", "score", float(corr_pl_score) if corr_pl_score is not None else None),
        ("post_length", "num_comments", float(corr_pl_comments) if corr_pl_comments is not None else None),
        ("score", "num_comments", float(corr_score_comments) if corr_score_comments is not None else None),
    ]
    corr_df = spark.createDataFrame(corr_rows, ["var_x", "var_y", "pearson_corr"])


    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ---- CSV outputs
    def _safe_to_csv(pdf, filepath):
        if os.path.isdir(filepath):
            shutil.rmtree(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pdf.to_csv(filepath, index=False)

    _safe_to_csv(bin_stats.toPandas(), os.path.join(output_dir, "eda_post_length_bin_stats.csv"))
    _safe_to_csv(corr_df.toPandas(), os.path.join(output_dir, "eda_post_length_corr.csv"))
    _safe_to_csv(summary_df.toPandas(), os.path.join(output_dir, "eda_post_length_summary.csv"))
    _safe_to_csv(desc_tbl.toPandas(), os.path.join(output_dir, "eda_post_length_describe.csv"))

    logger.info("CSV outputs written to data/csv/ (single-file).")

    # ---- Plots
    p95_len = df2.approxQuantile("post_length", [0.95], 0.001)[0]

    # Histogram (filter to x <= p95)
    pdf_len = df2.select("post_length").toPandas()
    pdf_len = pdf_len[pdf_len["post_length"] <= p95_len]

    plt.figure(figsize=(10, 5))
    plt.hist(pdf_len["post_length"], bins=40, edgecolor='black', alpha=0.75)
    plt.title("Distribution of Post Length (title + selftext) [x <= p95]")
    plt.xlabel("Post length (words)")
    plt.ylabel("Number of posts")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eda_post_length_hist.png"), dpi=300)
    plt.close()

    # Scatter
    pdf_scatter = (
        df2.select("post_length", "num_comments")
        .where(col("post_length") > 0)
        .toPandas()
    )
    pdf_scatter = pdf_scatter[pdf_scatter["post_length"] <= p95_len]

    bin_width = 25
    pdf_scatter["len_bin_idx"] = (pdf_scatter["post_length"] // bin_width).astype(int)
    binned = (pdf_scatter.groupby("len_bin_idx")
            .agg(post_len_avg=("post_length", "mean"),
                comments_avg=("num_comments", "mean"))
            .reset_index()
            .sort_values("post_len_avg"))

    plt.figure(figsize=(10, 5))
    plt.scatter(pdf_scatter["post_length"], pdf_scatter["num_comments"], s=5, alpha=0.15)
    plt.plot(binned["post_len_avg"], binned["comments_avg"], linewidth=2, label="Binned average", color='orange')
    plt.title("Post Length vs. Number of Comments [x<= p95]")
    plt.xlabel("Post length (words)")
    plt.ylabel("Number of comments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eda_post_length_vs_comments.png"), dpi=300)
    plt.close()

    logger.info("Plots saved under data/csv/plots/.")
    logger.info("EDA Problem 2 completed.")


# ---------------- Main ----------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python eda_problem2_post_length_engagement.py <spark_master_url>")
        sys.exit(1)

    master_url = sys.argv[1]
    spark = create_spark_session(master_url)

    submissions_path = "s3a://achai6000-reddit/reddit/submissions/"
    output_dir = "data/csv"

    analyze_post_length(spark, submissions_path, output_dir)
    spark.stop()