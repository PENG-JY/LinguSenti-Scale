#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EDA Problem 1: Time Pattern of Posting & Commenting Activities
Run: uv run python eda_problem1_time_pattern.py spark://<master-ip>:7077
"""

import sys
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, month, dayofweek, hour, to_timestamp
)

# ---------------- Logging Configuration ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Spark Session ----------------
def create_spark_session(master_url: str):
    spark = (
        SparkSession.builder
        .appName("EDA_Problem1_TimePattern_Cluster")
        .master(master_url)
        # Resource configuration
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")
        # Hadoop AWS config (for S3A)
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262"
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
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


# ---------------- EDA Functions ----------------
def analyze_time_patterns(spark, comments_path, submissions_path, output_dir):
    logger.info("Loading Reddit comments and submissions data from Parquet...")

    # Load from Parquet instead of JSON
    df_comments = spark.read.parquet(comments_path)
    df_submissions = spark.read.parquet(submissions_path)

    logger.info("Data loaded successfully.")
    logger.info(f"Comments count: {df_comments.count():,}")
    logger.info(f"Submissions count: {df_submissions.count():,}")

    # Convert UNIX timestamp to timestamp (if numeric)
    logger.info("Converting created_utc to timestamp...")
    df_comments = df_comments.withColumn("created_time", to_timestamp(col("created_utc")))
    df_submissions = df_submissions.withColumn("created_time", to_timestamp(col("created_utc")))

    # Extract time features
    df_comments = (
        df_comments.withColumn("month", month(col("created_time")))
                   .withColumn("day_of_week", dayofweek(col("created_time")))
                   .withColumn("hour_of_day", hour(col("created_time")))
    )

    df_submissions = (
        df_submissions.withColumn("month", month(col("created_time")))
                      .withColumn("day_of_week", dayofweek(col("created_time")))
                      .withColumn("hour_of_day", hour(col("created_time")))
    )

    # -------- Monthly Post Trend --------
    logger.info("Aggregating monthly posting trend...")
    monthly_posts = (
        df_submissions.groupBy("month")
        .agg(count("*").alias("post_count"))
        .orderBy("month")
    )
    monthly_df = monthly_posts.toPandas()
    monthly_df.to_csv(os.path.join(output_dir, "eda_time_monthly_trend.csv"), index=False)

    # -------- Hour × Weekday Heatmap --------
    logger.info("Computing hour × weekday activity pattern...")
    heatmap_data = (
        df_submissions.groupBy("day_of_week", "hour_of_day")
        .agg(count("*").alias("post_count"))
        .orderBy("day_of_week", "hour_of_day")
    )
    heatmap_df = heatmap_data.toPandas()
    heatmap_df.to_csv(os.path.join(output_dir, "eda_time_heatmap_hour_weekday.csv"), index=False)

    # -------- Weekday Summary Table --------
    logger.info("Summarizing average post & comment counts by weekday...")
    summary_df = (
        df_submissions.groupBy("day_of_week")
        .agg(count("*").alias("avg_posts"))
        .join(
            df_comments.groupBy("day_of_week").agg(count("*").alias("avg_comments")),
            on="day_of_week",
            how="outer"
        )
        .orderBy("day_of_week")
    ).toPandas()

    summary_df.to_csv(os.path.join(output_dir, "eda_time_summary_by_weekday.csv"), index=False)

    # -------- Visualizations --------
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Line Chart – Monthly Posts
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_df["month"], monthly_df["post_count"], marker="o")
    plt.title("Posts per Month (2023–2024)")
    plt.xlabel("Month")
    plt.ylabel("Post Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots/eda_time_monthly_trend.png"), dpi=300)
    plt.close()

    # Heatmap – Hour × Weekday
    logger.info("Creating heatmap visualization...")
    pivot_table = heatmap_df.pivot(index="day_of_week", columns="hour_of_day", values="post_count")
    plt.figure(figsize=(12, 6))
    plt.imshow(pivot_table, cmap="coolwarm", aspect="auto", origin="lower")
    plt.colorbar(label="Post Count")
    plt.title("Posting Activity by Hour × Weekday")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots/eda_time_heatmap_hour_weekday.png"), dpi=300)
    plt.close()

    logger.info("EDA completed and plots saved.")


# ---------------- Main Entry ----------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eda_problem1_time_pattern.py <spark_master_url>")
        sys.exit(1)

    master_url = sys.argv[1]
    spark = create_spark_session(master_url)

    # Update to your S3 Parquet directories
    comments_path = "s3://jp2132-spark-reddit/reddit/comments/"
    submissions_path = "s3://jp2132-spark-reddit/reddit/submissions/"
    output_dir = "data/csv"

    os.makedirs(output_dir, exist_ok=True)
    analyze_time_patterns(spark, comments_path, submissions_path, output_dir)
    spark.stop()
