#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Problem 4 — Predictive Linguistic Signals (Spark-only version)

This script does:
1. Load cleaned submissions from S3
2. Compute linguistic features
3. Compute high vs low engagement groups
4. Extract top words
5. Compute aggregated linguistic statistics
6. Convert Spark results → pandas → CSV
7. Upload CSV back to S3 (results/)
(no plotting inside spark — plot is done locally)

Output on S3:
    s3://kl1160-dsan6000-datasets/q4/results/nlp_problem4_top_words.csv
    s3://kl1160-dsan6000-datasets/q4/results/nlp_problem4_linguistic_stats.csv
    s3://kl1160-dsan6000-datasets/q4/results/nlp_problem4_submission_features.csv
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window


# --------------------------------------------------------
# Spark Session
# --------------------------------------------------------
def create_spark():
    spark = (
        SparkSession.builder
        .appName("Q4_LinguisticSignals_SparkOnly_Corrected")
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
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    spark = create_spark()

    input_path = "s3a://kl1160-dsan6000-datasets/clean/clean_submissions/"
    output_root = "s3a://kl1160-dsan6000-datasets/q4/"

    # ============================================================
    # 1. Load Cleaned Data
    # ============================================================
    df = spark.read.parquet(input_path).select(
        "id", "subreddit", "title", "clean_text", "num_comments"
    )

    # ============================================================
    # 2. Engagement Split (Median)
    # ============================================================
    median_nc = df.approxQuantile("num_comments", [0.5], 0.01)[0]

    df = df.withColumn(
        "engagement",
        F.when(F.col("num_comments") >= median_nc, "high").otherwise("low")
    )

    # ============================================================
    # 3. Tokenization
    # ============================================================
    tokenize = F.udf(lambda s: s.split("@") if s else [], ArrayType(StringType()))
    df = df.withColumn("tokens", tokenize("clean_text"))

    df = (
        df
        .withColumn("word_count", F.size("tokens"))
        .withColumn("unique_words", F.size(F.array_distinct("tokens")))
        .withColumn("unique_ratio", F.col("unique_words") / F.col("word_count"))
        .withColumn("question_marks", F.size(F.split("title", r"\?")) - 1)
    )

    # ============================================================
    # 4. Top Words
    # ============================================================
    exploded = df.select("engagement", F.explode("tokens").alias("word"))
    word_counts = exploded.groupBy("engagement", "word").count()

    w = Window.partitionBy("engagement").orderBy(F.desc("count"))
    top_words = word_counts.withColumn("rank", F.row_number().over(w)).filter("rank <= 50")

    # ============================================================
    # 5. Aggregated Stats
    # ============================================================
    ling_stats = (
        df.groupBy("engagement")
        .agg(
            F.avg("word_count").alias("avg_word_count"),
            F.avg("unique_ratio").alias("avg_unique_ratio"),
            F.avg("question_marks").alias("avg_question_marks"),
        )
    )

    # ============================================================
    # 6. Write Results to S3
    # ============================================================
    top_words.write.mode("overwrite").parquet(output_root + "top_words/")
    ling_stats.write.mode("overwrite").parquet(output_root + "linguistic_stats/")
    df.write.mode("overwrite").parquet(output_root + "submission_features/")

    print("\n=== Q4 computation complete! Parquet written to S3 ===")

    spark.stop()


if __name__ == "__main__":
    main()

