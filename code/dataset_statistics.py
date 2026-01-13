#!/usr/bin/env python3
"""
Dataset Statistics
Generates summary CSVs and performs basic data quality checks
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, min, max, length, to_date, from_unixtime,
    year, month, concat_ws, lit, when
)


# 1. Spark session
spark = (
    SparkSession.builder
    .appName("Dataset_Statistics")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)

NET_ID = "kl1160"
BASE_PATH = f"s3a://{NET_ID}-dsan6000-datasets/project/reddit/parquet"


# 2. Load filtered data
comments = spark.read.parquet(f"{BASE_PATH}/comments/")
submissions = spark.read.parquet(f"{BASE_PATH}/submissions/")

# Ensure timestamps - date
comments = comments.withColumn("date", to_date(from_unixtime(col("created_utc"))))
submissions = submissions.withColumn("date", to_date(from_unixtime(col("created_utc"))))


# 3. Dataset Summary
summary_df = spark.createDataFrame([
    ("comments",
     comments.count(),
     round(comments.rdd.map(lambda r: len(str(r))).count() * 200 / (1024 ** 3), 2),
     comments.agg(min("date")).first()[0],
     comments.agg(max("date")).first()[0]),
    ("submissions",
     submissions.count(),
     round(submissions.rdd.map(lambda r: len(str(r))).count() * 200 / (1024 ** 3), 2),
     submissions.agg(min("date")).first()[0],
     submissions.agg(max("date")).first()[0]),
], ["data_type", "total_rows", "size_gb", "date_range_start", "date_range_end"])

summary_df.show(truncate=False)


# 4. Subreddit-level Statistics
comments_stats = (
    comments.groupBy("subreddit")
    .agg(count("*").alias("num_comments"), avg("score").alias("avg_comment_score"),
         min("date").alias("start_date"), max("date").alias("end_date"))
)

submissions_stats = (
    submissions.groupBy("subreddit")
    .agg(count("*").alias("num_submissions"), avg("score").alias("avg_submission_score"),
         min("date").alias("start_date2"), max("date").alias("end_date2"))
)

subreddit_stats = (
    comments_stats.join(submissions_stats, on="subreddit", how="outer")
    .fillna(0)
    .withColumn("total_rows", col("num_comments") + col("num_submissions"))
    .withColumn("avg_score",
                when(col("num_comments") + col("num_submissions") > 0,
                     (col("avg_comment_score") * col("num_comments") +
                      col("avg_submission_score") * col("num_submissions")) /
                     (col("num_comments") + col("num_submissions")))
                .otherwise(0))
    .withColumn("date_range",
                concat_ws(" ~ ",
                          when(col("start_date") < col("start_date2"), col("start_date")).otherwise(col("start_date2")),
                          when(col("end_date") > col("end_date2"), col("end_date")).otherwise(col("end_date2"))))
    .select("subreddit", "num_comments", "num_submissions", "total_rows", "avg_score", "date_range")
)

subreddit_stats.show(truncate=False)


# 5. Temporal Distribution
comments_time = comments.withColumn("year_month", concat_ws("-", year("date"), month("date")))\
                        .groupBy("year_month").agg(count("*").alias("num_comments"))

submissions_time = submissions.withColumn("year_month", concat_ws("-", year("date"), month("date")))\
                              .groupBy("year_month").agg(count("*").alias("num_submissions"))

temporal = comments_time.join(submissions_time, on="year_month", how="outer").fillna(0)
temporal = temporal.withColumn("total_rows", col("num_comments") + col("num_submissions"))
temporal.show(10, truncate=False)


# 6. Data Quality Checks
print("\nData Quality Checks:")
comments_quality = comments.select(
    count(when(col("body").isNull(), 1)).alias("missing_body"),
    count(when(length("body") < 3, 1)).alias("too_short"),
    count(when(length("body") > 5000, 1)).alias("too_long")
)
comments_quality.show()

submissions_quality = submissions.select(
    count(when(col("title").isNull(), 1)).alias("missing_title"),
    count(when(length("title") < 3, 1)).alias("too_short"),
    count(when(length("title") > 500, 1)).alias("too_long")
)
submissions_quality.show()


# 7. Save Results as CSV
output_dir = "data/csv"
os.makedirs(output_dir, exist_ok=True)

summary_df.coalesce(1).write.mode("overwrite").csv(f"{output_dir}/dataset_summary.csv", header=True)
subreddit_stats.coalesce(1).write.mode("overwrite").csv(f"{output_dir}/subreddit_statistics.csv", header=True)
temporal.coalesce(1).write.mode("overwrite").csv(f"{output_dir}/temporal_distribution.csv", header=True)

print("\n✅ All summary CSVs saved successfully in:", output_dir)
spark.stop()
