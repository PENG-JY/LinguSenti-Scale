"""
Goal:
    - Load trained LR model comments_lr_model_full/
    - Load ALL comments parquet from S3
    - Apply same feature engineering
    - Predict controversiality probability
    - Save top 500 most controversial comments
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    length, regexp_extract, col, hour, from_unixtime
)
from pyspark.ml.functions import vector_to_array
from pyspark.ml import PipelineModel
import pandas as pd


def main():
    spark = SparkSession.builder.appName("CommentsMLPredict").getOrCreate()

    # -------------------------
    # 1. Load FULL dataset
    # -------------------------
    base = "s3a://jc3482-dsan6000-datasets/project/reddit/parquet/comments/"

    print("Loading ALL comment parquet files...\n")
    df = spark.read.parquet(base)
    print(f"Total rows = {df.count():,}")

    # -------------------------
    # 2. Feature Engineering (same as training!)
    # -------------------------
    df2 = (
        df
        .withColumn("body_length", length(col("body")))
        .withColumn("punctuation_count", (regexp_extract(col("body"), r"[!?]", 0) != "").cast("int"))
        .withColumn("num_words", length(regexp_extract(col("body"), r"(\\S+)", 0)))
        .withColumn("is_top_level", col("parent_id").startswith("t3_").cast("int"))
        .withColumn("hour", hour(from_unixtime(col("created_utc"))))
        .withColumn("label", col("controversiality").cast("int"))
    )

    # -------------------------
    # 3. Load trained model
    # -------------------------
    print("Loading trained model from comments_lr_model_full/")
    model = PipelineModel.load("comments_lr_model_full")

    # -------------------------
    # 4. Predict controversial probability
    # -------------------------
    print("Running prediction on FULL dataset...")
    pred = model.transform(df2)

    # Extract probability of class 1 from SparseVector
    pred = pred.withColumn(
        "prob_controversial",
        vector_to_array(col("probability"))[1]
    )

    # -------------------------
    # 5. Select top 500 most controversial
    # -------------------------
    top = (
        pred
        .select("id", "body", "score", "parent_id", "is_top_level", "prob_controversial")
        .orderBy(col("prob_controversial").desc())
        .limit(500)
        .toPandas()
    )

    # -------------------------
    # 6. Save results
    # -------------------------
    output_csv = "top_controversial_comments.csv"
    top.to_csv(output_csv, index=False)

    print(f"\nSaved top 500 controversial comments → {output_csv}")
    print(top.head())

   


    spark.stop()
    print("\nDONE!")


if __name__ == "__main__":
    main()
