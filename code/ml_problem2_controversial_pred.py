"""
comments_ml_test.py 

Goal:
    - Load ALL comment parquet files from S3 (3677 files, total ~1.6 GB)
    - Do feature engineering (length, punctuation, hour, top-level, etc.)
    - Train Logistic Regression model to predict controversiality
    - Output feature importance table and a visualization plot
    - Save importance CSV & PNG locally
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    length, regexp_extract, col, hour, from_unixtime
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

import pandas as pd
import matplotlib.pyplot as plt


def main():
    spark = SparkSession.builder.appName("CommentsMLFull").getOrCreate()

    # -------------------------
    # 1. Load ALL parquet files from S3
    # -------------------------
    base_path = "s3a://jc3482-dsan6000-datasets/project/reddit/parquet/comments/"

    print("Loading ALL Parquet files from S3 (~1.6 GB)...\n")
    df = spark.read.parquet(base_path)
    print("Schema:")
    df.printSchema()
    print(f"Total rows = {df.count():,}")

    # -------------------------
    # 2. Feature Engineering
    # -------------------------
    df2 = (
        df
        .withColumn("body_length", length(col("body")))
        .withColumn("punctuation_count", (regexp_extract(col("body"), r"[!?]", 0) != "").cast("int"))
        .withColumn("num_words", length(regexp_extract(col("body"), r"(\S+)", 0)))
        .withColumn("is_top_level", col("parent_id").startswith("t3_").cast("int"))
        .withColumn("hour", hour(from_unixtime(col("created_utc"))))
        .withColumn("label", col("controversiality").cast("int"))
    )

    feature_cols = ["score", "body_length", "punctuation_count", "num_words",
                    "is_top_level", "gilded", "hour"]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[assembler, lr])

    train, test = df2.randomSplit([0.8, 0.2], seed=42)

    print("\nTraining Logistic Regression model on FULL dataset...")
    model = pipeline.fit(train)
    pred = model.transform(test)

    # -------------------------
    # 3. Evaluation
    # -------------------------
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc = evaluator.evaluate(pred)
    print("\nAUC =", auc)

    # -------------------------
    # 4. Extract Feature Importance
    # -------------------------
    lr_model = model.stages[-1]
    coeffs = lr_model.coefficients.toArray()

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coeffs
    })

    importance_df["abs_coef"] = importance_df["coefficient"].abs()
    importance_df = importance_df.sort_values("abs_coef", ascending=False)

    importance_df.to_csv("feature_importance_comments_full.csv", index=False)
    print("\nSaved feature importance to feature_importance_comments_full.csv")

    # -------------------------
    # 5. Visualization
    # -------------------------
    plt.figure(figsize=(10, 5))
    plt.barh(importance_df["feature"], importance_df["abs_coef"])
    plt.xlabel("Absolute Coefficient")
    plt.title("Feature Importance for Predicting Controversiality (FULL DATA)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance_comments_full.png")
    print("Saved plot to feature_importance_comments_full.png")

    spark.stop()
    print("\nDONE!")


if __name__ == "__main__":
    main()
