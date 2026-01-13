#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML Task: Predict high-engagement Reddit relationship posts
Light-weight, stable version (CountVectorizer + Numeric + Comment Features)
Using Logistic Regression with class weights for imbalanced data.

Main steps:
1) Load cleaned submissions (parquet) and basic text column `clean_text`
2) Load cleaned comments (parquet), aggregate lightweight comment-level features per submission
3) Join submissions with aggregated comment features
4) Create additional numeric features from text and time (title / body / clean_text)
5) Define binary label: top 25% by num_comments vs the rest
6) Compute class weights and use them in Logistic Regression (no explicit oversampling)
7) Build Spark ML pipeline:
       - RegexTokenizer + StopWordsRemover
       - CountVectorizer (bag-of-words, sparse)
       - VectorAssembler over numeric + bow features
       - StandardScaler
       - LogisticRegression (with weightCol)
8) Hyperparameter tuning via CrossValidator (regParam, elasticNetParam)
9) Evaluate using default prediction (threshold=0.5) but also expose positive probability
10) Save metrics, sample predictions, and best model
"""

import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType

# Spark ML
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    VectorAssembler,
    StandardScaler,
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.functions import vector_to_array


# ===========================
# Column / feature names
# ===========================

TEXT_COL = "clean_text"
NUM_COMMENTS_COL = "num_comments"
SUBMISSION_ID = "id"  # submission ID in submissions dataset

# Simple text / time features (on submissions)
TOKEN_COUNT_COL = "token_count"
SENT_COUNT_COL = "sentence_count"
HAS_QMARK_COL = "has_question_mark"
HOUR_COL = "post_hour"
DOW_COL = "post_dow"
IS_WEEKEND_COL = "is_weekend"

# Extra lightweight NLP / structural features
CHAR_LEN_COL = "char_len_clean"
TITLE_TOKEN_COUNT_COL = "title_token_count"
TITLE_CHAR_LEN_COL = "title_char_len"
TITLE_HAS_QMARK_COL = "title_has_question_mark"
SELF_TOKEN_COUNT_COL = "selftext_token_count"
SELF_CHAR_LEN_COL = "selftext_char_len"

# Label + feature columns
LABEL_COL = "label_high_engagement"
FEATURES_COL = "features"
SCALED_FEATURES_COL = "features_scaled"
WEIGHT_COL = "class_weight"

# Quantile for defining "high engagement"
# ENG_Q = 0.75 → high engagement = top 25% by num_comments
ENG_Q = 0.75


# ===========================
# Spark session
# ===========================

def create_spark():
    """
    Create a SparkSession with conservative settings to avoid OOM on Parquet.
    We explicitly disable the vectorized Parquet reader (uses large contiguous
    columnar batches) which previously caused heap issues.
    """
    spark = (
        SparkSession.builder
        .appName("ML_High_Engagement_Light_LogReg")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ===========================
# Label creation
# ===========================

def add_label_column(df):
    """
    Add binary label column LABEL_COL using a quantile threshold on num_comments.
    label = 1.0 if num_comments >= q, else 0.0
    """
    q = df.approxQuantile(NUM_COMMENTS_COL, [ENG_Q], 0.01)[0]
    print(f"[INFO] Engagement threshold (top {ENG_Q * 100}%): {q}")

    return df.withColumn(
        LABEL_COL,
        F.when(F.col(NUM_COMMENTS_COL) >= q, 1.0).otherwise(0.0).cast(DoubleType()),
    )


# ===========================
# Comment aggregation
# ===========================

def load_comment_features(spark, comments_path):
    """
    Load comment-level data and aggregate lightweight features per submission.

    Expected columns in clean_comments parquet:
      - link_id: submission ID key used to join back to submissions
      - score
      - controversiality
      - clean_text
      - author
      - created_utc, etc.

    We compute:
      - comment_score_mean / max / sum
      - comment_len_mean / max / sum (length of clean_text)
      - comment_author_nunique
      - comment_is_op_ratio (here set to 0 because no is_op field exists)
      - comment_contro_mean / max / sum
    """
    print("[INFO] Loading comment data...")

    dfc = spark.read.parquet(comments_path)

    # Text length of comment clean_text
    dfc = dfc.withColumn("length", F.length(F.col("clean_text")))

    # We don't have an `is_op` flag in the schema → set constant 0
    dfc = dfc.withColumn("is_op", F.lit(0))

    agg = (
        dfc.groupBy("link_id")
        .agg(
            F.mean("score").alias("comment_score_mean"),
            F.max("score").alias("comment_score_max"),
            F.sum("score").alias("comment_score_sum"),
            F.mean("length").alias("comment_len_mean"),
            F.max("length").alias("comment_len_max"),
            F.sum("length").alias("comment_len_sum"),
            F.countDistinct("author").alias("comment_author_nunique"),
            F.mean(F.col("is_op").cast("double")).alias("comment_is_op_ratio"),
            F.mean("controversiality").alias("comment_contro_mean"),
            F.max("controversiality").alias("comment_contro_max"),
            F.sum("controversiality").alias("comment_contro_sum"),
        )
    )

    print("[INFO] Comment features aggregated.")
    return agg


# ===========================
# Feature engineering (submissions)
# ===========================

def enrich_features(df):
    """
    Add lightweight text and time-based features on submissions:

      On `clean_text`:
        - token_count: rough word count by whitespace split
        - sentence_count: number of '.' separated segments
        - has_question_mark: whether text contains '?'
        - char_len_clean: character length

      On `title`:
        - title_token_count: number of whitespace-separated tokens
        - title_char_len: character length
        - title_has_question_mark: whether title contains '?'

      On `selftext`:
        - selftext_token_count: number of whitespace-separated tokens
        - selftext_char_len: character length

      Time features:
        - post_hour: hour of day from created_utc
        - post_dow: day-of-week
        - is_weekend: 1 if weekend else 0
    """
    # Safely coalesce possibly-null text columns to empty strings
    clean = F.coalesce(F.col(TEXT_COL), F.lit(""))
    title = F.coalesce(F.col("title"), F.lit(""))
    selftext = F.coalesce(F.col("selftext"), F.lit(""))

    # clean_text based features
    df = df.withColumn(TOKEN_COUNT_COL, F.size(F.split(clean, r"\s+")))
    df = df.withColumn(SENT_COUNT_COL, F.size(F.split(clean, r"\.")))
    df = df.withColumn(
        HAS_QMARK_COL,
        F.when(clean.contains("?"), F.lit(1)).otherwise(F.lit(0)),
    )
    df = df.withColumn(CHAR_LEN_COL, F.length(clean))

    # title based features
    df = df.withColumn(
        TITLE_TOKEN_COUNT_COL,
        F.size(F.split(title, r"\s+")),
    )
    df = df.withColumn(
        TITLE_CHAR_LEN_COL,
        F.length(title),
    )
    df = df.withColumn(
        TITLE_HAS_QMARK_COL,
        F.when(title.contains("?"), F.lit(1)).otherwise(F.lit(0)),
    )

    # selftext based features
    df = df.withColumn(
        SELF_TOKEN_COUNT_COL,
        F.size(F.split(selftext, r"\s+")),
    )
    df = df.withColumn(
        SELF_CHAR_LEN_COL,
        F.length(selftext),
    )

    # time-based features from Unix epoch seconds
    df = df.withColumn("ts", F.to_timestamp(F.from_unixtime(F.col("created_utc"))))
    df = df.withColumn(HOUR_COL, F.hour("ts"))
    df = df.withColumn(DOW_COL, F.dayofweek("ts"))
    df = df.withColumn(
        IS_WEEKEND_COL,
        F.when(F.col(DOW_COL).isin([1, 7]), F.lit(1)).otherwise(F.lit(0)),
    )

    # For safety, fill any remaining nulls in numeric/time features with 0
    numeric_cols_to_fill = [
        TOKEN_COUNT_COL,
        SENT_COUNT_COL,
        HAS_QMARK_COL,
        CHAR_LEN_COL,
        TITLE_TOKEN_COUNT_COL,
        TITLE_CHAR_LEN_COL,
        TITLE_HAS_QMARK_COL,
        SELF_TOKEN_COUNT_COL,
        SELF_CHAR_LEN_COL,
        HOUR_COL,
        DOW_COL,
        IS_WEEKEND_COL,
    ]
    df = df.fillna(0, subset=numeric_cols_to_fill)

    return df


# ===========================
# Pipeline definition
# ===========================

def build_pipeline():
    """
    Build the Spark ML pipeline:

      Stage 1: RegexTokenizer → tokenize clean_text by non-word characters
      Stage 2: StopWordsRemover → remove English stopwords
      Stage 3: CountVectorizer → bag-of-words representation (sparse)
      Stage 4: VectorAssembler → combine numeric features + bow vector
      Stage 5: StandardScaler → scale combined feature vector
      Stage 6: LogisticRegression → linear classifier with class weights

    Class imbalance is handled via `weightCol` on LogisticRegression,
    not via oversampling inside this pipeline.
    """
    tokenizer = RegexTokenizer(
        inputCol=TEXT_COL,
        outputCol="tokens",
        pattern=r"\W+",
        minTokenLength=2,
        toLowercase=True,
    )

    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="tokens_nostop",
    )

    # Lightweight bag-of-words with a modest vocabulary size
    cv = CountVectorizer(
        inputCol="tokens_nostop",
        outputCol="bow",
        vocabSize=10000,
        minDF=20,
    )

    # Numeric features from submissions + comments
    numeric_cols = [
        TOKEN_COUNT_COL,
        SENT_COUNT_COL,
        HAS_QMARK_COL,
        CHAR_LEN_COL,
        TITLE_TOKEN_COUNT_COL,
        TITLE_CHAR_LEN_COL,
        TITLE_HAS_QMARK_COL,
        SELF_TOKEN_COUNT_COL,
        SELF_CHAR_LEN_COL,
        HOUR_COL,
        DOW_COL,
        IS_WEEKEND_COL,
        # Comment-level aggregates
        "comment_score_mean",
        "comment_score_max",
        "comment_score_sum",
        "comment_len_mean",
        "comment_len_max",
        "comment_len_sum",
        "comment_author_nunique",
        "comment_is_op_ratio",
        "comment_contro_mean",
        "comment_contro_max",
        "comment_contro_sum",
    ]

    assembler = VectorAssembler(
        inputCols=numeric_cols + ["bow"],
        outputCol=FEATURES_COL,
    )

    scaler = StandardScaler(
        inputCol=FEATURES_COL,
        outputCol=SCALED_FEATURES_COL,
        withMean=False,
        withStd=True,
    )

    lr = LogisticRegression(
        labelCol=LABEL_COL,
        featuresCol=SCALED_FEATURES_COL,
        weightCol=WEIGHT_COL,
        maxIter=50,
        elasticNetParam=0.0,  # pure L2, we will tune regParam
        family="binomial",
    )

    pipeline = Pipeline(stages=[tokenizer, remover, cv, assembler, scaler, lr])
    return pipeline, lr


# ===========================
# Training with cross-validation
# ===========================

def train_with_cv(train_df, pipeline, lr):
    """
    Train the Logistic Regression pipeline with cross-validation over a small grid:

      - regParam: [0.01, 0.1]
      - elasticNetParam: fixed at 0.0 (L2 only, already set in lr)

    Metric used: areaUnderROC (BinaryClassificationEvaluator).

    Class imbalance is handled via class weights (WEIGHT_COL).
    """
    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1])
        .build()
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2,
    )

    cv_model = cv.fit(train_df)
    print("[INFO] Best model selected.")
    return cv_model.bestModel


# ===========================
# Evaluation
# ===========================

def evaluate(model, df, split):
    """
    Run the model on `df`, compute probability for the positive class,
    and compute AUC / Accuracy / F1 using the default `prediction` column
    (which uses the model's internal threshold, 0.5 for LogisticRegression).

    Steps:
      1) model.transform(df)
      2) extract probability_positive = probability[1]
      3) compute AUC on rawPrediction
      4) compute Accuracy and F1 using prediction
      5) print confusion matrix using prediction
    """
    pred = model.transform(df).cache()

    # Extract positive-class probability from the probability vector
    pred = pred.withColumn(
        "probability_positive",
        vector_to_array("probability")[1],
    )

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="accuracy",
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="f1",
    )

    auc = evaluator_auc.evaluate(pred)
    acc = evaluator_acc.evaluate(pred)
    f1 = evaluator_f1.evaluate(pred)

    print(f"\n====== {split} Metrics (default threshold=0.5) ======")
    print(f"AUC = {auc:.4f}")
    print(f"Accuracy = {acc:.4f}")
    print(f"F1 = {f1:.4f}")

    print(f"[INFO] Confusion Matrix ({split}) using prediction")
    pred.groupBy(LABEL_COL, "prediction").count().show()

    return pred, {
        "split": split,
        "auc": float(auc),
        "accuracy": float(acc),
        "f1": float(f1),
    }


# ===========================
# Save CSV helper
# ===========================

def save_csv(df, path):
    """
    Save a small or medium-sized DataFrame as a single CSV (coalesce(1)),
    with header line.
    """
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(path)


# ===========================
# CLI argument parsing
# ===========================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--submissions-path", required=True)
    p.add_argument("--comments-path", required=True)
    p.add_argument("--metrics-path", required=True)
    p.add_argument("--pred-path", required=True)
    p.add_argument("--model-dir", required=True)
    return p.parse_args()


# ===========================
# Main
# ===========================

def main():
    args = parse_args()
    spark = create_spark()

    # 1) Load submissions
    df = spark.read.parquet(args.submissions_path)
    print(f"[INFO] Loaded submissions: {df.count()} rows")

    # 2) Load and aggregate comment features; join back to submissions
    comment_agg = load_comment_features(spark, args.comments_path)
    df = df.join(comment_agg, df[SUBMISSION_ID] == comment_agg["link_id"], how="left")

    # 3) Fill missing comment-derived values with 0 (for submissions with no comments)
    df = df.fillna(0)

    # 4) Add simple text/time-based features
    df = enrich_features(df)

    # 5) Add binary label
    df = add_label_column(df).cache()

    # 6) Train/test split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # 7) Compute class weights on training data for imbalanced labels
    label_counts = train_df.groupBy(LABEL_COL).count().collect()
    counts = {row[LABEL_COL]: row["count"] for row in label_counts}
    pos_count = counts.get(1.0, 0)
    neg_count = counts.get(0.0, 0)

    print(f"[INFO] Train label counts: {counts}")

    # Avoid division by zero; though with ENG_Q=0.75, pos_count should > 0
    if pos_count == 0 or neg_count == 0:
        print("[WARN] One of the classes has zero count in training; "
              "setting class weights to 1.0 for all rows.")
        train_df = train_df.withColumn(WEIGHT_COL, F.lit(1.0))
    else:
        pos_weight = float(neg_count) / float(pos_count)
        print(f"[INFO] Using positive class weight = {pos_weight:.4f}")
        train_df = train_df.withColumn(
            WEIGHT_COL,
            F.when(F.col(LABEL_COL) == 1.0, F.lit(pos_weight)).otherwise(F.lit(1.0)),
        )

    # 8) Build pipeline and train with cross-validation
    pipeline, lr = build_pipeline()
    model = train_with_cv(train_df, pipeline, lr)

    # 9) Evaluate on train and test using default probability threshold (0.5)
    pred_train, train_metrics = evaluate(model, train_df, "train")
    pred_test, test_metrics = evaluate(model, test_df, "test")

    # 10) Save metrics
    metrics_df = spark.createDataFrame([train_metrics, test_metrics])
    save_csv(metrics_df, args.metrics_path)

    # 11) Save a sample of test predictions for the website / report
    #     We use default prediction and the explicit positive probability.
    pred_test_sample = (
        pred_test.select(
            TEXT_COL,
            NUM_COMMENTS_COL,
            LABEL_COL,
            "prediction",
            "probability_positive",
        )
        .limit(1000)
    )

    save_csv(pred_test_sample, args.pred_path)

    # 12) Save best model
    model.write().overwrite().save(args.model_dir)
    print(f"[INFO] Saved model to: {args.model_dir}")

    spark.stop()


if __name__ == "__main__":
    main()

