#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import logging

import math
import numpy as np
import xgboost as xgb

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODEL_OUTPUT_PATH = "s3a://yw1150-dsan6000-project/reddit/ml/models/"
PLOTS_DIR = "data/plots"


SUBMISSIONS_PATH = "s3a://yw1150-dsan6000-project/reddit/submissions/"

DAILY_AGG_OUTPUT = "s3a://yw1150-dsan6000-project/reddit/ml/daily_posts/"
DAILY_FEATURES_OUTPUT = "s3a://yw1150-dsan6000-project/reddit/ml/daily_posts_features/"

METRICS_CSV = "data/csv/ml_post_volume_regression_metrics.csv"
FEATURE_IMPORTANCE_CSV = "data/csv/ml_post_volume_feature_importance.csv"

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------- Spark Session ----------------
def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("DSAN6000_PostVolumeRegression")
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


#  Data Loading
def load_submissions(spark: SparkSession):
    logger.info(f"Reading submissions from: {SUBMISSIONS_PATH}")
    df = spark.read.parquet(SUBMISSIONS_PATH)

    logger.info(f"Raw submissions rows: {df.count()}")
    logger.info(f"Columns: {df.columns}")

    df = df.withColumn("date", F.to_date("date"))
    
    logger.info("After to_date(date):")
    df.select("date").summary().show()

    return df



#  Daily Aggregation
def aggregate_daily_posts(df):
    daily = (
        df.groupBy("date")
        .agg(F.count("*").alias("n_posts"))
        .orderBy("date")
    )
    return daily


#  Feature Engineering
def add_time_features(daily_df):
    df = (
        daily_df
        .withColumn("day_of_week", F.dayofweek("date"))  # 1-7
        .withColumn("month", F.month("date"))
        .withColumn("day_of_year", F.dayofyear("date"))
    )

    df = df.withColumn(
        "is_weekend",
        F.col("day_of_week").isin([6, 7]).cast("int")
    )

    w = Window.orderBy("date")

    # Lag features
    df = df.withColumn("lag_1", F.lag("n_posts", 1).over(w))
    df = df.withColumn("lag_2", F.lag("n_posts", 2).over(w))
    df = df.withColumn("lag_3", F.lag("n_posts", 3).over(w))
    df = df.withColumn("lag_7", F.lag("n_posts", 7).over(w))
    df = df.withColumn("lag_14", F.lag("n_posts", 14).over(w))
    df = df.withColumn("lag_21", F.lag("n_posts", 21).over(w))
    df = df.withColumn("lag_28", F.lag("n_posts", 28).over(w))

    # Rolling windows
    w7 = w.rowsBetween(-6, 0)
    w14 = w.rowsBetween(-13, 0)
    w30 = w.rowsBetween(-29, 0)

    df = df.withColumn("rolling_mean_7", F.avg("n_posts").over(w7))
    df = df.withColumn("rolling_std_7", F.stddev("n_posts").over(w7))

    df = df.withColumn("rolling_mean_14", F.avg("n_posts").over(w14))
    df = df.withColumn("rolling_std_14", F.stddev("n_posts").over(w14))

    df = df.withColumn("rolling_mean_30", F.avg("n_posts").over(w30))

    # Fourier terms (seasonality)
    df = df.withColumn(
        "fourier_sin_1",
        F.sin(F.lit(2 * math.pi) * F.col("day_of_year") / F.lit(365.0))
    )
    df = df.withColumn(
        "fourier_cos_1",
        F.cos(F.lit(2 * math.pi) * F.col("day_of_year") / F.lit(365.0))
    )

    feature_cols = [
        "n_posts",
        "day_of_week", "month", "day_of_year", "is_weekend",
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "rolling_std_7", "rolling_std_14",
        "fourier_sin_1", "fourier_cos_1",
    ]

    df = df.dropna(subset=feature_cols)

    return df


def load_daily_features(spark: SparkSession):
    logger.info(f"Reading daily features from: {DAILY_FEATURES_OUTPUT}")
    df = spark.read.parquet(DAILY_FEATURES_OUTPUT)
    logger.info(f"Loaded {df.count()} rows of daily features.")
    return df


#  Train/Val/Test Split
def time_based_split(df, train_ratio=0.7, val_ratio=0.15):
    df = df.orderBy("date")

    w = Window.orderBy("date")
    df_indexed = df.withColumn("row_num", F.row_number().over(w))

    total = df_indexed.count()
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_df = df_indexed.filter(F.col("row_num") <= train_end)
    val_df = df_indexed.filter(
        (F.col("row_num") > train_end) & (F.col("row_num") <= val_end)
    )
    test_df = df_indexed.filter(F.col("row_num") > val_end)

    train_df = train_df.drop("row_num")
    val_df = val_df.drop("row_num")
    test_df = test_df.drop("row_num")

    return train_df, val_df, test_df


#  ML Pipeline
def build_regression_pipeline(feature_cols):
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features"
    )

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True
    )

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="n_posts",
        maxIter=50,
        maxDepth=5,
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, scaler, gbt])
    return pipeline


def train_and_tune(train_df, val_df, feature_cols):
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features"
    )

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True
    )

    train_val_df = train_df.unionByName(val_df)

    evaluator = RegressionEvaluator(
        labelCol="n_posts",
        predictionCol="prediction",
        metricName="rmse"
    )

    models_info = []

    # ===== GBT =====
    """
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="n_posts",
        maxIter=50,
        maxDepth=5,
        seed=42
    )

    pipeline_gbt = Pipeline(stages=[assembler, scaler, gbt])

    param_grid_gbt = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 5, 7])
        .addGrid(gbt.maxIter, [50, 100])
        .build()
    )

    cv_gbt = CrossValidator(
        estimator=pipeline_gbt,
        estimatorParamMaps=param_grid_gbt,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2,
        seed=42
    )

    cv_model_gbt = cv_gbt.fit(train_val_df)
    best_gbt = cv_model_gbt.bestModel
    rmse_gbt = evaluator.evaluate(best_gbt.transform(train_val_df))
    models_info.append(("GBTRegressor", best_gbt, rmse_gbt))
    """

    # ===== RandomForest =====
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="n_posts",
        numTrees=200,
        maxDepth=10,
        seed=42
    )

    pipeline_rf = Pipeline(stages=[assembler, scaler, rf])

    param_grid_rf = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, [8, 10, 12])
        .addGrid(rf.numTrees, [100, 200])
        .build()
    )

    cv_rf = CrossValidator(
        estimator=pipeline_rf,
        estimatorParamMaps=param_grid_rf,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2,
        seed=42
    )

    cv_model_rf = cv_rf.fit(train_val_df)
    best_rf = cv_model_rf.bestModel
    rmse_rf = evaluator.evaluate(best_rf.transform(train_val_df))
    models_info.append(("RandomForestRegressor", best_rf, rmse_rf))

    # ===== Ridge Regression (L2) =====
    lr_ridge = LinearRegression(
        featuresCol="features",
        labelCol="n_posts",
        regParam=0.1,
        elasticNetParam=0.0
    )

    pipeline_ridge = Pipeline(stages=[assembler, scaler, lr_ridge])
    best_ridge = pipeline_ridge.fit(train_val_df)
    rmse_ridge = evaluator.evaluate(best_ridge.transform(train_val_df))
    models_info.append(("RidgeRegression", best_ridge, rmse_ridge))

    # ===== ElasticNet =====
    lr_en = LinearRegression(
        featuresCol="features",
        labelCol="n_posts",
        regParam=0.1,
        elasticNetParam=0.5
    )

    pipeline_en = Pipeline(stages=[assembler, scaler, lr_en])
    best_en = pipeline_en.fit(train_val_df)
    rmse_en = evaluator.evaluate(best_en.transform(train_val_df))
    models_info.append(("ElasticNet", best_en, rmse_en))

    # ===== XGBoost (version-safe) =====
    pdf_train = (
        train_df
        .select(["n_posts"] + feature_cols)
        .toPandas()
    )
    pdf_val = (
        val_df
        .select(["n_posts"] + feature_cols)
        .toPandas()
    )

    X_train = pdf_train[feature_cols].values
    y_train = pdf_train["n_posts"].values

    X_val = pdf_val[feature_cols].values
    y_val = pdf_val["n_posts"].values

    # ===== XGBoost =====
    pdf_train_val = (
        train_val_df
        .select(["n_posts"] + feature_cols)
        .toPandas()
    )

    X_train_val = pdf_train_val[feature_cols].values
    y_train_val = pdf_train_val["n_posts"].values

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    xgb_model.fit(X_train_val, y_train_val)

    y_pred_xgb = xgb_model.predict(X_train_val)
    rmse_xgb = float(np.sqrt(np.mean((y_pred_xgb - y_train_val) ** 2)))

    models_info.append(("XGBoostRegressor", xgb_model, rmse_xgb))

    for name, _, rmse in models_info:
        logger.info(f"[train+val] {name} RMSE = {rmse:.4f}")

    models_dict = {name: model for name, model, _ in models_info}
    return models_dict, evaluator




def evaluate_and_save(models_dict, test_df, feature_cols, spark: SparkSession):
    metrics_rows = []

    evaluator_rmse = RegressionEvaluator(
        labelCol="n_posts", predictionCol="prediction", metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol="n_posts", predictionCol="prediction", metricName="mae"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="n_posts", predictionCol="prediction", metricName="r2"
    )

    best_name = None
    best_rmse = float("inf")
    predictions_for_plot = None

    for name, model in models_dict.items():
        if name == "XGBoostRegressor":
            # XGBoost: predict on pandas, then convert back to Spark
            pdf_test = (
                test_df
                .select(["date", "n_posts"] + feature_cols)
                .toPandas()
            )
            X_test = pdf_test[feature_cols].values
            y_pred = model.predict(X_test)
            pdf_test["prediction"] = y_pred

            preds = spark.createDataFrame(pdf_test)
        else:
            preds = model.transform(test_df)

        rmse = evaluator_rmse.evaluate(preds)
        mae = evaluator_mae.evaluate(preds)
        r2 = evaluator_r2.evaluate(preds)

        metrics_rows.append((name, rmse, mae, r2))

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            predictions_for_plot = preds


    metrics_df = spark.createDataFrame(
        metrics_rows, ["model", "rmse", "mae", "r2"]
    )

    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    metrics_df.toPandas().to_csv(METRICS_CSV, index=False)

    # Feature importance: prefer RandomForest; else any tree model with featureImportances
    tree_model = None
    if "RandomForestRegressor" in models_dict:
        last_stage = models_dict["RandomForestRegressor"].stages[-1]
        if hasattr(last_stage, "featureImportances"):
            tree_model = last_stage
    else:
        best_model = models_dict[best_name]
        if hasattr(best_model, "stages"):
            last_stage = best_model.stages[-1]
            if hasattr(last_stage, "featureImportances"):
                tree_model = last_stage

    if tree_model is not None:
        importances = tree_model.featureImportances.toArray().tolist()
        fi_rows = list(zip(feature_cols, importances))
        fi_df = spark.createDataFrame(fi_rows, ["feature", "importance"])

        os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_CSV), exist_ok=True)
        fi_df.toPandas().sort_values(
            "importance", ascending=False
        ).to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    print(f"Saved metrics to: {METRICS_CSV}")
    print(f"Saved feature importance to: {FEATURE_IMPORTANCE_CSV}")

    best_model = models_dict.get(best_name)
    if best_model is not None and hasattr(best_model, "write"):
        best_model.write().overwrite().save(MODEL_OUTPUT_PATH)
        logger.info(f"Saved best TEST model ({best_name}) to: {MODEL_OUTPUT_PATH}")
    else:
        logger.info(f"Best TEST model is {best_name} (non-Spark); not saving to MODEL_OUTPUT_PATH.")

    # Plot predictions over time for best model on test set
    if predictions_for_plot is not None:
        plot_path = os.path.join(PLOTS_DIR, "ml_post_volume_best_model_test_predictions.png")
        plot_predictions_over_time(predictions_for_plot, plot_path)
        logger.info(f"Saved prediction plot to: {plot_path}")
    logger.info(f"Best model on TEST: {best_name} (RMSE={best_rmse:.4f})")



def plot_predictions_over_time(predictions, output_path):
    # Collect test predictions to pandas
    pdf = (
        predictions
        .select("date", "n_posts", "prediction")
        .orderBy("date")
        .toPandas()
    )

    if pdf.empty:
        logger.warning("No predictions available for plotting.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(pdf["date"], pdf["n_posts"], label="Actual")
    plt.plot(pdf["date"], pdf["prediction"], label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Daily post volume")
    plt.title("Actual vs Predicted Daily Post Volume (Test Set)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


#  Main
def main():
    spark = create_spark_session()

    # Step 1: aggregation + feature engineering
    submissions_df = load_submissions(spark)
    daily_df = aggregate_daily_posts(submissions_df)

    logger.info(f"Daily df rows: {daily_df.count()}")
    daily_df.orderBy("date").show(5)

    # Write daily aggregation to S3
    daily_df.coalesce(1).write.mode("overwrite").parquet(DAILY_AGG_OUTPUT)
    logger.info(f"Saved daily aggregation to: {DAILY_AGG_OUTPUT}")

    # Build features
    daily_features_df = add_time_features(daily_df)

    logger.info(f"Feature df rows: {daily_features_df.count()}")
    daily_features_df.orderBy("date").show(5)

    # Write features to S3
    daily_features_df.coalesce(1).write.mode("overwrite").parquet(DAILY_FEATURES_OUTPUT)
    logger.info(f"Saved daily features to: {DAILY_FEATURES_OUTPUT}")


    daily_features_df = load_daily_features(spark)
    feature_cols = [
        "day_of_week", "month", "day_of_year", "is_weekend",
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "rolling_std_7", "rolling_std_14",
        "fourier_sin_1", "fourier_cos_1",
    ]

    train_df, val_df, test_df = time_based_split(daily_features_df)
    models_dict, evaluator = train_and_tune(train_df, val_df, feature_cols)
    evaluate_and_save(models_dict, test_df, feature_cols, spark)

    spark.stop()


if __name__ == "__main__":
    main()