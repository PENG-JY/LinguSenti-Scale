# ==========================================================
# Topic Type vs Average Comments (2023–2024)
# ==========================================================
from pyspark.sql import SparkSession, functions as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Create directories for outputs ---
os.makedirs("data/csv", exist_ok=True)
os.makedirs("data/plots", exist_ok=True)

# --- Spark session with S3 support ---
spark = (
    SparkSession.builder
    .appName("Q3_TopicType_AvgComments")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
    .getOrCreate()
)

# --- S3 path ---
submissions_path = "s3a://jc3482-dsan6000-datasets/project/reddit/parquet/submissions/"

# --- Read data ---
df = spark.read.parquet(submissions_path)
df = df.withColumn("num_comments", F.col("num_comments").cast("int"))
df = df.withColumn("score", F.col("score").cast("int"))

# --- Create topic labels using regex keywords ---
df_labeled = df.withColumn(
    "topic_label",
    F.when(F.lower(F.col("title")).rlike("break.?up|separat|divorc"), "breakup")
     .when(F.lower(F.col("title")).rlike("marri|wife|husband|spouse"), "marriage")
     .when(F.lower(F.col("title")).rlike("boyfriend|girlfriend|dating|crush"), "dating")
     .when(F.lower(F.col("title")).rlike("family|parent|mom|dad|sibling"), "family")
     .when(F.lower(F.col("title")).rlike("confess|secret|truth"), "confession")
     .otherwise("other")
)

# --- Aggregate: avg comments & avg score ---
topic_stats = (
    df_labeled.groupBy("topic_label")
    .agg(
        F.count("*").alias("count"),
        F.avg("num_comments").alias("avg_num_comments"),
        F.avg("score").alias("avg_score")
    )
    .orderBy(F.desc("avg_num_comments"))
)

# --- Save results ---
topic_stats.coalesce(1).write.mode("overwrite").option("header", True)\
    .csv("data/csv/eda_topic_summary.csv")

# --- Visualization ---
pdf = topic_stats.toPandas()
sns.set_style("whitegrid")
plt.figure(figsize=(8,5))
sns.barplot(
    data=pdf.sort_values("avg_num_comments", ascending=False),
    x="avg_num_comments", y="topic_label", hue="topic_label",
    palette="viridis", legend=False
)
plt.title("Average Comments by Topic Type (2023–2024)")
plt.xlabel("Average Number of Comments")
plt.ylabel("Topic Type")
plt.tight_layout()
plt.savefig("data/plots/eda_avg_comments_by_topic.png", dpi=300)
plt.close()
