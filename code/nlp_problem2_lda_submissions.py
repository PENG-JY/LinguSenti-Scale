from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, monotonically_increasing_id
from pyspark.ml.feature import RegexTokenizer, CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.ml.functions import vector_to_array
import pandas as pd

# ----------------------------------------------------
# 1. Spark Session
# ----------------------------------------------------
spark = (
    SparkSession.builder
    .appName("LDA_Submissions_Full")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.InstanceProfileCredentialsProvider")
    .getOrCreate()
)

# ----------------------------------------------------
# 2. Load Data from S3
# ----------------------------------------------------
input_path = "s3a://jc3482-dsan6000-datasets/project/reddit/parquet/cleaned/cleaned_submissions/clean_submissions/"
df = spark.read.parquet(input_path).select("id", "clean_text").na.drop()

# ----------------------------------------------------
# 3. NLP Pipeline (tokens → count vector)
# ----------------------------------------------------
tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="tokens", pattern="\\W+")
df = tokenizer.transform(df)

cv = CountVectorizer(inputCol="tokens", outputCol="features", vocabSize=20000, minDF=5)
cv_model = cv.fit(df)
df = cv_model.transform(df)

# ----------------------------------------------------
# 4. LDA
# ----------------------------------------------------
lda = LDA(k=15, maxIter=10, featuresCol="features")
lda_model = lda.fit(df)

# Topic descriptions
topics = lda_model.describeTopics(10).toPandas()
vocab = cv_model.vocabulary

def decode_topic_words(row):
    return ", ".join([vocab[i] for i in row["termIndices"]])

topics["topic_words"] = topics.apply(decode_topic_words, axis=1)

# ----------------------------------------------------
# 5. Topic assignment per submission
# ----------------------------------------------------
df_topics = lda_model.transform(df)

# Convert vectorUDT → array and compute top topic
df_topics = (
    df_topics
    .withColumn("topic_array", vector_to_array("topicDistribution"))
    .withColumn("top_topic", expr("array_position(topic_array, array_max(topic_array)) - 1"))
)

df_assign = df_topics.select("id", "clean_text", "top_topic")

# ----------------------------------------------------
# 6. Pick representative sample submissions
# ----------------------------------------------------
df_topics_pd = df_assign.toPandas()

samples = (
    df_topics_pd.groupby("top_topic")
    .apply(lambda x: x.sort_values("id").head(1))  # simple deterministic sample
    .reset_index(drop=True)
)

# ----------------------------------------------------
# 7. Save outputs
# ----------------------------------------------------
topics.to_csv("lda_topics_summary.csv", index=False)
samples.to_csv("lda_topic_samples.csv", index=False)
df_assign.to_csv("lda_submissions_assignments.csv", index=False)

print("\nLDA DONE!")
print("Saved: lda_topics_summary.csv")
print("Saved: lda_topic_samples.csv (one best sample per topic)")
print("Saved: lda_submissions_assignments.csv")
