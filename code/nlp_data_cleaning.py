
import sys
import logging
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import sparknlp
from sparknlp.pretrained import ResourceDownloader
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Stemmer


# ---------------- Logger ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- Spark Session Builder ----------------
def create_spark_session(master_url):
    logger.info(f"Initializing Spark session on master: {master_url}")

    spark = (
        SparkSession.builder
        .appName("Problem3_Reddit_Cluster")
        .master(master_url)

        # Spark NLP JAR (MUST be high)
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.4.4")

        # Spark performance + your configs
        .config("spark.executor.instances", "3")
        .config("spark.executor.memory", "4g")
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")
        .config("spark.driver.memory", "4g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        # S3 configs
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain"
        )
        .config("spark.hadoop.fs.s3a.connection.maximum", "200")
        .config("spark.hadoop.fs.s3a.threads.max", "200")

        # Adaptive execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

        # Disable Arrow (required for Spark NLP)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created successfully")
    return spark


# ---------------- JSON-safe Cleaning ----------------
def build_json_safe_text(df, col):
    return (df
        .withColumn(col, F.col(col).cast("string"))
        .withColumn(col, F.regexp_replace(col, r"\\n", " "))
        .withColumn(col, F.regexp_replace(col, r"\\t", " "))
        .withColumn(col, F.regexp_replace(col, r"\\\"", "\""))
        .withColumn(col, F.regexp_replace(col, r"&nbsp;", " "))
        .withColumn(col, F.regexp_replace(col, r"[\{\}]", " "))
    )


# ---------------- Main ----------------
def main():

    if len(sys.argv) < 2:
        raise ValueError("Usage: uv run python nlp_data_cleaning.py spark://<MASTER_IP>:7077")

    master_url = sys.argv[1]
    spark = create_spark_session(master_url)

    print("Spark version:", spark.version)
    print("Spark NLP version:", sparknlp.version())
    print("Connected to Spark master:", master_url)

    # ----------------------------------------
    # IMPORT SPARK NLP ANNOTATORS *AFTER* spark session is created
    # ----------------------------------------
    from sparknlp.base import DocumentAssembler, Finisher
    from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel
    from pyspark.ml import Pipeline

    # ---------------- Create NLP Pipeline ----------------
    document = DocumentAssembler().setInputCol("raw_text").setOutputCol("document")

    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")

    normalizer = Normalizer().setInputCols(["token"]).setOutputCol("norm").setLowercase(True)

    stopwords = StopWordsCleaner().setInputCols(["norm"]).setOutputCol("clean_tokens")

    stemmer = Stemmer() \
        .setInputCols(["clean_tokens"]) \
        .setOutputCol("stem")

    finisher = Finisher() \
        .setInputCols(["stem"]) \
        .setOutputCols(["clean_text"]) \
        .setOutputAsArray(False)

    pipeline = Pipeline(stages=[
        document,
        tokenizer,
        normalizer,
        stopwords,
        stemmer,
        finisher
    ])

    # empty DF for fitting pipeline
    empty_df = spark.createDataFrame([[""]]).toDF("raw_text")
    pipeline_model = pipeline.fit(empty_df)

    # ---------------- Combined Cleaning Function ----------------
    def apply_cleaning(df):
        df = df.withColumn(
            "raw_text",
            F.regexp_replace("raw_text", r"http\S+|www\S+", " ")
        ).withColumn(
            "raw_text",
            F.regexp_replace("raw_text", r"&amp;|&quot;|&gt;|&lt;", " ")
        ).withColumn(
            "raw_text",
            F.regexp_replace("raw_text", r"[\*\_>\[\]\(\)]", " ")
        )

        df = build_json_safe_text(df, "raw_text")
        df = pipeline_model.transform(df)
        return df

    # # ============================================
    # # 4. CLEAN SUBMISSIONS
    # # ============================================
    # print("\n=== Cleaning Submissions ===")

    # SUB_IN  = "s3a://jp2132-spark-reddit/reddit/submissions/"
    # SUB_OUT = "s3a://jp2132-spark-reddit/reddit/clean/clean_submissions/"

    # sub = spark.read.parquet(SUB_IN)
    # print("Submissions count =", sub.count())

    # sub = sub.filter(
    #     (F.col("selftext") != "[removed]") &
    #     (F.col("selftext") != "[deleted]")
    # )

    # sub = build_json_safe_text(sub, "title")
    # sub = build_json_safe_text(sub, "selftext")

    # sub = sub.withColumn("raw_text", F.concat_ws(" ", "title", "selftext"))
    # clean_sub = apply_cleaning(sub)

    # clean_sub.write.mode("overwrite").parquet(SUB_OUT)
    # print("✔ Submissions cleaned and saved.")

    # ============================================
    # 5. CLEAN COMMENTS (optional)
    # ============================================
    print("\n=== Cleaning Comments ===")
    COM_IN  = "s3a://jp2132-spark-reddit/reddit/comments/"
    COM_OUT = "s3a://jp2132-spark-reddit/reddit/clean/clean_comments/"
    
    com = spark.read.parquet(COM_IN)
    print("Comments count =", com.count())
    
    com = com.filter(
        (F.col("body") != "[removed]") &
        (F.col("body") != "[deleted]")
    )
    
    com = build_json_safe_text(com, "body")
    com = com.withColumn("raw_text", F.col("body"))
    
    clean_com = apply_cleaning(com)
    clean_com.write.mode("overwrite").parquet(COM_OUT)
    
    print("✔ Comments cleaned and saved.\n")

    spark.stop()
    print("\n==============================")
    print("Cleaning job completed.")
    print("==============================\n")


# ---------------- Script Entry ----------------
if __name__ == "__main__":
    main()
