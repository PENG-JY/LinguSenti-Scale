import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 0. PATHS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MERGED_CSV = os.path.join(BASE_DIR, "data", "csv", "merged_topics_with_samples.csv")
SUMMARY_CSV = os.path.join(BASE_DIR, "data", "csv", "lda_topics_summary.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# 1. Load data
# ---------------------------------------------------
print("Loading:")
print(" -", MERGED_CSV)
print(" -", SUMMARY_CSV)

df = pd.read_csv(MERGED_CSV)            # contains topic, topic_words, clean_text
df_summary = pd.read_csv(SUMMARY_CSV)    # contains topic, topic_words

print("\nData Loaded:")
print(df.head())
print(df_summary.head())

# ---------------------------------------------------
# 2. Clean @ artifacts in text
# ---------------------------------------------------
def clean_text(x):
    if isinstance(x, str):
        return x.replace("@", " ").replace("  ", " ").strip()
    return x

df["topic_words"] = df["topic_words"].apply(clean_text)
df["clean_text"] = df["clean_text"].apply(clean_text)

# ---------------------------------------------------
# 3. Clean summary with representative sample
# ---------------------------------------------------
topic_summary_clean = (
    df[["topic", "topic_words", "clean_text"]]
    .drop_duplicates(subset=["topic"])
    .sort_values("topic")
    .rename(columns={"clean_text": "sample_submission"})
)

summary_path = f"{OUTPUT_DIR}/topic_summary_clean.csv"
topic_summary_clean.to_csv(summary_path, index=False)
print("Saved:", summary_path)

# ---------------------------------------------------
# 4. Top words bar chart per topic
# ---------------------------------------------------
print("\nGenerating top words bar charts...")

for _, row in topic_summary_clean.iterrows():
    topic = row["topic"]
    words = row["topic_words"]

    if isinstance(words, str):
        word_list = [w.strip() for w in words.split(",")[:10]]
    else:
        continue

    plt.figure(figsize=(10,4))
    plt.barh(word_list, range(len(word_list))[::-1])
    plt.xlabel("Rank")
    plt.title(f"Top Words for Topic {topic}")
    plt.tight_layout()

    plt_path = f"{OUTPUT_DIR}/topic_{topic}_topwords.png"
    plt.savefig(plt_path, dpi=200)
    plt.close()
    print("Saved:", plt_path)

# ---------------------------------------------------
# 5. Save sample submissions into txt files
# ---------------------------------------------------
print("\nSaving representative submission for each topic...")

samples_dir = os.path.join(OUTPUT_DIR, "topic_samples")
os.makedirs(samples_dir, exist_ok=True)

for _, row in topic_summary_clean.iterrows():
    topic = row["topic"]
    sample = row["sample_submission"]

    file_path = os.path.join(samples_dir, f"topic_{topic}_sample.txt")

    with open(file_path, "w") as f:
        f.write(sample if isinstance(sample, str) else "")

    print("Saved:", file_path)

print("\nAll topic interpretation artifacts generated!")
print("Output directory:", OUTPUT_DIR)
