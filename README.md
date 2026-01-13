# LinguSenti: Behavioral & Linguistic Patterns in Online Relationships

**Team:** Jiayi Peng, Kexin Lyu, Stacy Che, Yanan Wu

---

## Project Overview

Millions of users seek advice and support on Reddit's r/relationship_advice, sharing deeply personal, emotional experiences.  
**Our project explores:**  
_How do linguistic patterns, emotional expressions, and posting time influence community engagement and sentiment in large-scale online relationship forums?_

Leveraging distributed big data processing (Apache Spark on AWS EC2), we analyze nearly 10 million posts and comments to reveal actionable insights for online community management, support, and moderation.

---

## Dataset

- **Source:** Reddit comments and submissions (Pushshift; r/relationship_advice)
- **Time Range:** 2023-06-01 to 2024-07-31
- **Scale:** 8.67 million comments, 1.15 million submissions (total ~1.83 GB)
- **Average post score:** 8.78

For summary details, see  
[`data/csv/dataset_summary.csv`](data/csv/dataset_summary.csv),  
[`data/csv/subreddit_statistics.csv`](data/csv/subreddit_statistics.csv).

---

## Business Questions

We address **10 business questions** using EDA, NLP, and ML:

### Exploratory Data Analysis (EDA)
1. _How do posting and commenting time patterns vary?_  
   **→** Cycles by season, hour, day of week.
2. _How do post lengths correlate with engagement (score/comment count)?_  
   **→** Relationship between post length and user response.
3. _Which post types ("confession", "breakup", "family", "marriage", etc.) receive the most comments?_  
   **→** Topic-specific engagement patterns.

### Natural Language Processing (NLP)
4. _How do emotional tones shift across temporal patterns?_  
   **→** Emotion cycles by hour, weekday, and month.
5. _What are the main themes discovered by topic modeling?_  
   **→** LDA/BERTopic reveals trust issues, infidelity, family pressure, etc.
6. _Does sentiment influence engagement?_  
   **→** Correlation of post sentiment with reactions.
7. _What early linguistic markers predict high engagement?_  
   **→** Vocabulary, question forms, and self-disclosure as predictors.

### Machine Learning (ML)
8. _Can we predict whether a comment is "controversial"?_  
   **→** Using text and metadata features (AUC ≈ 0.76).
9. _Can we predict high-engagement posts (top 25% comment count)?_  
   **→** Early prediction of viral discussions (AUC ≈ 0.65).
10. _Can we forecast daily posting volume?_  
    **→** Regression using time and engagement features (R² ≈ 0.56).

See detailed approaches in [`BUSINESS_QUESTIONS.md`](BUSINESS_QUESTIONS.md).

---

## Methodology

- **Distributed Data Processing:** PySpark on AWS Spark cluster, S3 for storage
- **EDA:** Temporal aggregations, user activity plots, heatmaps
- **NLP:** Text cleaning, tokenization, NRC emotion lexicon mapping, topic modeling (LDA), sentiment scoring
- **ML:** Logistic and Ridge Regression, feature engineering, cross-validation, feature importance evaluation

Each pipeline stage builds on the previous:  
_Data acquisition and filtering → EDA (behavioral foundations) → NLP (emotion & themes) → ML (prediction) → Conclusion & recommendations._

All results are presented in the website ([docs/index.html](docs/index.html)) and supporting analysis pages.

---

## Key Findings & Business Impact

- **Temporal & Topic Patterns:** User engagement spikes in evenings and summer, with marriage and breakup topics generating most activity.  
- **Emotion peaks and cycles:** Positive emotion rises mid-week and in spring, negative emotions cluster on weekends and late at night.
- **Predictive power:** ML models reliably forecast controversy, high-engagement, and posting volume using simple features.
- **Platform recommendations:**  
  - Prioritize support and moderation for emotionally intense/high-traffic windows.
  - Surface and encourage emotionally expressive, advice-seeking posts.
  - Use predictive signals for moderation scheduling and system scaling.

See [docs/conclusion.html](docs/conclusion.html) for more insights and recommendations.

---

## Repository Structure

```
project-root/
├── code/ # PySpark & Python scripts for EDA/NLP/ML
├── data/
│ ├── csv/ # Dataset statistics, precomputed tables
│ └── plots/ # Analysis visualizations
├── docs/ # Website (HTML) and final report
├── website-source/ # Quarto or markdown sources for website
├── BUSINESS_QUESTIONS.md
├── EDA.md / NLP.md / ML.md
├── README_NEW.md # (this file, the new full project readme)
└── ...
```

---

## How to Explore

- **Main results site:** [docs/index.html](docs/index.html)
- **Detailed analysis:** [docs/eda.html](docs/eda.html), [docs/nlp.html](docs/nlp.html), [docs/ml.html](docs/ml.html)
- **Full technical roadmap / answers:** [BUSINESS_QUESTIONS.md](BUSINESS_QUESTIONS.md)

---

## Acknowledgments

- Project members: Jiayi Peng, Kexin Lyu, Stacy Che, Yanan Wu (DSAN 6000 Big Data Analytics, Fall 2025)
- Technology & Data: Apache Spark, PySpark, AWS EC2, Quarto, Reddit (Pushshift)
