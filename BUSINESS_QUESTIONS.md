# Business Questions and Technical Approaches

**Project:** Understanding Emotional Dynamics in Online Relationship Discussions

**Team:** Stacy Che, Kexin Lyu, Jiayi Peng, Yanan Wu

**Dataset:** Reddit r/relationship_advice (2023-2024)

**High-Level Problem Statement:** How can we understand emotional dynamics and communication patterns in online relationship discussions to improve digital well-being and support?

---

## Question 1: How do posting behaviors vary by season, hour, and day of week?

**Analysis Type:** EDA

**Technical Approach:**
- Load submissions and comments DataFrames from S3 into PySpark and filter by `subreddit == 'relationship_advice'`, then cache for repeated access

- Convert Unix `created_utc` timestamps to datetime using `from_unixtime()` and extract temporal features with `dayofweek()`, `hour()`, `date_format()` functions on both DataFrames

- Use `groupBy('day_of_week', 'hour')` with `count()` and `agg()` operations to aggregate posting volumes (submissions) and commenting volumes (comments) separately across temporal dimensions

- Union or compare aggregated results from both submissions and comments to identify complementary patterns

- Collect aggregated results and create dual heatmaps (Matplotlib/Seaborn) visualizing activity patterns by weekday-hour matrix for both posts and comments

- Expected outcome: Identify optimal times for community activity and understand when users seek relationship advice most frequently (hypothesis: posts peak late evening, comments distributed throughout day)

---

## Question 2: How do post lengths and engagement (score / comment count) correlate over time?

**Analysis Type:** EDA

**Technical Approach:**
- Use PySpark SQL `length()` function to calculate character counts and `size(split())` for word counts on `selftext` column from submissions DataFrame

- Calculate combined text length from `concat(title, ' ', selftext)` to capture total post content length

- Create time windows (monthly/quarterly) using `window()` function on `created_utc` timestamp for temporal aggregation

- Use existing `num_comments` column directly from submissions (no join needed), along with `score` for engagement metrics

- Compute Pearson correlation using `corr('text_length', 'score')` and `corr('text_length', 'num_comments')` within each time window, plus statistical aggregations (`avg()`, `stddev()`)

- Visualize correlation trends over time and create scatter plots with regression lines to identify temporal patterns in post length-engagement relationship

- Expected outcome: Understand whether longer or shorter posts receive more engagement and how this relationship evolves (hypothesis: concise posts may perform better)

---

## Question 3: Which types of posts (“confession”, “breakup”, “family”, “marriage”, etc.) receive the most comments on average?

**Analysis Type:** EDA

**Technical Approach:**
- Create topic categories by applying keyword-based classification on `title` and `selftext` columns using `when()` with `contains()` or `rlike()` for pattern matching (e.g., "dating", "marriage", "family", "breakup", "cheating")

- Extract year from `created_utc` using `year()` function and handle multi-label cases by prioritizing keywords or assigning to "General" category

- Use `groupBy('topic_category', 'year').count()` to aggregate topic frequencies by year across 2023-2024

- Calculate percentage distributions using window functions `sum().over(Window.partitionBy('year'))` for year-wise totals

- Perform year-over-year comparison by pivoting data with `pivot('year')` and calculating growth rates using PySpark SQL expressions

- Create stacked bar charts and line plots showing topic distribution evolution from 2023 to 2024

- Expected outcome: Identify dominant relationship issue categories (e.g., Dating, Communication, Trust) and detect shifts in community concerns over time (e.g., increased mental health or therapy discussions)

---

## Question 4: How do emotional tones in Reddit relationship discussions change across different temporal patterns (hour of day, weekday, and month), and what do these shifts reveal about user commenting behavior and community sentiment dynamics?

**Analysis Type:** NLP

**Technical Approach:**

- Focus analysis on comments (clean_text from the cleaned comments dataset) because comments better represent community emotional responses than submissions.

- Use a lexicon-based emotion detection approach:
  - Load the NRC Emotion Lexicon as the external labeled dataset for 10 emotions (anger, fear, joy, sadness, etc.).
  - Apply token-level stemming (same stemmer used during cleaning) to both comment tokens and NRC lexicon entries to ensure consistent matching.

- Tokenize comments using the same splitter as in Q1 (e.g., split(clean_text, "@")), explode tokens, and join with NRC to assign emotions.

- Count matched tokens per emotion and pivot into wide format:
  - Produce columns such as anger_count, joy_count, fear_count.
  - Compute emotion_token_count and normalize to emotion densities (e.g., anger_density = anger_count / total_tokens) to control for varying comment lengths.

- Extract temporal variables from created_utc:
  - hour using hour()
  - month using month()
  - weekday using dayofweek() for ordering and date_format(..., "E") for label display

- Aggregate emotion densities across temporal dimensions:
  - groupBy('hour').avg(...) to analyze hourly emotional cycles
  - groupBy('weekday').avg(...) to compare weekday vs weekend shifts
  - groupBy('month').avg(...) to study seasonal or holiday-related patterns

- Write per-comment emotion features to S3 Parquet and export aggregated tables for visualization.

- Visualize temporal emotion patterns through:
  - hourly emotion trend plots
  - weekday emotion plots or heatmaps
  - monthly emotion trend plots
  - histograms illustrating emotion density distributions

- Expected outcome:
  - Detect temporal emotional rhythms (e.g., increases in sadness or fear late at night)
  - Identify weekday versus weekend differences in community emotion
  - Observe seasonal trends such as holiday-related emotional shifts
  - Provide insights into how user commenting behavior reflects broader sentiment dynamics within the community.


---

## Question 5: What are the most common themes from topic modeling (LDA or BERTopic) in relationship stories (e.g., trust issues, infidelity, family pressure)?

**Analysis Type:** NLP

**Technical Approach:**
- Concatenate `title` and `selftext` columns from submissions DataFrame to create full post text for analysis

- Build Spark NLP preprocessing pipeline: DocumentAssembler → Tokenizer → Normalizer → StopWordsCleaner → Lemmatizer → Finisher to clean and normalize text

- Convert processed tokens to TF-IDF features using `HashingTF` and `IDF` transformers from PySpark ML

- Apply Spark MLlib `LDA` model with 8-12 topics (k value) using `.fit()` on vectorized features, setting maxIterations=20, optimizer='online'

- Extract topic-term distributions using `describeTopics(maxTermsPerTopic=15)` and map term indices back to words using vocabulary lookup

- Assign dominant topic to each document using `transform()` and analyze topic prevalence with `groupBy('topic').count()`

- Manually label topics based on top 10-15 keywords and visualize using pyLDAvis or custom word clouds for each topic

- Expected outcome: Identify and categorize main relationship problems (e.g., trust, infidelity, communication breakdown, family conflict) for targeted support strategies

---

## Question 6: How does sentiment in posts correlate with engagement metrics (score and comment count)?

**Analysis Type:** NLP

**Technical Approach:**
- Apply Spark NLP sentiment pipeline (from Question 5) to submissions to extract sentiment scores and convert categorical sentiment to numerical scores (-1 for negative, 0 for neutral, +1 for positive)

- Use alternative scoring with `ViveknSentimentModel` confidence scores for continuous sentiment values from `title` + `selftext` concatenation

- Join sentiment results with engagement metrics using post `id` as key (or add sentiment as new column to submissions DataFrame)

- Calculate Pearson and Spearman correlations using `corr()` function between sentiment scores and engagement metrics (`score`, `num_comments`)

- Fit linear regression model using Spark ML `LinearRegression` with sentiment as predictor and engagement as target, examining R² and coefficients

- Create scatter plots with regression lines and box plots grouped by sentiment category to visualize engagement distributions

- Expected outcome: Determine whether emotionally charged posts (especially negative) receive more attention and responses, informing content moderation priorities

---

## Question 7: Can we identify early linguistic markers that predict high-engagement posts?

**Analysis Type:** NLP

**Technical Approach:**
- Loaded cleaned submissions and tokenized clean_text using "@"-delimited tokens.

- Labeled posts as high vs. low engagement using the median of num_comments.

- Performed token explosion and computed word frequencies with groupBy("engagement", "word").count().

- Applied window functions (row_number()) to extract the top 50 frequent words for each engagement group.

- Engineered linguistic features: word_count, unique_ratio, question_marks.

- Aggregated group-level statistics and produced top-word plots comparing high vs. low engagement.

---

## Question 8: Can we predict whether a Reddit comment will be “controversial” based on its comment text and metadata?

**Analysis Type:** ML

**Technical Approach:**

* Construct binary controversiality label directly from Reddit metadata (`controversiality` column), where
  **1 = controversial comment** and **0 = non-controversial**.

* Engineer linguistic and behavioral features from each comment, including:

  * `body_length` (character count)
  * `punctuation_count` (presence of “?” or “!” as emotional indicators)
  * `is_top_level` (whether comment is a "direct reply" to submission)
  * `gilded` (whether comment received awards)
  * `hour` (posting time extracted from `created_utc`)
  * Additional lexical signals such as negativity keywords (e.g., “toxic”, “abusive”, “gaslight”)

* Extract metadata features: comment score (upvotes), thread depth inferred from parent-child structure, and engagement indicators (e.g., gilded = social reward). These features are combined with linguistic cues to form a unified representation.

* Assemble all engineered features using `VectorAssembler` to create a single model input column.

* Split the dataset 80/20 and build a Spark ML Pipeline with assembler + classifier (`LogisticRegression`).

* Train and evaluate models using `BinaryClassificationEvaluator` with AUC-ROC as the main metric. Extract logistic regression coefficients to interpret directional effects and use tree-based feature importance for nonlinear models.

* Expected outcome: Develop a predictive model (target AUC > 0.65) that identifies comment traits associated with high controversy—e.g., emotionally charged punctuation, awards (gilded), short but opinionated comments, and deeper thread-level interactions. These insights help characterize which linguistic and behavioral cues trigger disagreement on r/relationship_advice.


---

## Question 9: Can we predict whether a Reddit relationship post will receive high engagement (top 25% comment count)?

**Analysis Type:** ML

**Technical Approach:**

- Define the prediction task as a binary classification problem where the goal is to determine whether a Reddit relationship post belongs to the top 25% most commented posts. The binary target variable label_high_engagement is created using a quantile threshold on num_comments.

- Load cleaned submissions and comments data from S3, then aggregate lightweight comment-level signals (e.g., mean/max/sum comment score, comment length features, unique commenters, controversiality scores) using groupBy(link_id) to capture early conversational dynamics surrounding each post.

- Engineer lightweight yet effective text and metadata features from the submission text (clean_text) including:
  - Bag-of-Words (CountVectorizer, vocab size 10k)
  - Token count, sentence count
  - Whether the post contains a question mark
  - Time-based features (posting hour, day of week, weekend indicator)

- Build a Spark ML pipeline combining:
  - RegexTokenizer → tokenization
  - StopWordsRemover → stopword filtering
  - CountVectorizer → bag-of-words vectorization
  - VectorAssembler → combine numeric + BOW features
  - StandardScaler → feature scaling
  - LogisticRegression (with class weights) → final classifier

- Address class imbalance by computing class weights based on training-set label distribution and applying them directly inside Logistic Regression (weightCol). This avoids oversampling overhead and improves minority-class recall (high-engagement posts).

- Train the model with CrossValidator using a small but effective hyperparameter grid (regularization strength, elasticNet mix), optimizing for areaUnderROC and ensuring consistent performance across folds.

- Evaluate the model using:
  - ROC curve
  - AUC, Accuracy, F1 Score
  - Confusion matrix
  - Positive-class probability distribution plots

- Expected outcome: Build a lightweight, scalable model capable of identifying high-engagement posts with balanced recall and precision (AUC ≈ 0.65+). This enables early detection of potentially viral discussions, informing content prioritization, recommendation logic, and moderation workflows.

---

## Question 10: Can we predict the daily posting volume of r/relationship_advice using temporal and engagement-driven signals?

**Analysis Type:** ML

**Technical Approach:**

- Formulate the problem as a supervised regression task predicting the next day's post volume (n_posts) using time-dependent signals rather than fitting a traditional forecasting model.

- Engineer a comprehensive set of lag features (lag-1, lag-2, lag-3, lag-7, lag-14, lag-21, lag-28) to capture short-term momentum, weekly cycles, and multi-week posting rhythms.

- Add rolling window statistics (7-day, 14-day, and 30-day rolling means; 7-day and 14-day rolling standard deviations) to encode local posting trends and volatility.

- Include calendar-based seasonality features (day_of_week, month, day_of_year, is_weekend) and smooth Fourier terms to capture weekly/annual periodicity not handled by raw calendar indicators.

- Combine all predictors using a VectorAssembler + StandardScaler inside a Spark ML pipeline.

- Train multiple models (Random Forest, Ridge Regression, ElasticNet, and XGBoost) to compare learning capacity.

- Evaluate models using RMSE, MAE, and R² on a chronologically-held-out test set, ensuring no leakage across time.

- Extract Random Forest feature importances to understand which temporal signals most strongly drive posting volume.

- Expected outcome: Develop a scalable model (target R² >=0.5) that captures stable posting rhythms and enables early workload estimation for moderation and downstream analytics.

## Summary

**EDA Questions:** 1, 2, 3 (3 questions)

**NLP Questions:** 4, 5, 6, 7 (4 questions)

**ML Questions:** 8, 9, 10 (3 questions)

These questions span multiple analysis techniques and provide comprehensive insights into emotional dynamics, communication patterns, and relationship conflict types in online communities. Each question leverages distributed big data processing on AWS EC2 Spark clusters to analyze millions of Reddit posts and comments efficiently:

- **PySpark DataFrames** for scalable data manipulation and aggregation

- **Spark NLP** for distributed text processing, sentiment analysis, and feature extraction

- **Spark MLlib** for distributed machine learning model training and evaluation

- **S3** for data storage and intermediate results persistence
