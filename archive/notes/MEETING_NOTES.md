# Spotify Data Mining — Project Briefing
**CISC 4631 | Group 3 | Updated 2026-04-22**

> **Post-third-round-meeting update (2026-04-22):** pipeline revised per Yanjun's feedback —
> Rock restored via random sampling (not dropped), all notebooks now use a 60/20/20 train/eval/test
> split, mutual-information feature selection added, and a class-balance check with conditional
> SMOTE added to Nb 02. See `THIRD_ROUND_MEETING.md` for the meeting record.

---

## The Big Picture

We have a dataset of **550,000 Spotify songs** with audio measurements for each song (how danceable it is, how loud, how fast, etc.) plus a popularity score from 0–100. The project asks: **can a computer learn patterns in those audio measurements well enough to predict genre or popularity?**

We split the work into four notebooks that run in order.

---

## Notebook 00 — `00_data_setup.ipynb`
### "The Kitchen"

**What it does:** Loads the raw data, explores it, cleans it up, and saves two ready-to-use datasets to Google Drive. The other two notebooks pull from those saved files — they never touch the raw 550k-row dataset.

### Step by step

**1. Load**
Two files come from Kaggle:
- `artists.csv` — one row per artist, with follower count and popularity
- `songs.csv` — one row per song, with 10 audio measurements + genre + year + popularity

We call these `dfA` (artists) and `dfS` (songs) throughout.

**2. Genre consolidation**
The original dataset had 10 genres, several of which overlap heavily in sound. We merged similar ones:

| Original | Merged into |
|----------|-------------|
| Hip-Hop + R&B | Hip-Hop/R&B |
| Country + Folk | Country/Folk |
| Jazz + Blues | Jazz/Blues |

This leaves **7 genres** total. Merging makes sense because audio features can't reliably distinguish, say, a Folk song from a Country song — they sound nearly identical to a computer.

**3. Artist EDA (Exploratory Data Analysis)**
We look at the artist data before touching songs:
- Distribution of follower counts (extremely skewed — a few artists have billions of followers)
- Popularity by genre
- **Artist segmentation:** we split every artist into one of four quadrants based on followers vs. popularity:
  - **Mainstream Giant** — high followers, high popularity (think Taylor Swift)
  - **Rising Star** — low followers, high popularity (buzzing but not huge yet)
  - **Fading Icon** — high followers, low popularity (legacy artists whose streams have dried up)
  - **Underground Gem** — low followers, low popularity (niche, cult artists)

**4. Song EDA**
Nine sections of exploration on the songs:
- Popularity is heavily skewed — most songs sit near 0. About 40% have popularity ≤ 5.
- Songs per year shows exponential growth peaking around 2020.
- Rock dominates at ~35% of all songs — way more than any other genre.
- Audio features have very weak linear correlation with popularity (none above 0.15). This tells us we'll need non-linear models.
- Feature–popularity relationships differ by genre — danceability predicts popularity in Hip-Hop but barely matters in Classical.

**5. Year filter**
We keep only songs from **2000 onward**. Spotify's popularity score is based on recent streaming activity, so pre-2000 songs are systematically underscored — not because they're bad, but because fewer people stream them today.

**6. Export Dataset A — for genre classification**
- Keep all 7 genres including Rock
- Take up to **5,000 songs per genre** (randomly sampled) — all 7 genres contribute equally without Rock's volume overwhelming the classifier. Total: **35,000 rows**.
- Save as `df_genre_balanced.csv` → used by Notebooks 01 and 03

**7. Export Dataset B — for popularity prediction**
- Assign each song a **global popularity class**: Low (score ≤ 20), Mid (21–45), High (≥ 46)
- Assign each song a **genre-relative popularity class**: compare the song only to other songs in its genre — a Classical song at 30 is a standout; a Pop song at 30 is average
- Take **500 songs per (class × genre) combination** so classes are balanced
- Save as `df_popularity_stratified.csv` → used by Notebook 02

---

## Notebook 01 — `01_genre_classification.ipynb`
### "Can audio features predict genre?"

**Research Question:** If I give a computer the audio measurements of a song — but don't tell it the genre — can it figure out what genre it is?

**What it loads:** `df_genre_balanced.csv` (the balanced 7-genre dataset from Notebook 00)

**The features:** We start with the 12 audio measurements:
`danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, key, mode`

**Feature selection:** Mutual information (computed on the training set only) ranks each feature's non-linear dependence with genre. Bottom-scoring features are dropped — typically `key` and `mode`, which contribute almost nothing. `K` is a tunable knob in the notebook.

**The split:** 60/20/20 train/eval/test, stratified by genre.
- **Train (60%)** — the model fits on this, plus 10-fold cross-validation for stability estimates.
- **Eval (20%)** — held out during training; used for model comparison and hyperparameter decisions.
- **Test (20%)** — touched exactly once at the end for the final reported numbers. Never used for tuning.

### The two models

**Model 1 — Logistic Regression (baseline)**
- Think of this as drawing straight lines to divide genres in the feature space
- It's the "simple" model — we use it as a floor to beat
- Includes StandardScaler (automatically normalizes all features to the same scale before training)
- Also runs **10-fold cross-validation**: splits the data 10 different ways and averages the accuracy — gives a more reliable score than a single test

**Model 2 — Random Forest**
- Builds 100 decision trees, each trained on a random subset of songs and features
- Each tree votes, and the majority wins — this is called ensemble learning
- Much better at capturing non-linear relationships (genre boundaries aren't straight lines)
- Also runs 10-fold cross-validation

### What we report
- Accuracy + classification report for each model on the test set
- Cross-validation score in class format: `CV Accuracy: 0.XX (+/- 0.XX)`
- Side-by-side bar chart of Accuracy, Macro F1, Weighted F1
- Random Forest feature importance chart (which audio features mattered most?)
- Random Forest confusion matrix (which genres get mixed up?)

### What we expect to find
- Random Forest should significantly outperform Logistic Regression — genre is a non-linear problem
- Hip-Hop/R&B and Electronic should be the easiest genres to identify (very distinct sound profiles)
- Classical vs. Jazz/Blues might be the hardest (both acoustic, low energy, often instrumental)
- Baseline random-chance accuracy for a 7-class balanced problem = **14.3%** — both models should far exceed this

---

## Notebook 02 — `02_popularity_prediction.ipynb`
### "Can audio features predict whether a song will be popular?"

**Research Questions:**
1. Can audio features predict whether a song is globally popular?
2. Can audio features predict whether a song is popular *within its own genre*?
3. Do the same features drive both, or does genre context change what matters?

**What it loads:** `df_popularity_stratified.csv` (the stratified popularity dataset from Notebook 00)

**Two targets (this is the key twist):**
- `y_global` — Low / Mid / High based on fixed score thresholds (same cutoff for every genre)
- `y_genre` — Low / Mid / High based on where the song ranks *within its genre* (a Classical song at 30 = High; a Pop song at 30 = Mid)

Both targets use the same selected features and the same 60/20/20 train/eval/test split, stratified on the global label.

**Class balance check + SMOTE:** After splitting, we check the imbalance ratio (max class count / min class count) of each target. If it exceeds 1.5×, we apply SMOTE on the training set only (never on eval/test). In practice, Nb 00's stratified sampling usually keeps classes balanced, so SMOTE won't fire — but we run the check so we can show Yanjun we verified.

**Feature selection:** Mutual information against the primary target (`y_global`) on train only. Same selected features are used for both targets and all models.

### The models

**Baseline classifiers** (tested on both targets):

| Model | What it does |
|-------|-------------|
| KNN (k=5) | Finds the 5 most similar songs in the training set and takes a majority vote |
| Decision Tree | Learns a series of yes/no questions about audio features to split songs into classes |
| Naive Bayes | Uses probability — assumes each feature independently contributes to the class |

Each baseline runs through the `evaluate()` function which:
1. Fits the model
2. Runs **10-fold cross-validation** on the training data and prints `CV Accuracy: 0.XX (+/- 0.XX)`
3. Prints a full classification report on the test set
4. Shows a confusion matrix

**Neural Network — PyTorch MLP**
A small neural network with 3 hidden layers (128 → 64 → 32 neurons). More powerful than the baselines but also more complex:
- Trained for 60 epochs with the Adam optimizer
- Uses BatchNorm and Dropout to prevent overfitting
- Learning rate decays every 20 epochs (StepLR)
- Trained and evaluated separately for both targets

### What we report
- CV accuracy + test accuracy for all 4 models × 2 targets = 8 evaluations
- Side-by-side comparison chart of all models on both targets
- MLP learning curves (accuracy over training epochs)
- Final summary table: Accuracy, Macro F1, Weighted F1 for each

### What we expect to find
- Audio features are a **weak predictor** of global popularity (the EDA showed correlations near zero)
- Genre-relative labels might be slightly easier — you're comparing apples to apples within each genre
- Non-linear models (DT, MLP) should outperform Naive Bayes
- If genre-relative accuracy is meaningfully higher than global accuracy → audio features carry genre-context signal
- If they're similar → audio features alone don't capture what makes a song stand out in its genre

---

## Notebook 03 — `03_recommendation_system.ipynb`
### "Can audio features recommend similar songs?"

**Research Question:** Given a query song, can a recommender find other songs with genuinely similar audio — without any listening history?

**What it loads:** `df_genre_balanced.csv` (the balanced 7-genre dataset from Notebook 00)

**The approach:** Content-based filtering using cosine similarity on audio features, implemented with scikit-learn `NearestNeighbors`. Three versions compare different design choices.

### The three versions

**V1 — Baseline**
- 12 audio features (including `key`, `mode`)
- StandardScaler + cosine similarity, single-stage k-NN

**V2 — Improved Audio**
- **Feature selection:** drop `key` and `mode` (MI < 0.03 against genre — essentially noise)
- **Attribute transformation:** log-transform `tempo` and `duration_ms` to fix right-skew
- Same cosine similarity, still single-stage

**V3 — Hybrid (cross-notebook)**
- V2's retrieval (top-50 audio-similar candidates) **+ Notebook 01's Random Forest as a re-ranker**
- Two-stage retrieve-and-rerank: score each candidate as `α × audio_similarity + (1−α) × (RF-predicted genre matches query)`, then return top-10
- Default α = 0.5 (classifier match dominates within the candidate pool)
- Standard recommender-systems architecture (used by YouTube, search ranking, etc.)

### Evaluation metrics — three, complementary

- **Same-genre rate @10** — fraction of top-10 recommendations matching the query's genre. Primary metric. Random baseline = 14.3% (1/7).
- **Intra-list diversity** — mean pairwise distance among the top-10 recommendations. Higher = less repetitive.
- **Catalog coverage** — fraction of the 35k catalog that ever appears in any top-10 across 500 queries. Higher = less "popularity magnet."

### What we found

| Version | Same-Genre Rate | Diversity | Coverage | Change from V1 |
|---------|----------------|-----------|----------|----------------|
| V1 — baseline | 33.7% | 1.61 | 13.1% | — |
| V2 — improved audio | 33.9% | 2.27 | 13.1% | +0.2pp sgr, +41% diversity |
| **V3 — hybrid (α=0.5)** | **44.2%** | **2.33** | 13.0% | **+10.5pp sgr, +45% diversity** |

- **V1 → V2:** preprocessing barely moves same-genre rate (33-34% appears to be a real audio-feature ceiling) but substantially improves diversity.
- **V2 → V3:** Notebook 01's classifier as a re-ranker adds a real **+10.3pp** on same-genre rate while preserving diversity and coverage.
- **All three versions beat the 14.3% random baseline by 2.4–3.1×.**

### Honest caveats

- Same-genre rate is partially circular as a V3 metric (re-rank by predicted genre, evaluate on true genre — which correlate). Diversity and coverage remain clean across all three versions.
- V3's ceiling is bounded by the RF re-ranker's ~51% test accuracy. A stronger classifier would improve V3.

### Class-topic grounding

- **V1, V2:** Topic 2 — Data Processing and Feature Selection (distance measures, feature selection, attribute transformation)
- **V3:** Topic 5 — Classification (Random Forest, ensemble methods) + retrieve-and-rerank architecture

---

## How to run it

Run the notebooks **in order**:

```
00_data_setup.ipynb               →  creates the two CSV files on Drive
01_genre_classification.ipynb     →  reads df_genre_balanced.csv
02_popularity_prediction.ipynb    →  reads df_popularity_stratified.csv
03_recommendation_system.ipynb    →  reads df_genre_balanced.csv; uses Nb 01's RF as re-ranker
```

All four notebooks mount Google Drive at the top. The shared Drive folder is:
`My Drive / data-mining-spotify-team3 / cleanedData /`

You only need to re-run `00` if you change the data (genre merges, year cutoff, sample sizes). For modeling experiments, just re-run `01`, `02`, or `03` directly.

---

## Quick reference — shared constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `SEED` | 42 | Random seed — keeps results reproducible |
| `YEAR_CUTOFF` | 2000 | Drop songs before this year |
| `DRIVE_DATA_PATH` | `.../cleanedData` | Where CSVs are saved/loaded |
| `AUDIO_FEATURES` | 10 features | Continuous audio measurements |
| `KEY_FEATURES` | key, mode | Musical key (not very useful — included for completeness) |
| `ALL_FEATURES` | 12 total | Everything fed to the models |
| `LABEL_MAP` | Low=0, Mid=1, High=2 | Converts text class labels to numbers for the models |
