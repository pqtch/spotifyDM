# Spotify Data Mining — CISC 4631 | Group 3

Audio-feature-based analysis of the Kaggle 550k Spotify Songs dataset. Four pipelined notebooks: data prep + EDA, genre classification, popularity prediction, and a cross-notebook recommender that consumes the trained classifiers.

---

## Current state (as of 2026-04-23)

| Notebook | What it does | Output |
|---|---|---|
| [00_data_setup.ipynb](00_data_setup.ipynb) | Load, clean, EDA, export two sampled CSVs | `df_genre_balanced.csv` (48,265 songs) + `df_popularity_stratified.csv` (10,500 songs) |
| [01_genre_classification.ipynb](01_genre_classification.ipynb) | 7-class genre classifier (LR, RF, XGBoost) | Pickled XGBoost model + label encoder + feature list |
| [02_popularity_prediction.ipynb](02_popularity_prediction.ipynb) | 4 popularity targets × 4 models (KNN, DT, NB, XGBoost) | Pickled XGBoost binary≥median model + scaler + feature list + threshold |
| [03_recommendation_system.ipynb](03_recommendation_system.ipynb) | Content-based recommender V1 → V2 → V3 → V4, loads pickles from Nb 01 and Nb 02 | Evaluation metrics + demo (no artifacts written) |

All four notebooks run end-to-end on Colab. The pipeline is fully wired.

---

## Run order (Colab)

1. **`00_data_setup.ipynb`** — required first. Writes the two cleaned CSVs to Drive.
2. **`01_genre_classification.ipynb`** — reads `df_genre_balanced.csv`. Writes genre pickles to Drive.
3. **`02_popularity_prediction.ipynb`** — reads `df_popularity_stratified.csv`. Writes popularity pickles to Drive.
4. **`03_recommendation_system.ipynb`** — reads `df_genre_balanced.csv` + both sets of pickles. No artifacts written.

Dependencies: `1 → 2`, `1 → 3`, and `(1 + 2 + 3) → 4`. Notebooks 2 and 3 can run in parallel after 1. Notebook 4 requires 1, 2, and 3 to have written their artifacts.

---

## Environment

- **Platform**: Google Colab (not local Python). Each notebook starts with `drive.mount('/content/drive')`.
- **Shared Drive folder**: `My Drive/data-mining-spotify-team3/`
  - `cleanedData/` — CSV outputs from Nb 00
  - `models/` — pickle outputs from Nb 01 and Nb 02
- **Raw dataset**: Kaggle `serkantysz/550k-spotify-songs-audio-lyrics-and-genres`, loaded via `kagglehub` in Nb 00.
- **Shared constants** (defined in Nb 00, redefined in each downstream notebook to stay in sync):
  - `SEED = 42`
  - `YEAR_CUTOFF = 2000` (post-2000 songs only — Spotify popularity scoring is streaming-era biased)
  - `DRIVE_DATA_PATH = '/content/drive/MyDrive/data-mining-spotify-team3/cleanedData'`
  - `MODEL_PATH = '/content/drive/MyDrive/data-mining-spotify-team3/models'`
  - `AUDIO_FEATURES` = 10 audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms)
  - `KEY_FEATURES = ['key', 'mode']`
  - `ALL_FEATURES = AUDIO_FEATURES + KEY_FEATURES` (12 total)

---

## Data flow

```
Kaggle 550k Spotify songs
      ↓
  Nb 00 filters: year >= 2000 AND popularity > 0 AND non-null audio features
      ↓
  Nb 00 consolidates genres: Hip-Hop+R&B → Hip-Hop/R&B, Country+Folk → Country/Folk, Jazz+Blues → Jazz/Blues (7 genres total)
      ↓
  Nb 00 builds two samples:
    ├── df_genre_balanced.csv      (6,895 × 7 = 48,265 rows)   → used by Nb 01, Nb 03
    └── df_popularity_stratified.csv (500 × 3 × 7 = 10,500 rows) → used by Nb 02
      ↓
  Nb 01 trains genre classifier on 48k balanced → pickle
  Nb 02 trains popularity classifier on 10.5k stratified → pickle
      ↓
  Nb 03 loads 48k + both pickles, builds V1 → V2 → V3 → V4 recommender
```

---

## Key results (test-set numbers)

### Nb 01 — Genre Classification (7-class, balanced)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 43.6% | 0.42 |
| Random Forest | 50.1% | 0.49 |
| **XGBoost** | **51.2%** | **0.51** |

Random baseline = 14.3% → XGBoost is **3.58× baseline**.

Per-class F1 (XGBoost): Rock 0.64, Electronic 0.62, Hip-Hop/R&B 0.61, Classical 0.54, Jazz/Blues 0.50, Country/Folk 0.37, **Pop 0.27** (floor — Pop is audio-heterogeneous).

### Nb 02 — Popularity Prediction

| Target | Best Model | Accuracy | Baseline | Real Lift |
|---|---|---|---|---|
| Global 3-class | XGBoost | 39.2% | 33.3% | +5.9pp ✓ |
| Genre-relative 3-class | XGBoost | 45.9% (Macro F1 0.38) | 33.3% | inflated by majority bias |
| Binary ≥ High (≥46) | XGBoost | 63.7% | **66.7% majority** | **−3pp ✗** (accuracy paradox) |
| **Binary ≥ median** | **XGBoost** | **55.2%** | **50%** | **+5.2pp ✓** (cleanest signal) |

### Nb 03 — Recommender (500 held-out queries from Nb 01's test set, α=0.5)

| Version | Same-Genre Rate | Diversity | Coverage |
|---|---|---|---|
| V1 baseline | 34.7% | 1.59 | 9.8% |
| V2 improved audio (drop key/mode + log-transform) | 34.8% | 2.30 | 9.8% |
| **V3 hybrid (+ Nb 01 genre classifier)** | **46.3%** | 2.34 | 9.7% |
| V4 hybrid (+ Nb 02 popularity classifier) | 43.4% | 2.34 | 9.7% |

**V3 → V4 is −2.9pp.** V4 is intentionally shown as worse — see "Design decisions" #6 below.

---

## Design decisions that must not be undone

### 1. Sample sizes are asymmetric by design
Nb 01 and Nb 03 use `df_genre_balanced.csv` (48,265 songs, 6,895 per genre × 7 genres). Nb 02 uses `df_popularity_stratified.csv` (10,500 songs, 500 per popularity-class × genre). This is because:
- Genre classification needs balanced *genres*.
- Popularity prediction needs balanced *popularity bins*.

The two cannot coexist in a single sample. Do not try to unify them.

### 2. Nb 03 uses TWO feature spaces
- **Retrieval** (V2): 10 features, log-transformed on `tempo` and `duration_ms`, then StandardScaler-fit → cosine similarity via `NearestNeighbors`.
- **Classifier re-ranking**: 12 **raw** features (unscaled), matches what Nb 01 trained on.

This is deliberate. Retrieval benefits from normalization (cosine similarity over scaled vectors); tree classifiers don't (trees split on raw values, so preprocessing just adds noise). Forcing a single feature space would either hurt retrieval diversity or force Nb 01 to retrain on V2's 10-feature space (breaking the standalone genre-classification story).

### 3. Nb 03 loads pickles from Nb 01 and Nb 02 — it does NOT retrain inline
- `§11.1` loads `genre_xgb_model.pkl` + `genre_label_encoder.pkl` + `genre_feature_list.pkl` from `MODEL_PATH`.
- `§12.1` loads `pop_xgb_binary_p50.pkl` + `pop_scaler.pkl` + `pop_feature_list.pkl` + `pop_median_threshold.pkl`.

If you see inline training in Nb 03, it's a regression — the pipeline story (Nb 01 and Nb 02 produce reusable artifacts consumed by Nb 03) breaks. A previous draft of `§11` trained a fresh Random Forest inline. That has been removed.

### 4. Nb 01 keeps all 12 features, even though MI suggests dropping 2
MI is a univariate filter; it cannot see interaction signal. Empirically, dropping `liveness` and `key` cost ~1pp test accuracy because XGBoost extracts interaction value from them (e.g., XGBoost assigns `mode` ~10% importance despite MI ≈ 0.02). The `K=12` line in `§3 Feature Selection` is deliberate — MI is used as a *ranking diagnostic*, not a *filter*.

### 5. Nb 02's `Binary ≥ High` target underperforms the majority baseline
63.7% accuracy vs. 66.7% majority — XGBoost has learned to say "Not-Popular" and ride the class imbalance (70% of actual Popular songs are mislabeled). We kept this result because it's a presentation-worthy example of the accuracy paradox. **Do not "fix" it by rebalancing or dropping the target.**

### 6. V4 < V3 is intentional — don't "fix" it
Adding Nb 02's weak popularity signal (~55% accuracy) to V3's stronger genre signal dilutes the re-ranker. This is the key presentation finding: *weak auxiliary signals can hurt when combined naively with a strong primary signal.* Validates Nb 02's standalone conclusion that audio features carry limited popularity signal. The notebook's `§12.4` explains this explicitly.

### 7. Nb 03 evaluates on queries drawn from Nb 01's held-out test set
Nb 03's `§11.1` recreates Nb 01's 80/20 stratified split using the same `SEED=42` to obtain the same `test_idx`. Evaluation queries come from `test_idx` only — queries the Nb 01 classifier has never seen during training. This prevents circular evaluation. Do not evaluate V3/V4 on queries that were in Nb 01's training set.

---

## Known non-issues (look wrong, aren't)

- **Nb 02's scaler is fit on 10.5k songs, applied in Nb 03 to 48k songs.** Both samples are random draws from the same post-2000 Spotify population, so means/stds are nearly identical. Defensible; documented in Nb 03 `§12.1` inline comments.
- **Nb 03 `§10.1` runs MI on the whole dataset, not train-only.** This is a diagnostic plot informing V2's feature-selection decision, not feature selection for a supervised model. MI leakage is not a concern here.
- **Nb 03 `§10.5` and `§11.5` tables show slightly different V1/V2 numbers.** `§10.4` evaluates on 500 random queries drawn from all 48k songs. `§11.3` evaluates on 500 queries drawn from `test_idx` (the held-out 20%). Different query samples → slightly different numbers. Not a bug.
- **Nb 01 and Nb 02 both use `SEED=42`, but their train/test splits differ.** Nb 01 uses a 3-way 60/20/20 split; Nb 02 uses the same 3-way split but on a different dataset. The `SEED` determines the reshuffling, which will produce different indices because the dataframes are different sizes.

---

## File map

```
spotifyDM/
├── 00_data_setup.ipynb
├── 01_genre_classification.ipynb
├── 02_popularity_prediction.ipynb
├── 03_recommendation_system.ipynb
├── README.md  (this file)
└── archive/
    └── notes/   (meeting notes — not part of the pipeline)

Drive/My Drive/data-mining-spotify-team3/
├── cleanedData/
│   ├── df_genre_balanced.csv         (from Nb 00; 48,265 rows × 14 cols)
│   └── df_popularity_stratified.csv  (from Nb 00; 10,500 rows × 17 cols)
└── models/
    ├── genre_xgb_model.pkl            (Nb 01 — XGBClassifier, trained on 12 raw features)
    ├── genre_label_encoder.pkl        (Nb 01 — sklearn LabelEncoder)
    ├── genre_feature_list.pkl         (Nb 01 — list of 12 feature names, MI-ranked)
    ├── pop_xgb_binary_p50.pkl         (Nb 02 — XGBClassifier, binary ≥ training median)
    ├── pop_scaler.pkl                 (Nb 02 — StandardScaler fit on 10.5k songs × 12 features)
    ├── pop_feature_list.pkl           (Nb 02 — list of 10 feature names, drops tempo + duration_ms)
    └── pop_median_threshold.pkl       (Nb 02 — float, = 31.0)
```

---

## Team

- Patch
- Paromita
- Nick

## Dates

- **Progress Presentation**: 2026-04-24 (Friday, 1:25 PM)
- **Final Deliverable**: 2026-05-08
