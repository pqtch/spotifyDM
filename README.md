# spotifyDM
# 🎵 Spotify Data Mining Project (CISC 4631)

This project focuses on identifying **hyper-specific subgenres** and predicting song **popularity** using audio features, artist metadata, and Natural Language Processing (NLP) of lyrics.

*subject to change*

---

## 🚀 Overview
Modern recommendation engines often struggle to cluster songs into granular "Micro-Genres." This project leverages the **Spotify Million Song Dataset** to bridge that gap by combining:
- **Supervised Learning:** Predicting song popularity scores (0-100).
- **Unsupervised Learning:** Clustering songs into niche subgenre groups.
*note: plan to investigate others*
---

## 🛠️ Tech Stack & Environment
This project uses **`uv`** for extremely fast, reproducible Python environment management.


### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

### Installation
```bash
# Clone the repository
git clone git@github.com:YourUsername/spotifyDM.git
cd spotifyDM

# Initialize the environment and install dependencies
uv sync
```

---

## 📊 Data Auditing & Preprocessing

### 1. Data Audit (`describeSongs.py`)
Provides summary statistics, missing value reports, and **Outlier Detection** for audio features like:
- `danceability`, `energy`, `loudness`, `tempo`, etc.
- **Correlation Matrix:** Analyzes how features like `loudness` relate to `popularity`.

---

## 🧪 Ongoing Experiments
- [x] Initial Data Auditing &  Report
- [ ] Preprocessing Pipeline Development
- [ ] 

---

## 👥 The Team
- [pqtch]
- [Team Member Names]

---

## 📅 Important Dates
- **Project Report Due:** May 8, 2026
- **Final Presentation:** May 2026