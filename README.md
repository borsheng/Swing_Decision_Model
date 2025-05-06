# MLB Swing Decision Analysis

**Purpose & Motivation**  
Every hitter faces a constant trade-off: swing at a borderline pitch or take the chance to work a better count?  Our **Swing Decision Model** quantifies that choice by learning from three seasons of Statcast data (2022–2024) — producing a `decision_value` score that measures how good or bad a player’s swing decision is in each balls–strikes count.  This empowers analysts and coaches to:

- **Compare hitters** on an apples-to-apples basis: Who makes the best decisions in 0–2, 3–1, etc.?  
- **Identify count-specific strengths and weaknesses**, tailoring practice plans or in-game coaching.  
- **Extend to high-leverage contexts** (late innings, close scores, runners in scoring position) to uncover clutch performance.  

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Getting Started](#getting-started)  
3. [Data](#data)  
4. [Pipeline & Scripts](#pipeline--scripts)  
5. [Interactive Notebook](#interactive-notebook)  
6. [Usage Examples](#usage-examples)  
7. [Outputs](#outputs)  
8. [Requirements](#requirements)  
9. [Future Work](#future-work)  
10. [License](#license)  

---

## Project Structure

```bash

SWING_DECISION_MODEL/
├─ data/
│  ├ raw/
│  │  ├ statcast_2022_2024.csv       # Raw pitch data
│  │  └ run_expectancy.csv            # Base–out run expectancy lookup
│  └ processed/
│     └ statcast_features.csv         # Model-ready features
├─ models/
│  ├ preprocessor.joblib              # Saved preprocessing pipeline
│  └ xgb_model.json                   # Trained XGBoost model
├─ notebook/
│  └ Swing_Decision_Analysis.ipynb    # End-to-end Notebook
├─ outputs/
│  ├ feature_importance.png           # Top-20 feature importances
│  ├ top10.csv / bottom10.csv         # League leaderboard
│  └ {PlayerName}.png                 # Per-player decision plots
├─ src/
│  ├ data_loading.py                  # Fetch raw Statcast data
│  ├ feature_engineering.py           # Compute features & `decision_value`
│  ├ train_model.py                   # Train & save pipeline + model
│  ├ evaluate.py                      # Plot feature importances
│  ├ leaderboard.py                   # Generate top/bottom 10 CSVs
│  └ player_analysis.py               # CLI for per-player analysis
├─ requirements.txt                   # Python dependencies
└─ README.md                          # This file
```

---

## Getting Started

### Prerequisites

- Python 3.8+  
- Git  

### Installation

```bash
git clone https://github.com/borsheng/SWING_DECISION_MODEL.git
cd SWING_DECISION_MODEL
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

---

## Data

### Raw Data

* **`statcast_2022_2024.csv`**
  Pitch-by-pitch Statcast data for 2022–2024, fetched via `src/data_loading.py`.

* **`run_expectancy.csv`**
  Empirical run-expectancy table mapping base/out states to expected runs.

### Processed Data

* **`statcast_features.csv`**
  Combines raw data + lag/context features + computed `decision_value` label.

---

## Pipeline & Scripts

1. **Feature Engineering**

   ```bash
   python src/feature_engineering.py
   ```

   Produces `data/processed/statcast_features.csv`.

2. **Model Training**

   ```bash
   python src/train_model.py
   ```

   * Train/test split
   * `ColumnTransformer` + XGBoost regressor with early stopping
   * Saves `models/preprocessor.joblib` & `models/xgb_model.json`.

3. **Feature Importance**

   ```bash
   python src/evaluate.py
   ```

   Loads the saved pipeline & model, plots top-20 importances.

4. **Leaderboard Generation**

   ```bash
   python src/leaderboard.py
   ```

   * Lookup MLBAM IDs → names
   * Predict all pitches
   * Filter players with ≥1,000 pitches
   * Outputs `top10.csv` & `bottom10.csv`.

5. **Single-Player Analysis (CLI)**

   ```bash
   python src/player_analysis.py --name "Mike Trout"
   ```

   Prints per-count decision differences vs league and saves a plot (e.g. `outputs/Mike_Trout.png`).

---

## Interactive Notebook

Open `notebook/Swing_Decision_Analysis.ipynb` to step through:

* Data loading & caching
* Inline feature engineering
* Model training & serialization
* Inline feature importance, leaderboard & per-player plots
* Narrative commentary & next-steps guidance

---

## Usage Examples

```bash
# Build features & train model
python src/feature_engineering.py
python src/train_model.py

# Inspect importances & leaderboards
python src/evaluate.py
python src/leaderboard.py

# Examine an individual player
python src/player_analysis.py --name "Justin Turner"
```

---

## Outputs

* **`outputs/feature_importance.png`** – Top 20 feature importances.
* **`outputs/top10.csv`, `outputs/bottom10.csv`** – League leaderboards.
* **`outputs/{PlayerName}.png`** – Decision-quality plots by count.

---

## Requirements

Key dependencies:

* `pandas`, `numpy`
* `scikit-learn`
* `xgboost`
* `matplotlib`
* `pybaseball`

See `requirements.txt` for full list.

---

## Future Work

* **Deploy as an interactive web app**  
   Build a Streamlit or Flask application to allow users (coaches, analysts) to input a player name or game situation and visualize decision-quality metrics in real-time.  

* **Implement continuous training pipeline**  
   Automate data ingestion from Statcast, feature engineering, model retraining, and leaderboard updates via CI/CD (GitHub Actions or Airflow).  

* **Extend to situational analysis**
   Analyze decision quality in high-leverage situations (late innings, close score) and other game contexts (e.g., runners in scoring position, playoff games) to uncover deeper performance patterns.  

---



