# data_preprocessing.py

import pandas as pd
import numpy as np

# Paths for raw and processed data
RAW_PATH = "data/raw/statcast_2022_2024.csv"
PROCESSED_PATH = "data/processed/statcast_preprocessed.csv"

def preprocess_data():
    """
    Preprocess Statcast data for swing-decision feature engineering:
      1. Keep only necessary columns for feature engineering
      2. Drop missing values in critical columns
      3. Fill missing tracking columns with group mean (pitcher, season, pitch_type)
      4. Remove extreme outliers based on domain knowledge
      5. Ensure zone is numeric and drop invalid zone entries
      6. Remove duplicate rows, reset index, and save to CSV
    """

    # Load raw data and parse game_date as datetime
    df = pd.read_csv(RAW_PATH, parse_dates=['game_date'])

    # 1. Keep only the columns required by feature_engineering.py
    keep_cols = [
        # Pre-contact numeric features
        'release_speed', 'plate_x', 'plate_z',
        'release_pos_x', 'release_pos_z', 'release_extension',
        'pfx_x', 'pfx_z',
        'balls', 'strikes', 'outs_when_up',
        'on_1b', 'on_2b', 'on_3b',
        'home_score', 'away_score',
        'inning', 'inning_topbot',
        # Pre-contact categorical features
        'pitch_type', 'stand', 'p_throws', 'zone',
        # Columns needed for label and lag-feature generation
        'description',            # for computing decision_value
        'launch_speed',           # for outcome_value()
        'launch_angle',           # for outcome_value()
        'game_pk', 'at_bat_number', 'pitch_number',
        'batter', 'pitcher', 'game_date'
    ]
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    # 2. Drop rows with missing critical values
    df.dropna(subset=[
        'description', 'pitch_type', 'release_speed',
        'plate_x', 'plate_z'
    ], inplace=True)

    # 3. Add game_year for grouping, then fill missing tracking columns with group mean
    df['game_year'] = df['game_date'].dt.year
    tracking_cols = ['release_pos_x', 'release_pos_z', 'release_extension', 'pfx_x', 'pfx_z']
    for col in tracking_cols:
        df[col] = (
            df.groupby(['pitcher', 'game_year', 'pitch_type'])[col]
              .transform(lambda x: x.fillna(x.mean()))
        )

    # 4. Remove extreme outliers based on domain knowledge
    df = df[
        (df['release_speed'].between(70, 110)) &    # speed between 70–110 MPH
        (df['release_extension'].between(3.0, 8.0)) &  # extension between 3–8 ft
        (df['plate_x'].between(-2.5, 2.5)) &       # plate_x between -2.5 to 2.5 ft
        (df['plate_z'].between(0.0, 6.0))          # plate_z between 0 to 6 ft
    ]

    # 5. Remove duplicate rows and reset index
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save the cleaned data to CSV
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Preprocessed data saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess_data()
