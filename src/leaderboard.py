import os
import pandas as pd
import joblib
import xgboost as xgb
from pybaseball import playerid_reverse_lookup

# numeric_features（train_model.py / leaderboard.py / player_analysis.py）
numeric_features = [
    'release_speed', 'plate_x', 'plate_z',
    'release_pos_x', 'release_pos_z', 'release_extension',
    'pfx_x', 'pfx_z',
    'balls', 'strikes', 'outs_when_up',
    'runners_on', 'score_diff', 'inning',
    'pre_run_exp', 'is_strike', 'strike_streak'
]

# categorical_features
categorical_features = [
    'pitch_type', 'stand', 'p_throws', 'zone',
    'inning_topbot', 'prev_event'
]

def compute_leaderboard(
    df: pd.DataFrame,
    model: xgb.XGBRegressor,
    preprocessor,
    min_pitches: int = 1000,
    top_n: int = 10
) -> None:
    """
    Generate top and bottom batting leaderboards based on mean decision value.

    Steps:
    1. Predict decision values for all pitches in the DataFrame.
    2. Filter out batters with fewer than min_pitches seen.
    3. Compute mean prediction per batter, scaled to per 100 pitches.
    4. Save the top_n and bottom_n batters to CSV files in outputs/.
    """
    df_work = df.copy()

    # 1) Predict on all data
    X_all = df_work[numeric_features + categorical_features]
    df_work['pred'] = model.predict(preprocessor.transform(X_all))

    # 2) Identify batters with enough pitches
    pitch_counts = df_work['batter'].value_counts()
    valid_batters = pitch_counts[pitch_counts >= min_pitches].index

    # 3) Compute each valid batter's mean decision value, scale and round
    mean_preds = (
        df_work[df_work['batter'].isin(valid_batters)]
        .groupby('batter')['pred']
        .mean()
    ) * 100
    mean_preds = mean_preds.round(2)

    # 4) Determine top and bottom performers
    top_batters = mean_preds.nlargest(top_n).index
    bottom_batters = mean_preds.nsmallest(top_n).index

    # 5) Map ID to batter name
    name_map = df.drop_duplicates(subset=['batter']) \
                 .set_index('batter')['batter_name']

    # 6) Build DataFrames and save
    top_df = pd.DataFrame(
        [(name_map[i], mean_preds[i]) for i in top_batters],
        columns=['batter_name', 'mean_pred']
    )
    bottom_df = pd.DataFrame(
        [(name_map[i], mean_preds[i]) for i in bottom_batters],
        columns=['batter_name', 'mean_pred']
    )

    os.makedirs('outputs', exist_ok=True)
    top_df.to_csv("outputs/top10.csv", index=False)
    bottom_df.to_csv("outputs/bottom10.csv", index=False)

if __name__ == "__main__":
    # Load processed dataset
    df = pd.read_csv("data/processed/statcast_features.csv")

    # Reverse-lookup batter names and merge into df
    batter_ids = df['batter'].unique().tolist()
    lookup = playerid_reverse_lookup(batter_ids, key_type='mlbam')
    lookup['batter_name'] = lookup['name_first'] + ' ' + lookup['name_last']
    df = df.merge(
        lookup[['key_mlbam', 'batter_name']],
        left_on='batter', right_on='key_mlbam', how='left'
    )

    # Load model and preprocessor
    model = xgb.XGBRegressor()
    model.load_model("models/xgb_model.json")
    preproc = joblib.load("models/preprocessor.joblib")

    # Generate leaderboard CSVs
    compute_leaderboard(df, model, preproc)
