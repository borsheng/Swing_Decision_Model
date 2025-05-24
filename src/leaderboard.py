import os
import pandas as pd
import joblib
from pybaseball import playerid_reverse_lookup

def compute_leaderboard(
    df: pd.DataFrame,
    pipeline,
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

    # 1) Predict on all data using full pipeline
    feature_cols = pipeline.named_steps['preprocessor'].feature_names_in_
    X_all = df_work[feature_cols]
    df_work['pred'] = pipeline.predict(X_all)

    # 2) Filter batters with sufficient pitch count
    pitch_counts = df_work['batter'].value_counts()
    valid_batters = pitch_counts[pitch_counts >= min_pitches].index

    # 3) Compute per-batter mean prediction
    mean_preds = (
        df_work[df_work['batter'].isin(valid_batters)]
        .groupby('batter')['pred']
        .mean()
    ) * 100
    mean_preds = mean_preds.round(2)

    # 4) Top and bottom N
    top_batters = mean_preds.nlargest(top_n).index
    bottom_batters = mean_preds.nsmallest(top_n).index

    # 5) Map ID to names
    name_map = df.drop_duplicates(subset=['batter']) \
                 .set_index('batter')['batter_name']

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
    df = pd.read_csv("data/processed/statcast_features.csv")

    # Reverse-lookup batter names
    batter_ids = df['batter'].unique().tolist()
    lookup = playerid_reverse_lookup(batter_ids, key_type='mlbam')
    lookup['batter_name'] = lookup['name_first'] + ' ' + lookup['name_last']
    df = df.merge(
        lookup[['key_mlbam', 'batter_name']],
        left_on='batter', right_on='key_mlbam', how='left'
    )

    # Load full pipeline
    pipeline = joblib.load("models/xgb_pipeline.joblib")

    # Generate leaderboard
    compute_leaderboard(df, pipeline)
