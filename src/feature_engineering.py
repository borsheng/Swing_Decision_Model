import pandas as pd
import numpy as np

# ----------------------------
# Feature configuration
# ----------------------------
# Numeric pre-contact features (original pitch/tracking + count/base/out + new)
numeric_features = [
    # Pitch/tracking
    'release_speed', 'plate_x', 'plate_z',
    'release_pos_x', 'release_pos_z', 'release_extension',
    'pfx_x', 'pfx_z',
    # Count & context
    'balls', 'strikes', 'outs_when_up',
    'runners_on', 'score_diff', 'pre_run_exp',
    # Lag strike features
    'is_strike', 'strike_streak',
    # Inning number 
    'inning'
]

# Categorical pre-contact features (original + new)
categorical_features = [
    'pitch_type', 'stand', 'p_throws', 'zone', 'inning_topbot',
    'prev_event'
]

# ----------------------------
# Outcome mapping (label)
# ----------------------------
def outcome_value(row):
    # (same as before)
    desc    = row['description']
    speed   = row.get('launch_speed', np.nan)
    angle   = abs(row.get('launch_angle', np.nan))
    strikes = row['strikes']
    zone    = row.get('zone', 5)
    in_zone = 1 <= zone <= 9

    if desc == 'hit_into_play':
        if pd.isna(speed) or pd.isna(angle):
            return 0.0
        if speed > 95 and 10 <= angle <= 30:
            return 0.45 if in_zone else 0.20
        if speed > 85 and 10 <= angle <= 30:
            return 0.25 if in_zone else 0.05
        return 0.05 if in_zone else 0.00

    if desc == 'swinging_strike':
        return -0.15 if in_zone else -0.30

    if desc in ('foul', 'foul_tip'):
        return -0.05 if strikes < 2 else 0.0

    if desc == 'called_strike':
        return -0.05

    if desc == 'ball':
        return +0.10

    return 0.0


# ----------------------------
# Main feature engineering
# ----------------------------
def feature_engineering(raw_df: pd.DataFrame, re_table: pd.DataFrame) -> pd.DataFrame:
    """
    Produce model-ready DataFrame with:
      - drop / impute bad data
      - recompute runners_on, score_diff, pre_run_exp
      - correct prev_event shifting description
    """
    df = raw_df.copy()

    # -- 1. Label --
    df['decision_value'] = df.apply(outcome_value, axis=1)

    # -- 2. Simple features --
    # Drop any raw columns that might collide
    for col in ['runners_on', 'score_diff', 'pre_run_exp', 'base_state', 'prev_event']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # recompute
    df['runners_on'] = df[['on_1b', 'on_2b', 'on_3b']].sum(axis=1)
    df['score_diff'] = np.where(
        df['inning_topbot'] == 'Top',
        df['away_score'] - df['home_score'],
        df['home_score'] - df['away_score']
    )

    # -- 3. Base–Out → run expectancy --
    df['base_state'] = (
        df['on_1b'].fillna(0).astype(int).astype(str) +
        df['on_2b'].fillna(0).astype(int).astype(str) +
        df['on_3b'].fillna(0).astype(int).astype(str)
    )

    # prepare run-expectancy table
    re_merge = re_table.rename(columns={'outs': 'pre_outs', 'run_exp': 'pre_run_exp'}).copy()
    re_merge['pre_outs'] = re_merge['pre_outs'].astype(int)
    df['outs_when_up'] = df['outs_when_up'].astype(int)

    df = df.merge(
        re_merge[['base_state', 'pre_outs', 'pre_run_exp']],
        left_on=['base_state', 'outs_when_up'],
        right_on=['base_state', 'pre_outs'],
        how='left'
    ).drop(columns=['pre_outs'])

    # drop any rows where we still lack a run-expectancy
    df = df.dropna(subset=['pre_run_exp'])

    # -- 4. Lag features --
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])

    # use previous pitch's description rather than a nonexistent 'events' column
    df['prev_event'] = (
        df.groupby(['game_pk', 'at_bat_number'])['description']
          .shift(1)
          .fillna('start')
    )

    df['is_strike'] = df['description'].isin(
        ['called_strike', 'swinging_strike']
    ).astype(int)
    df['strike_streak'] = (
        df.groupby(['game_pk', 'at_bat_number'])['is_strike']
          .cumsum() - df['is_strike']
    )

    # -- 5. Final selection --
    model_cols = numeric_features + categorical_features + ['decision_value']

    df['batter'] = raw_df['batter']

    return df[model_cols + ['batter']]


# ----------------------------
# Script entrypoint
# ----------------------------
if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw/statcast_2022_2024.csv")
    re_table = pd.read_csv(
        "data/raw/run_expectancy.csv",
        dtype={"base_state": str}
    )
    proc_df = feature_engineering(raw_df, re_table)
    proc_df.to_csv("data/processed/statcast_features.csv", index=False)
