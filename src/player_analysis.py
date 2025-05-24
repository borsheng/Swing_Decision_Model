import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pybaseball import playerid_reverse_lookup

def compute_league_avg(df: pd.DataFrame, pipeline) -> float:
    """
    Compute the league-wide average decision value per 100 pitches.
    """
    X = df[pipeline.named_steps['preprocessor'].feature_names_in_]
    preds = pipeline.predict(X) * 100
    return round(preds.mean(), 2)

def analyze_player(df: pd.DataFrame, pipeline, batter_id: int) -> pd.DataFrame:
    """
    For a given batter ID, returns a DataFrame comparing his
    per-count avg decision value (per 100 pitches) vs. league,
    and saves a scatter plot to outputs/player_{batter_id}.png.
    """
    # 1) Predict league-wide
    X_all = df[pipeline.named_steps['preprocessor'].feature_names_in_]
    df['pred'] = pipeline.predict(X_all) * 100

    # 2) Compute league mean by count
    league_by_count = df.groupby(['balls', 'strikes'])['pred'].mean()

    # 3) Subset this batter and compute his mean by count
    sub = df[df['batter'] == batter_id].copy()
    X_sub = sub[pipeline.named_steps['preprocessor'].feature_names_in_]
    sub['pred'] = pipeline.predict(X_sub) * 100
    player_by_count = sub.groupby(['balls', 'strikes'])['pred'].mean()

    # 4) Combine into DataFrame
    comp = pd.DataFrame({
        'player': player_by_count,
        'league': league_by_count
    }).dropna()
    comp['diff'] = (comp['player'] - comp['league']).round(2)

    batter_name = df.loc[df['batter']==batter_id, 'batter_name'].iloc[0]
    safe_name = batter_name.replace(' ', '_')

    # 5) Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for (b, s), row in comp.iterrows():
        x = b + s * 0.15
        y = row['diff']
        ax.scatter(x, y)
        ax.text(x, y, f"{b}-{s}", fontsize=9, ha='center', va='bottom')
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_xlabel('Balls–Strikes Count')
    ax.set_ylabel('Diff vs League (per 100 pitches)')
    ax.set_title(f'Batter {safe_name} Decision Quality by Count')
    plt.tight_layout()

    os.makedirs('outputs', exist_ok=True)
    out_file = f"outputs/{safe_name}.png"
    plt.savefig(out_file)
    plt.close(fig)

    return comp

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a batter's swing decision by name"
    )
    parser.add_argument(
        '--name', '-n',
        required=True,
        help="Full batter name, e.g. 'Mike Trout'"
    )
    args = parser.parse_args()

    # 1) Load processed features
    df = pd.read_csv("data/processed/statcast_features.csv")

    # 2) Reverse-lookup batter names and merge
    batter_ids = df['batter'].unique().tolist()
    print("Gathering player lookup table. This may take a moment...")
    lookup = playerid_reverse_lookup(batter_ids, key_type='mlbam')
    lookup['batter_name'] = lookup['name_first'] + ' ' + lookup['name_last']
    df = df.merge(
        lookup[['key_mlbam', 'batter_name']],
        left_on='batter', right_on='key_mlbam', how='left'
    )

    # 3) Case-insensitive name matching
    df['name_lower'] = df['batter_name'].str.lower()
    target = args.name.lower()
    exact = df.loc[df['name_lower'] == target, 'batter'].unique()
    if len(exact) >= 1:
        batter_id = exact[0]
    else:
        # try substring match
        candidates = df.loc[df['name_lower'].str.contains(target), 'batter_name'].unique()
        if len(candidates) == 0:
            print(f"No data found for '{args.name}'. Available names include:")
            for nm in sorted(df['batter_name'].unique())[:20]:
                print("  -", nm)
            sys.exit(1)
        else:
            print(f"No exact match for '{args.name}'. Did you mean:")
            for nm in candidates:
                print("  -", nm)
            sys.exit(1)

    # 4) Load full pipeline
    pipeline = joblib.load("models/xgb_pipeline.joblib")

    # 5) Compute league average
    league_avg = compute_league_avg(df, pipeline)
    print(f"League avg decision value (per 100 pitches): {league_avg}")

    # 6) Analyze this batter
    comp_df = analyze_player(df, pipeline, batter_id=batter_id)
    print(comp_df.round(2))

if __name__ == "__main__":
    main()
