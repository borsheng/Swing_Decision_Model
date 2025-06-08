import argparse
from pathlib import Path
import pandas as pd
import joblib
from pybaseball import playerid_reverse_lookup

DEFAULT_MIN_PITCHES = 1000
DEFAULT_TOP_N = 10


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _attach_names(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has human‑readable `batter_name`."""
    if "batter_name" in df.columns and df["batter_name"].notna().any():
        return df
    ids = df["batter"].unique().tolist()
    print(f"→ Downloading player lookup for {len(ids)} IDs …")
    look = playerid_reverse_lookup(ids, key_type="mlbam")
    look["batter_name"] = look["name_first"] + " " + look["name_last"]
    return df.merge(look[["key_mlbam", "batter_name"]],
                    left_on="batter", right_on="key_mlbam", how="left")


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────────────────────────────────────

def build_leaderboards(feature_csv: Path | str,
                       model_path: Path | str,
                       out_dir: Path | str,
                       min_pitches: int,
                       top_n: int) -> None:
    # 1. Load & attach names
    df = pd.read_csv(feature_csv)
    print(f"✓ Loaded {len(df):,} rows")
    df = _attach_names(df)

    # 2. Predict Decision Δ per 100 pitches
    pipe = joblib.load(model_path)
    cols_in = pipe.named_steps["prep"].feature_names_in_
    df["decision100"] = pipe.predict(df[cols_in]) * 100
    print("✓ Model predictions complete")

    # 3. Zero‑center：聯盟平均 → 0
    league_mean = df["decision100"].mean()
    df["score"] = (df["decision100"] - league_mean).round(2)
    print(f"✓ League mean = {league_mean:.2f}  (scores now centered)")

    # 4. Pitch‑count filter
    counts = df["batter"].value_counts()
    valid_ids = counts[counts >= min_pitches].index
    print(f"✓ {len(valid_ids)} batters ≥ {min_pitches} pitches")
    if not len(valid_ids):
        print("⚠ No batters meet min_pitches threshold, aborting …")
        return

    agg = (
        df[df["batter"].isin(valid_ids)]
        .groupby(["batter", "batter_name"])["score"].mean().reset_index()
        .rename(columns={"score": "decision_adv_per100"})
        .sort_values("decision_adv_per100", ascending=False)
    )

    top = agg.head(top_n)
    bot = agg.tail(top_n)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    top_path = out / "top10.csv"; bot_path = out / "bottom10.csv"
    top.to_csv(top_path, index=False)
    bot.to_csv(bot_path, index=False)
    print(f"✓ Leaderboards saved → {top_path.resolve()} & {bot_path.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli():
    ap = argparse.ArgumentParser(description="Generate zero‑centered batter leaderboards")
    ap.add_argument("--features", default="data/processed/statcast_features.csv")
    ap.add_argument("--model", default="models/xgb_decision.joblib")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--min_pitches", type=int, default=DEFAULT_MIN_PITCHES)
    ap.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    args = ap.parse_args()

    build_leaderboards(args.features, args.model, args.out_dir, args.min_pitches, args.top_n)


if __name__ == "__main__":
    _cli()