import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────── settings ────────────────────────────────
_GRID_SIZE_FT = 0.3          # 0.3 
_MIN_OBS_PER_BUCKET = 30      # fallback threshold for EV stats

_SWING_DESCRIPTIONS = {
    "swinging_strike", "foul", "foul_tip",
    "hit_into_play", "hit_into_play_score", "swinging_strike_blocked",
}

_BASIC_FEATURES = [
    "pitch_type", "batter", "pitcher", "stand", "p_throws",
    "release_speed", "release_spin_rate", "release_extension",
    "plate_x", "plate_z", "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b", "inning", "inning_topbot",
]

# ─────────────────────── run‑expectancy helper ───────────────────────────
def _state(on1, on2, on3, outs):
    return f"{int(bool(on1))}{int(bool(on2))}{int(bool(on3))}_{outs}"

def _compute_delta_re(df: pd.DataFrame, re_table: pd.DataFrame) -> pd.Series:
    df["pre_state"] = (
        df[["on_1b", "on_2b", "on_3b", "outs_when_up"]]
        .apply(lambda r: _state(*r), axis=1)
    )
    df["post_state"] = df["pre_state"]  
    lookup = re_table.set_index("state")["RE"]
    pre = df["pre_state"].map(lookup)
    post = df["post_state"].map(lookup)
    return post - pre                    

# ─────────────────────── EV‑based decision metric ────────────────────────
def _add_ev_decision(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EV_swing / EV_take with 0.3 ft grid & sample threshold."""
    df = df.dropna(subset=["plate_x", "plate_z"]).copy()

    # 1) 0.3 ft grid indices
    df["g_x"] = np.floor(df["plate_x"] / _GRID_SIZE_FT).astype(int)
    df["g_z"] = np.floor(df["plate_z"] / _GRID_SIZE_FT).astype(int)
    key = ["g_x", "g_z", "pitch_type", "balls", "strikes"]

    # 2) bucket stats with count filter
    swing_stats = (
        df[df.is_swing == 1]
        .groupby(key)["delta_re"]
        .agg(["mean", "count"])
        .query("count >= @_MIN_OBS_PER_BUCKET")
        .rename(columns={"mean": "ev_swing"})
    )
    take_stats = (
        df[df.is_swing == 0]
        .groupby(key)["delta_re"]
        .agg(["mean", "count"])
        .query("count >= @_MIN_OBS_PER_BUCKET")
        .rename(columns={"mean": "ev_take"})
    )

    # 3) join平均值，稀疏格 fallback→全聯盟均值
    df = (
        df.join(swing_stats["ev_swing"], on=key)
          .join(take_stats["ev_take"],  on=key)
    )
    df["ev_swing"].fillna(df["ev_swing"].mean(), inplace=True)
    df["ev_take"].fillna(df["ev_take"].mean(),   inplace=True)

    # 4) decision delta
    chosen = np.where(df.is_swing == 1, df.ev_swing, df.ev_take)
    best   = np.maximum(df.ev_swing, df.ev_take)
    df["decision_delta"] = chosen - best
    return df

# ───────────────────────────── pipeline ──────────────────────────────────
def build_feature_table(
    pre_csv: Path | str = "data/processed/statcast_preprocessed.csv",
    re_csv:  Path | str = "data/raw/run_expectancy.csv",
    out_csv: Path | str = "data/processed/statcast_features.csv",
) -> Path:
    df = pd.read_csv(pre_csv, low_memory=False)
    df["is_swing"] = df["description"].isin(_SWING_DESCRIPTIONS).astype(int)

    if "delta_run_exp" in df and not df["delta_run_exp"].isna().all():
        df["delta_re"] = df["delta_run_exp"]
    else:
        re_table = pd.read_csv(re_csv)
        df["delta_re"] = _compute_delta_re(df, re_table)

    df = _add_ev_decision(df)
    df = df[_BASIC_FEATURES + ["is_swing", "decision_delta"]]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✓ Feature table saved ({len(df):,} rows) → {out_csv}")
    return Path(out_csv)

if __name__ == "__main__":
    build_feature_table()
