import pandas as pd
import numpy as np
from pathlib import Path

_SWING_DESCRIPTIONS = {
    "swinging_strike",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_score",
    "swinging_strike_blocked",
}

_BASIC_FEATURES = [
    "pitch_type", "batter", "pitcher", "stand", "p_throws",
    "release_speed", "release_spin_rate", "release_extension",
    "plate_x", "plate_z", "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b", "inning", "inning_topbot",
]

_GRID_SIZE_CM = 5  # 5 cm × 5 cm spatial bucket
_MIN_OBS_PER_BUCKET = 25  # fallback threshold


# -------- Run‑Expectancy helper ------------------------------------------------

def _state(on1, on2, on3, outs):
    return f"{int(bool(on1))}{int(bool(on2))}{int(bool(on3))}_{outs}"


def _compute_delta_re(df: pd.DataFrame, re_table: pd.DataFrame) -> pd.Series:
    df["pre_state"] = (
        df[["on_1b", "on_2b", "on_3b", "outs_when_up"]]
        .apply(lambda r: _state(*r), axis=1)
    )
    # assume post‑state unchanged (fine for decision metric); can refine w/ events.
    df["post_state"] = df["pre_state"]
    lookup = re_table.set_index("state")["RE"]
    pre = df["pre_state"].map(lookup)
    post = df["post_state"].map(lookup)
    delta = post - pre  # runs_scored omitted, we only need relative value
    return delta


# -------- EV‑based Decision metric -------------------------------------------

def _add_ev_decision(df: pd.DataFrame) -> pd.DataFrame:
    """Add EV_swing / EV_take columns + decision_delta.

    NOTE: rows missing plate_x or plate_z will be *dropped* (<< 1% of data)
    to avoid NaN→int casting errors.
    """
    # ── 0. remove rows lacking spatial coords ─────────────────────────────
    df = df.dropna(subset=["plate_x", "plate_z"]).copy()

    # ── 1. grid bucket indices (5 cm) ─────────────────────────────────────
    df["g_x"] = (np.floor(df["plate_x"] * 100 / _GRID_SIZE_CM)).astype(int)
    df["g_z"] = (np.floor(df["plate_z"] * 100 / _GRID_SIZE_CM)).astype(int)

    key = ["g_x", "g_z", "pitch_type", "balls", "strikes"]

    # ── 2. bucket‑level EV tables ─────────────────────────────────────────
    swing_ev = (
        df[df.is_swing == 1]
        .groupby(key, dropna=False)["delta_re"].mean()
        .rename("ev_swing")
    )
    take_ev = (
        df[df.is_swing == 0]
        .groupby(key, dropna=False)["delta_re"].mean()
        .rename("ev_take")
    )

    df = df.join(swing_ev, on=key).join(take_ev, on=key)

    # ── 3. league‑wide fallback for sparse buckets ────────────────────────
    df["ev_swing"].fillna(df["ev_swing"].mean(), inplace=True)
    df["ev_take"].fillna(df["ev_take"].mean(), inplace=True)

    # ── 4. Decision Δ ─────────────────────────────────────────────────────
    chosen_ev = np.where(df.is_swing == 1, df.ev_swing, df.ev_take)
    best_ev = np.maximum(df.ev_swing, df.ev_take)
    df["decision_delta"] = chosen_ev - best_ev
    return df


# -------- Pipeline -----------------------------------------------------------

def build_feature_table(
    pre_csv: Path | str = "data/processed/statcast_preprocessed.csv",
    re_csv: Path | str = "data/raw/run_expectancy.csv",
    out_csv: Path | str = "data/processed/statcast_features.csv",
) -> Path:
    df = pd.read_csv(pre_csv, low_memory=False)

    # label whether swing
    df["is_swing"] = df["description"].isin(_SWING_DESCRIPTIONS).astype(int)

    # ΔRE per pitch
    if "delta_run_exp" in df.columns and not df["delta_run_exp"].isna().all():
        df["delta_re"] = df["delta_run_exp"]
    else:
        re_table = pd.read_csv(re_csv)
        df["delta_re"] = _compute_delta_re(df, re_table)

    # EV‑based decision metric
    df = _add_ev_decision(df)

    features = _BASIC_FEATURES + ["is_swing", "decision_delta"]
    df = df[features]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✓ Feature table saved ({len(df):,} rows) → {out_csv}")
    return Path(out_csv)

if __name__ == "__main__":
    build_feature_table()
