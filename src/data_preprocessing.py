# data_preprocessing.py

import pandas as pd
import numpy as np
from pathlib import Path

_NUMERIC = [
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "plate_x",
    "plate_z",
]
_BOUNDS = {
    "release_speed": (60, 110),
    "release_spin_rate": (1500, 4000),
    "plate_x": (-2.5, 2.5),
    "plate_z": (0, 6.5),
}


def clean(raw_csv: Path | str, out_csv: Path | str = "data/processed/statcast_preprocessed.csv") -> Path:
    df = pd.read_csv(raw_csv, low_memory=False)

    # basic numeric bounds
    for col, (lo, hi) in _BOUNDS.items():
        df = df.loc[df[col].between(lo, hi)]

    # impute medians by pitch_type
    df[_NUMERIC] = (
        df.groupby("pitch_type")[_NUMERIC]
        .transform(lambda x: x.fillna(x.median()))
    )

    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✓ Preprocessed rows: {len(df):,}")
    return Path(out_csv)

if __name__ == "__main__":
    import sys
    clean(sys.argv[1] if len(sys.argv) > 1 else "data/raw/statcast_2022_2024.csv")
