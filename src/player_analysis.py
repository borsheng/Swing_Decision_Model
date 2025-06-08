import argparse
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pybaseball import playerid_reverse_lookup

_COUNTS_ORDER = ["0-0","1-0","0-1","1-1","2-1","1-2","3-1","2-2","3-2"]


def _attach_names(df: pd.DataFrame) -> pd.DataFrame:
    if "batter_name" in df.columns and df["batter_name"].notna().any():
        return df
    ids = df["batter"].unique().tolist()
    look = playerid_reverse_lookup(ids, key_type="mlbam")
    look["batter_name"] = look["name_first"] + " " + look["name_last"]
    return df.merge(look[["key_mlbam", "batter_name"]],
                    left_on="batter", right_on="key_mlbam", how="left")


def analyse_batter(name: str,
                   feature_csv: str = "data/processed/statcast_features.csv",
                   model_path: str = "models/xgb_decision.joblib",
                   out_dir: str = "outputs") -> pd.DataFrame:
    df = pd.read_csv(feature_csv)
    df = _attach_names(df)
    df["name_lower"] = df["batter_name"].str.lower()

    matches = df.loc[df["name_lower"] == name.lower(), "batter"].unique()
    if len(matches) == 0:
        raise ValueError(f"No exact match for '{name}' in dataset.")
    batter_id = matches[0]

    pipe = joblib.load(model_path)
    cols_in = pipe.named_steps["prep"].feature_names_in_

    df["pred"] = pipe.predict(df[cols_in]) * 100

    league = df.groupby(["balls", "strikes"])["pred"].mean()
    player = df[df["batter"] == batter_id].groupby(["balls", "strikes"])["pred"].mean()

    comp = pd.DataFrame({"player": player, "league": league}).dropna()
    comp["diff"] = (comp["player"] - comp["league"]).round(2)

    # plot
    safe_name = name.replace(" ", "_")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,4))
    for (b, s), row in comp.iterrows():
        count_str = f"{b}-{s}"
        x = _COUNTS_ORDER.index(count_str) if count_str in _COUNTS_ORDER else len(_COUNTS_ORDER)
        ax.scatter(x, row["diff"])
        ax.text(x, row["diff"], count_str, ha="center", va="bottom", fontsize=8)
    ax.axhline(0, ls="--", color="gray")
    ax.set_xticks(range(len(_COUNTS_ORDER)))
    ax.set_xticklabels(_COUNTS_ORDER, rotation=40)
    ax.set_ylabel("Decision Δ vs League (per 100 pitches)")
    ax.set_title(f"{name} – Decision Quality by Count")
    plt.tight_layout()
    img_path = Path(out_dir) / f"{safe_name}.png"
    plt.savefig(img_path, dpi=300)
    plt.close(fig)
    print(f"✓ Plot saved → {img_path}")
    return comp


def _cli():
    ap = argparse.ArgumentParser(description="Analyse batter decision quality vs league")
    ap.add_argument("--name", "-n", required=True, help="Full batter name, e.g. 'Mike Trout'")
    args = ap.parse_args()
    df = analyse_batter(args.name)
    print(df)


if __name__ == "__main__":
    _cli()