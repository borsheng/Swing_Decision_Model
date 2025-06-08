import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from xgboost import XGBRegressor

_NUM = [
    "release_speed", "release_spin_rate", "release_extension",
    "plate_x", "plate_z", "balls", "strikes", "outs_when_up",
]
_CAT = ["pitch_type", "stand", "p_throws"]


def train(csv: Path | str = "data/processed/statcast_features.csv", model_dir: Path | str = "models") -> Path:
    df = pd.read_csv(csv)
    X = df[_NUM + _CAT + ["is_swing"]]
    y = df["decision_delta"]
    groups = df["batter"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), _NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), _CAT),
    ])
    model = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.85, colsample_bytree=0.8,
        objective="reg:squarederror", n_jobs=-1, random_state=42,
    )
    pipe = Pipeline([("prep", pre), ("xgb", model)])

    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        groups=groups,  # pass batter IDs so each split keeps hitters independent
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    print("CV MAE:", -scores.mean(), "+/-", scores.std())

    pipe.fit(X, y)

    # ── attach real feature names to XGBoost booster ──
    feat_names = pipe.named_steps["prep"].get_feature_names_out()
    pipe.named_steps["xgb"].get_booster().feature_names = feat_names.tolist()

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    mpath = Path(model_dir) / "xgb_decision.joblib"
    joblib.dump(pipe, mpath)
    print(f"✓ Model saved → {mpath}")
    return mpath

if __name__ == "__main__":
    train()
