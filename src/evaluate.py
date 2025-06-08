import matplotlib.pyplot as plt
import joblib
import shap
import pandas as pd
from pathlib import Path


def evaluate(model_path: Path | str = "models/xgb_decision.joblib",
             feature_csv: Path | str = "data/processed/statcast_features.csv",
             out_dir: Path | str = "outputs") -> None:
    pipe = joblib.load(model_path)
    model = pipe.named_steps["xgb"]
    booster = model.get_booster()

    # ── ensure readable feature names ──
    if booster.feature_names and booster.feature_names[0].startswith("f"):
        feat_names = pipe.named_steps["prep"].get_feature_names_out().tolist()
        booster.feature_names = feat_names

    imp = booster.get_score(importance_type="gain")
    top = sorted(imp.items(), key=lambda t: t[1], reverse=True)[:20]
    labels, gains = zip(*top)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── bar plot ──
    plt.figure(figsize=(8,5))
    plt.barh(labels, gains)
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances (Gain)")
    plt.tight_layout()
    fi_path = out / "feature_importance.png"
    plt.savefig(fi_path, dpi=300)
    print(f"✓ Feature importance saved → {fi_path}")

        # ── SHAP summary (sample 10k) ──
    df_sample = pd.read_csv(feature_csv, nrows=10_000)
    X_raw = df_sample.drop(columns=["decision_delta"])

    # Apply same preprocessing and keep column names
    feature_names = pipe.named_steps["prep"].get_feature_names_out()
    X_trans = pipe.named_steps["prep"].transform(X_raw)
    X_df = pd.DataFrame(X_trans, columns=feature_names)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_df)

    shap.summary_plot(shap_values, X_df, show=False)
    shap_path = out / "shap_summary.png"
    plt.savefig(shap_path, dpi=300)
    print(f"✓ SHAP summary saved → {shap_path}")

if __name__ == "__main__":
    evaluate()