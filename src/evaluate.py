import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

LABEL_MAP = {
    'release_speed':    'Pitch Release Speed (mph)',
    'plate_x':          'Plate Horizontal Location',
    'plate_z':          'Plate Vertical Location',
    'release_pos_x':    'Release Point X',
    'release_pos_z':    'Release Point Z',
    'release_extension':'Pitcher Release Extension (ft)',
    'pfx_x':            'Horizontal Movement (in)',
    'pfx_z':            'Vertical Movement (in)',
    'balls':            'Balls Count',
    'strikes':          'Strikes Count',
    'pre_run_exp':      'Pre-run Expectation',
    'is_strike':        'Is Strike (0/1)',
    'strike_streak':    'Consecutive Strikes',
    'outs_when_up':     'Outs When Up',
    'runners_on':       'Runners On Base',
    'score_diff':       'Score Differential',
    'inning':           'Inning Number',
    'stand_L':          'Batter is Left-Handed',
    'stand_R':          'Batter is Right-Handed',
    'p_throws_L':       'Pitcher is Left-Handed',
    'p_throws_R':       'Pitcher is Right-Handed',
    'pitch_type_FF':    'Fastball (Four-Seam)',
    'pitch_type_SL':    'Slider',
    'pitch_type_CH':    'Changeup',
    'pitch_type_CU':    'Curveball',
    'pitch_type_SI':    'Sinker',
    'pitch_type_nan':   'Pitch Type Missing',
    'zone_1.0':         'Zone 1',
    'zone_2.0':         'Zone 2',
    'zone_3.0':         'Zone 3',
    'zone_4.0':         'Zone 4',
    'zone_5.0':         'Zone 5',
    'zone_6.0':         'Zone 6',
    'zone_7.0':         'Zone 7',
    'zone_8.0':         'Zone 8',
    'zone_9.0':         'Zone 9',
    'zone_10.0':        'Zone 10',
    'zone_11.0':        'Zone 11',
    'zone_12.0':        'Zone 12',
    'zone_13.0':        'Zone 13',
    'zone_14.0':        'Zone 14',
    'zone_nan':         'Zone Missing',
    'prev_event_start':          'Prev Event: Start',
    'prev_event_ball':           'Prev Event: Ball',
    'prev_event_called_strike':  'Prev Event: Called Strike',
    'prev_event_swinging_strike':'Prev Event: Swinging Strike',
    'prev_event_foul':           'Prev Event: Foul',
}


def plot_feature_importance(pipeline, top_n=20, output_path="outputs/feature_importance.png"):
    """
    Plot the top_n most important features from the XGBoost model
    embedded inside a sklearn pipeline, replacing raw names with human-readable labels.
    """
    # 1. Extract model and preprocessor
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']

    # 2. Get raw feature names from preprocessor
    raw_names = preprocessor.get_feature_names_out()
    clean_names = [n.split("__", 1)[1] for n in raw_names]
    friendly_names = [LABEL_MAP.get(n, n) for n in clean_names]

    # 3. Get importance scores from the XGBoost model
    importances = model.feature_importances_

    # 4. Top-N
    top_idx = importances.argsort()[-top_n:]
    top_features = [friendly_names[i] for i in top_idx]
    top_scores   = importances[top_idx]

    # 5. Plot
    plt.figure(figsize=(8, 6))
    plt.barh(top_features, top_scores)
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()

    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    
def plot_shap_summary(pipeline, data_path="data/processed/statcast_features.csv", output_path="outputs/shap_summary.png"):
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']

    # Load and transform original data
    df = pd.read_csv(data_path)
    # Sample 1000 rows from original DataFrame (not yet transformed)
    df_sample = df.sample(n=10000, random_state=42)
    X_raw = df_sample[preprocessor.feature_names_in_]
    X_sample = preprocessor.transform(X_raw)

    # SHAP computation
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    # Feature name handling
    raw_names = preprocessor.get_feature_names_out()
    clean_names = [n.split("__", 1)[1] for n in raw_names]
    friendly_names = [LABEL_MAP.get(n, n) for n in clean_names]

    # SHAP summary plot
    shap.summary_plot(
        shap_values,
        features=X_sample,
        feature_names=friendly_names,
        show=False
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    # Load the complete pipeline
    pipeline = joblib.load("models/xgb_pipeline.joblib")
    # Plot feature importance from the embedded XGBoost model
    plot_feature_importance(pipeline)
    plot_shap_summary(pipeline)
