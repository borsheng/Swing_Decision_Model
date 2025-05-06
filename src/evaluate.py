"""
evaluate.py

Generate a horizontal bar chart of the top N feature importances
from the trained XGBoost model, using human-readable labels.
"""

import os
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# Mapping from cleaned feature names (after removing "num__"/"cat__") to friendly labels
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


def plot_feature_importance(model, preprocessor, top_n=20, output_path="outputs/feature_importance.png"):
    """
    Plot the top_n most important features from the XGBoost model,
    replacing machine names with human-readable labels.
    """
    # 1. Retrieve the pipeline feature names (e.g. "num__release_speed", "cat__pitch_type_FF")
    raw_names = preprocessor.get_feature_names_out()
    # 2. Strip off the "num__" or "cat__" prefix
    clean_names = [n.split("__", 1)[1] for n in raw_names]
    # 3. Map to friendly labels (fallback to the raw name if not in LABEL_MAP)
    friendly_names = [LABEL_MAP.get(n, n) for n in clean_names]

    # 4. Get importance scores from the trained model
    importances = model.feature_importances_

    # 5. Identify indices of the top_n features
    top_idx = importances.argsort()[-top_n:]

    # 6. Prepare names & values for plotting
    top_features = [friendly_names[i] for i in top_idx]
    top_scores   = importances[top_idx]

    # 7. Plot
    plt.figure(figsize=(8, 6))
    plt.barh(top_features, top_scores)
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()

    # 8. Ensure output folder exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    # Load the trained model
    model = xgb.XGBRegressor()
    model.load_model("models/xgb_model.json")
    # Load the preprocessing pipeline
    preprocessor = joblib.load("models/preprocessor.joblib")

    # Generate and save the labeled feature importance plot
    plot_feature_importance(model, preprocessor)


