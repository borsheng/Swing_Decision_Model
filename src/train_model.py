import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Numeric and categorical feature definitions
numeric_features = [
    'release_speed', 'plate_x', 'plate_z',
    'release_pos_x', 'release_pos_z', 'release_extension',
    'pfx_x', 'pfx_z',
    'balls', 'strikes', 'outs_when_up',
    'runners_on', 'score_diff', 'inning',
    'pre_run_exp', 'is_strike', 'strike_streak'
]

categorical_features = [
    'pitch_type', 'stand', 'p_throws', 'zone',
    'inning_topbot', 'prev_event'
]


def train_model():
    """
    Load processed feature data, evaluate Ridge and XGBoost with GroupKFold CV,
    report MAE, RMSE, and R^2, then train final XGBoost model on full dataset
    and save the complete pipeline.
    """
    # Load feature-engineered data
    df = pd.read_csv("data/processed/statcast_features.csv")
    X = df[numeric_features + categorical_features]
    y = df['decision_value']
    groups = df['batter']

    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Model definitions
    models = {
        'Ridge': Ridge(alpha=1.0),
        'XGB': xgb.XGBRegressor(
            n_estimators=500,
            objective='reg:squarederror',
            random_state=42
        )
    }

    # GroupKFold cross-validation
    cv = GroupKFold(n_splits=5)
    scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']

    # Evaluate each model
    for name, mdl in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', mdl)
        ])
        cv_results = cross_validate(
            pipeline, X, y,
            groups=groups,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )
        mae = -np.mean(cv_results['test_neg_mean_absolute_error'])
        rmse = -np.mean(cv_results['test_neg_root_mean_squared_error'])
        r2 = np.mean(cv_results['test_r2'])
        print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

    # Final training on full dataset with XGBoost
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', models['XGB'])
    ])
    final_pipeline.fit(X, y)

    # Save the trained pipeline
    joblib.dump(final_pipeline, "models/xgb_pipeline.joblib")
    print("Final XGBoost pipeline saved to models/xgb_pipeline.joblib")


if __name__ == '__main__':
    train_model()
