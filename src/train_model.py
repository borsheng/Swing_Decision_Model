import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# numeric_features（train_model.py / leaderboard.py / player_analysis.py）
numeric_features = [
    'release_speed', 'plate_x', 'plate_z',
    'release_pos_x', 'release_pos_z', 'release_extension',
    'pfx_x', 'pfx_z',
    'balls', 'strikes', 'outs_when_up',
    'runners_on', 'score_diff', 'inning',
    'pre_run_exp', 'is_strike', 'strike_streak'
]

# categorical_features
categorical_features = [
    'pitch_type', 'stand', 'p_throws', 'zone',
    'inning_topbot', 'prev_event'
]



def train_model():
    """
    Load processed features, split into train/test, build a preprocessing pipeline,
    train an XGBoost regressor with early stopping, and save both model and preprocessor.
    """
    # Load feature-engineered data
    df = pd.read_csv("data/processed/statcast_features.csv")
    X = df[numeric_features + categorical_features]
    y = df['decision_value']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define preprocessing steps:
    # - Median imputation for numeric features
    # - One-hot encoding for categorical features
    num_imputer = SimpleImputer(strategy='median')
    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preprocessor = ColumnTransformer([
        ('num', num_imputer, numeric_features),
        ('cat', cat_encoder, categorical_features)
    ], remainder='drop')

    # Fit the pipeline and transform training and test data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Initialize XGBoost regressor with early stopping parameters
    model = xgb.XGBRegressor(
        n_estimators=500,
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=10
    )

    # Train the model, monitoring performance on the test set
    model.fit(
        X_train_transformed, y_train,
        eval_set=[(X_test_transformed, y_test)],
        verbose=True
    )

    # Save the trained model and preprocessing pipeline
    model.save_model("models/xgb_model.json")
    joblib.dump(preprocessor, "models/preprocessor.joblib")


if __name__ == "__main__":
    train_model()