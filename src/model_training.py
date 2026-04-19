import sys 
import json
from pathlib import Path 
from typing import Any, Optional
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor

from config import MODELS_DIR, PROCESSED_DATA_DIR


def validate_training_data(df: pd.DataFrame) -> None:
    """Validate that the feature dataset is ready for model training."""
    if "total_points" not in df.columns:
        raise KeyError("Missing target column: total_points")

    if df.empty:
        raise ValueError("Training dataset is empty")

    total_missing = int(df.isna().sum().sum())
    if total_missing != 0:
        raise ValueError(f"Training dataset contains {total_missing} missing values")

    invalid_cols = [
        col for col in df.drop(columns=["total_points"]).columns
        if not (
            pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        )
    ]
    if invalid_cols:
        raise TypeError(f"Non-numeric feature columns found: {invalid_cols}")


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing folds"""
    y = df["total_points"]
    X = df.drop(columns=["total_points"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_xg_boost_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    estimator: Optional[BaseEstimator] = None
) -> BaseEstimator:
    """Train fine-tuned XGBoost model or a provided estimator."""
    
    if estimator is None:
        estimator = XGBRegressor(
            objective="reg:squarederror",
            colsample_bytree=0.8,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=5,
            n_estimators=300,
            subsample=0.8,
            random_state=42,
            eval_metric="rmse",
        )
    
    estimator.fit(X_train, y_train)
    return estimator


def evaluate_model(estimator: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate regression model using RMSE, MAE, and R2"""    
    y_pred = estimator.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation Metrics")
    print("-" * 35)
    print(f"RMSE : {rmse:>10.2f}")
    print(f"MAE  : {mae:>10.2f}")
    print(f"R²   : {r2:>10.3f}")
    print("-" * 35)

    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_training_artifacts(
    estimator: BaseEstimator,
    metrics: dict[str, float],
    model_path: Path = MODELS_DIR / "xg_boost_tuned.joblib",
    metrics_path: Path = MODELS_DIR / "xg_boost_tuned_metrics.json",
) -> None:
    """Persist the trained model and evaluation metrics."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))


def execute_model_training_pipeline(df: pd.DataFrame) -> tuple[BaseEstimator, dict[str, float], dict[str, Any]]:
    """Train XGBoost model with feature engineered dataset"""
    validate_training_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_xg_boost_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    split_data_bundle = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    return model, metrics, split_data_bundle
    

if __name__ == "__main__":
    pokemon_df = pd.read_parquet(PROCESSED_DATA_DIR / "pokemon_data_features_evo_stage.parquet")
    model, metrics, _ = execute_model_training_pipeline(pokemon_df)
    save_training_artifacts(model, metrics)
