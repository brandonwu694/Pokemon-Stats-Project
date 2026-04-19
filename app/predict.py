import json

import numpy as np
import pandas as pd
import joblib

from config import MODELS_DIR
from app.schemas import PredictionInput


model = joblib.load(MODELS_DIR / "xg_boost_tuned.joblib")
MODEL_COLUMNS_PATH = MODELS_DIR / "xg_boost_tuned_columns.json"


def load_model_columns() -> list[str]:
    """Load the feature schema used during training."""
    if MODEL_COLUMNS_PATH.exists():
        return json.loads(MODEL_COLUMNS_PATH.read_text())

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    raise FileNotFoundError(
        "Missing model feature schema. Re-run model training to generate "
        f"{MODEL_COLUMNS_PATH.name}."
    )


MODEL_COLUMNS = load_model_columns()


def preprocess_input(data: PredictionInput) -> pd.DataFrame:
    """Convert API input into model-ready feature vector"""
    df = pd.DataFrame([data.model_dump()])

    # Apply the same numeric transforms used during feature engineering.
    df["log_height_m"] = np.log1p(df["height_m"])
    df["log_weight_kg"] = np.log1p(df["weight_kg"])
    df = df.drop(columns=["height_m", "weight_kg"])

    # Make type_2 feature 'none' if necessary
    if df["type_2"].isna().any():
        df["type_2"] = df["type_2"].fillna("none")

    # Match training-time category spelling
    df["status"] = df["status"].str.replace("_", " ", regex=False)
    df["growth_rate"] = df["growth_rate"].str.replace("_", " ", regex=False)

    # One-hot encode
    df_encoded = pd.get_dummies(df)

    # Align columns
    df_encoded = df_encoded.reindex(columns=MODEL_COLUMNS, fill_value=0)

    return df_encoded


def make_prediction(data: PredictionInput) -> float:
    """Generate a prediction from validated API input."""
    model_input = preprocess_input(data)
    prediction = model.predict(model_input)[0]
    return float(prediction)
