import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from data_cleaning import execute_data_cleaning_pipeline, validate_dataset
from feature_engineering import (
    execute_feature_engineering_pipeline,
    get_base_name,
    validate_feature_dataset,
)
from model_training import execute_model_training_pipeline, save_training_artifacts


def verify_input_files() -> tuple[Path, Path]:
    """Verify required raw input files exist."""
    pokedex_path = RAW_DATA_DIR / "pokedex.csv"
    evolution_path = RAW_DATA_DIR / "pokemon_evolutions.csv"

    if not pokedex_path.exists():
        raise FileNotFoundError(f"Missing required input file: {pokedex_path}")

    if not evolution_path.exists():
        raise FileNotFoundError(f"Missing required input file: {evolution_path}")

    return pokedex_path, evolution_path


def prepare_evolution_data(evolution_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare evolution dataset for feature merge."""
    evolution_df = evolution_df.rename(columns={"Name": "name"})
    evolution_df = get_base_name(evolution_df)
    evolution_df = evolution_df.drop_duplicates(subset=["base_name", "EvoStage"])
    return evolution_df[["base_name", "EvoStage"]]


def main() -> None:
    """Run the full Pokemon modeling pipeline."""
    pokedex_path, evolution_path = verify_input_files()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw datasets...")
    pokemon_raw = pd.read_csv(pokedex_path)
    evolution_raw = pd.read_csv(evolution_path)

    print("Cleaning Pokemon dataset...")
    pokemon_clean = execute_data_cleaning_pipeline(pokemon_raw)
    validate_dataset(pokemon_clean)
    pokemon_clean.to_parquet(PROCESSED_DATA_DIR / "pokemon_data_clean.parquet", index=False)

    print("Preparing evolution data and engineering features...")
    evolution_prepared = prepare_evolution_data(evolution_raw)
    pokemon_features = execute_feature_engineering_pipeline(pokemon_clean, evolution_prepared)
    validate_feature_dataset(pokemon_features)
    pokemon_features.to_parquet(
        PROCESSED_DATA_DIR / "pokemon_data_features_evo_stage.parquet",
        index=False,
    )

    print("Training model...")
    model, metrics, _ = execute_model_training_pipeline(pokemon_features)
    feature_columns = pokemon_features.drop(columns=["total_points"]).columns.tolist()
    save_training_artifacts(model, metrics, feature_columns)

    print("\nFull pipeline completed successfully.")


if __name__ == "__main__":
    main()
