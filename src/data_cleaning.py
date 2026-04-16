import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import RAW_DATA_DIR


def impute_value(df: pd.DataFrame, col: str, round_to_int: bool = True) -> pd.DataFrame:
    """Impute missing numerical values with the median, preserving floats when needed."""
    df = df.copy()
    fill_value = df[col].median()
    df[col] = df[col].fillna(fill_value)
    if round_to_int:
        df[col] = df[col].round().astype(int)
    return df


def impute_values(df: pd.DataFrame) -> pd.DataFrame:
    """Perform imputations on required columns"""
    required = ["catch_rate", "base_friendship", "base_experience", "weight_kg"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    df = df.copy()
    df = impute_value(df, "catch_rate")
    df = impute_value(df, "base_friendship")
    df = impute_value(df, "base_experience")
    df = impute_value(df, "weight_kg", round_to_int=False)

    return df


def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns unnecessary for modeling or EDA purposes"""
    df = df.copy()
    df = df.drop(
    columns= [
            "Unnamed: 0",
            "pokedex_number",
            "japanese_name",
            "species",
            "moves",
            "smogon_description",
            "bulba_description",
            "ability_1_description",
            "ability_2_description",
            "ability_hidden_description",
            "hp",
            "attack",
            "defense",
            "sp_attack",
            "sp_defense",
            "speed",
            "egg_type_1",
            "egg_type_2",
            "egg_type_number",
            "percentage_male",
            "egg_cycles"
        ]
    )
    return df


def clean_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize spacing, strip trailing whitespaces, and convert str dtypes to lowercase"""
    df = df.copy()
    str_cols = df.select_dtypes(include=["object", "string"]).columns

    df[str_cols] = (
        df[str_cols]
        .apply(lambda col: col.str.strip().str.lower().str.replace(r"\s+", " ", regex=True))
    )

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove missing values from columns"""
    required = ["ability_2", "ability_hidden", "weight_kg", "growth_rate", "type_2"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    df = df.copy()
    
    df["ability_2"] = df["ability_2"].fillna("None")
    df["ability_hidden"] = df["ability_hidden"].fillna("None")

    df["growth_rate"] = df["growth_rate"].fillna(
        df["growth_rate"].mode()[0]
    )

    df["type_2"] = df["type_2"].fillna("None")
    return df


def execute_data_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Execute data cleaning pipeline"""
    df = df.copy()
    # Drop unnecessary columns
    df = drop_cols(df)

    # Drop missing columns
    df = handle_missing_values(df)

    # Impute/handle missing values
    df = impute_values(df)

    # Clean str/object dtype cols
    df = clean_str_columns(df)

    # Add missing ability for Regidrago
    assert df.at[1039, "name"] == "regidrago"
    df.loc[1039, "ability_1"] = "dragon's maw"

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Print a compact validation summary for the cleaned dataset."""
    missing_by_col = df.isna().sum()
    total_missing = int(missing_by_col.sum())

    assert total_missing == 0

    print("\nDataset validation")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Total missing values: {total_missing}")
    print("\nMissing values by column:")
    print(missing_by_col.to_string())
    print("\nSample rows:")
    print(df.head(5).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    pokemon_df = pd.read_csv(RAW_DATA_DIR / "pokedex.csv")
    pokemon_df_clean = execute_data_cleaning_pipeline(pokemon_df)
    validate_dataset(pokemon_df_clean)
