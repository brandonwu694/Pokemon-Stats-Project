import sys 
import re
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log transformation to weight and height features to reduce skewness"""
    df = df.copy()

    required = ["weight_kg", "height_m"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    df["log_weight_kg"] = np.log1p(df["weight_kg"])
    df["log_height_m"] = np.log1p(df["height_m"])

    return df


def create_special_variant_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature that denotes if Pokemon is a special variant (e.g., Mega, Primal)"""
    if "name" not in df.columns:
        raise KeyError("Missing name column")
    
    df = df.copy()
    df["is_mega"] = df["name"].str.contains("mega", case=False, na=False).astype(int)
    df["is_primal"] = df["name"].str.contains("primal", case=False, na=False).astype(int)
    df["is_form"] = df["name"].str.contains("form", case=False, na=False).astype(int)

    return df


def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove features from dataframe that are not relevant to modeling"""
    required = [
        "name",
        "total_points",
        "generation",
        "log_height_m",
        "log_weight_kg",
        "is_mega",
        "is_primal",
        "is_form",
        "status",
        "type_1",
        "type_2",
        "growth_rate"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    df = df.copy()
    df = df[required]

    return df


def one_hot_encode_cols(df: pd.DataFrame) -> pd.DataFrame:
    """One hot encode categorical features"""
    required = ["status", "type_1", "type_2", "growth_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    df = df.copy()
    df = pd.get_dummies(
        df,
        columns=required,
        drop_first=True,
    )

    return df


def normalize_name(name: str) -> str:
    """Normalize Pokemon names to ensure proper matching"""
    name = name.strip().lower()
    # Replace spaces with hyphens
    name = name.replace(" ", "-")
    # Handle gender symbols
    name = name.replace("♀", "-f").replace("♂", "-m")
    # Remove special characters (keep letters, numbers, hyphens)
    name = re.sub(r"[^a-z0-9\-]", "", name)
    return name


def classify_special_pokemon(name: str) -> str:
    """Convert special cases of Pokemon into their base species name."""

    # Handle remaining manual special cases first
    name_map = {
        "dusk-mane-necrozma": "necrozma",
        "dawn-wings-necrozma": "necrozma",
        "ultra-necrozma": "necrozma",
        "eternatus-eternamax": "eternatus",
        "ash-greninja": "greninja",
        "own-tempo-rockruff": "rockruff",
        "partner-pikachu": "pikachu",
        "partner-eevee": "eevee",
        "hoopa-hoopa-unbound": "hoopa",
        "hoopa-hoopa-confined": "hoopa",
        "hoopa-unbound": "hoopa",
        "hoopa-confined": "hoopa",
        "keldeo-ordinary-form": "keldeo",
        "keldeo-resolute-form": "keldeo",
        "no-drive-genesect": "genesect",
        "douse-drive-genesect": "genesect",
        "shock-drive-genesect": "genesect",
        "burn-drive-genesect": "genesect",
        "chill-drive-genesect": "genesect",
        "galarian-darmanitan": "galarian-darmanitan",
        "galarian-darmanitan-zen-mode": "galarian-darmanitan",
        "castform-normal": "castform",
        "castform-sunny-form": "castform",
        "castform-rainy-form": "castform",
        "castform-snowy-form": "castform",
        "burmy-plant-cloak": "burmy",
        "burmy-sandy-cloak": "burmy",
        "burmy-trash-cloak": "burmy",
        "cherrim-overcast-form": "cherrim",
        "cherrim-sunshine-form": "cherrim",
        "shellos-west-sea": "shellos",
        "shellos-east-sea": "shellos",
        "gastrodon-west-sea": "gastrodon",
        "gastrodon-east-sea": "gastrodon",
        "deerling-spring-form": "deerling",
        "deerling-summer-form": "deerling",
        "deerling-autumn-form": "deerling",
        "deerling-winter-form": "deerling",
        "sawsbuck-spring-form": "sawsbuck",
        "sawsbuck-summer-form": "sawsbuck",
        "sawsbuck-autumn-form": "sawsbuck",
        "sawsbuck-winter-form": "sawsbuck",
        "vivillon-meadow-pattern": "vivillon",
        "floette-red-flower": "floette",
        "floette-yellow-flower": "floette",
        "floette-orange-flower": "floette",
        "floette-blue-flower": "floette",
        "floette-white-flower": "floette",
        "florges-red-flower": "florges",
        "florges-yellow-flower": "florges",
        "florges-orange-flower": "florges",
        "florges-blue-flower": "florges",
        "florges-white-flower": "florges",
        "furfrou-natural-form": "furfrou",
        "xerneas-neutral-mode": "xerneas",
        "xerneas-active-mode": "xerneas",
        "silvally-type-normal": "silvally",
        "minior-meteor-form": "minior",
        "minior-red-core": "minior",
        "minior-core-form": "minior",
        "mimikyu-disguised-form": "mimikyu",
        "mimikyu-busted-form": "mimikyu",
        "alcremie-vanilla-cream": "alcremie",
        "unown-one-form": "unown",
        "flabb": "flabebe",
        "flabb-red-flower": "flabebe",
        "flabb-yellow-flower": "flabebe",
        "flabb-orange-flower": "flabebe",
        "flabb-blue-flower": "flabebe",
        "flabb-white-flower": "flabebe",
        "galarian-darmanitan-standard-mode": "galarian-darmanitan",

    }

    if name in name_map:
        return name_map[name]

    # Mega Pokemon
    if name.startswith("mega-"):
        name = name.replace("mega-", "", 1)

        # Special cases of Mega Pokemon (e.g., mega-charizard-x)
        if name.endswith("-x") or name.endswith("-y"):
            name = name.rsplit("-", 1)[0]

    # Primal form Pokemon
    if name.startswith("primal-"):
        name = name.replace("primal-", "", 1)

    # Handle size variants (e.g., pumpkaboo-small-size)
    size_variants = {"-small-size", "-average-size", "-large-size", "-super-size"}
    for s in size_variants:
        if name.endswith(s):
            name = name.replace(s, "")
            break

    # Handle forme variants (e.g., keldeo-ordinary-forme)
    forme_variants = {"-ordinary-forme", "-resolute-forme"}
    for f in forme_variants:
        if name.endswith(f):
            name = name.replace(f, "")
            break

    return name


def get_base_name(df: pd.DataFrame) -> pd.DataFrame:
    """Derive base name of special variant Pokemon"""
    if "name" not in df.columns:
        raise KeyError("Missing name column")
    df = df.copy()
    df["name"] = df["name"].apply(normalize_name)
    df["base_name"] = df["name"].apply(classify_special_pokemon)
    return df 


def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Merge pokedex dataset with evolutions dataset to use evolution stage as a modeling feature"""
    if "base_name" not in df1.columns:
        raise KeyError("Missing base_name column")
    
    required2 = ["base_name", "EvoStage"]
    missing = [c for c in required2 if c not in df2.columns]

    if missing:
        raise KeyError(f"Missing required columns in df2: {missing}")
    
    df1, df2 = df1.copy(), df2.copy()

    df1 = pd.merge(
        df1,
        df2,
        on="base_name",
        how="left"
    )

    df1 = df1.rename(columns={"EvoStage": "evo_stage"})
    df1["evo_stage"] = df1["evo_stage"].astype(int)

    df1 = df1.drop(columns=["name", "base_name"])

    return df1


def validate_feature_dataset(df: pd.DataFrame) -> None:
    required = ["total_points", "generation", "log_height_m", "log_weight_kg", "is_mega", "is_primal", "is_form", "evo_stage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    forbidden = ["name", "base_name", "status", "type_1", "type_2", "growth_rate"]
    present = [c for c in forbidden if c in df.columns]
    if present:
        raise ValueError(f"Columns should have been removed or encoded: {present}")

    assert df.isna().sum().sum() == 0
    assert pd.api.types.is_integer_dtype(df["evo_stage"])

    print("Feature dataset validation passed.")
    print(df.head().to_string(index=False))


def execute_feature_engineering_pipeline(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Exeucte feature engineering pipeline"""
    df1 = df1.copy()
    # Apply log transformations to height and weight
    df1 = apply_log_transform(df1)

    # Create special variant feature 
    df1 = create_special_variant_flag(df1)

    # Filter to features used in modeling
    df1 = select_feature_columns(df1)

    # One-hot encode categorical variables 
    df1 = one_hot_encode_cols(df1)

    # Get base names of spcial variant Pokemon
    df1 = get_base_name(df1)

    # Merge evolution dataset to pokedex dataset
    df1 = merge_datasets(df1, df2)

    return df1 


if __name__ == "__main__":
    pokemon_df = pd.read_parquet(PROCESSED_DATA_DIR / "pokemon_data_clean.parquet")
    evolution_df = pd.read_csv(RAW_DATA_DIR / "pokemon_evolutions.csv")

    evolution_df = evolution_df.rename(columns={"Name": "name"})
    evolution_df = get_base_name(evolution_df)
    evolution_df = evolution_df.drop_duplicates(subset=["base_name", "EvoStage"])
    evolution_df = evolution_df[["base_name", "EvoStage"]]

    pokemon_df_features = execute_feature_engineering_pipeline(pokemon_df, evolution_df)
    validate_feature_dataset(pokemon_df_features)
