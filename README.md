# Pokemon Project

This project builds a regression model to predict a Pokemon's `total_points` from non-battle-summary features in the cleaned Pokedex dataset.

## Initial Model Features

The current baseline feature set in [notebooks/02_feature_engineering.ipynb](/Users/brandonwu/Documents/Pokemon_Project/notebooks/02_feature_engineering.ipynb) is designed for a linear model and keeps the feature space intentionally lean.

Target:
- `total_points`

Numeric features:
- `generation`
- `height_m`
- `log_weight_kg`

Categorical features to one-hot encode:
- `status`
- `type_1`
- `type_2`
- `growth_rate`

Design choices:
- `log_weight_kg` is used instead of raw `weight_kg` to reduce skew and give linear models a smoother signal.
- `base_experience` is excluded because it is too close to the prediction target and would make the problem less interpretable.
- Ability columns and matchup (`against_*`) columns are excluded from the baseline to avoid a wide, redundant feature set in the first linear model.

## Why This Is The Baseline

This feature set is meant to produce an interpretable first model before adding more expressive or higher-dimensional inputs. It favors:
- lower multicollinearity
- easier coefficient interpretation
- less risk of overfitting on a small dataset


