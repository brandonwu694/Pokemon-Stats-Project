# Pokemon Project

This project builds a regression model to predict a Pokemon's `total_points` from non-battle-summary features in the cleaned Pokedex dataset.

## Initial Model Features

Target:
- `total_points`

Numeric features:
- `generation`
- `log_height_m`
- `log_weight_kg`

Categorical features to one-hot encode:
- `status`
- `type_1`
- `type_2`
- `growth_rate`

Design choices:
- `log_weight_kg` is used instead of raw `weight_kg` to reduce skew and give linear models a smoother signal.
- `log_height_m` is used instead of raw `height_m` to reduce skew for a cleaner linear model input.

This feature set is meant to produce an interpretable first model before adding more expressive or higher-dimensional inputs. It favors:
- lower multicollinearity
- easier coefficient interpretation
- less risk of overfitting on a small dataset

In earlier stages of feature engineering, several features were proposed in hopes to give the model more context, but many turned out to be repetitive. For example, a feature such as `is_dual_type` was eliminated because a Pokemon's typing was already represented through features `type_1` and `type_2`. 

## Modeling Plan

The project will first use this baseline feature set across progressively more complex models. New features will only be added if performance stalls and results suggest that the current feature space is not capturing enough predictive information.

## Baseline Results

Initial regression baseline using `DummyRegressor(strategy="mean")`:

| Model | RMSE | MAE | R^2 |
| --- | ---: | ---: | ---: |
| Dummy Regressor | 124.00 | 99.92 | -0.003 |
