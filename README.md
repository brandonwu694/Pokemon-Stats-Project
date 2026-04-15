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

After error analysis on the initial XGBoost model, special-variant indicator features were added to the tree-based modeling pipeline:
- `is_mega`
- `is_primal`
- `is_form`

These variant indicators were introduced because several of the worst predictions involved Mega, Primal, or alternate-form Pokemon that were not being represented explicitly by the original baseline feature space.

## Modeling Plan

The project will first use this baseline feature set across progressively more complex models. New features will only be added if performance stalls and results suggest that the current feature space is not capturing enough predictive information.

## Linear Model Results

The linear-model results below were produced on the original baseline feature set before the special-variant indicators were added.

| Model | RMSE | MAE | R^2 |
| --- | ---: | ---: | ---: |
| Dummy Regressor | 124.00 | 99.92 | -0.003 |
| Linear Regression | 68.22 | 55.46 | 0.696 |
| Ridge Regression | 68.22 | 55.46 | 0.696 |
| Lasso Regression | 68.21 | 55.44 | 0.697 |

Both Ridge and Lasso regression showed negligible improvement over linear regression, suggesting that the current feature set is reasonably stable and not strongly benefiting from L2 or L1 regularization.

## Tree Based Model Results

The tree-based results below reflect the later feature set that includes the added special-variant indicators from error analysis.

| Model | RMSE | MAE | R^2 |
| --- | ---: | ---: | ---: |
| Decision Tree Regressor | 77.27 | 52.49 | 0.611 |
| Random Forest Regressor | 59.34 | 43.37 | 0.770 |
| Gradient Boosting Regressor | 58.85 | 43.56 | 0.774 |
| XGBoost Regressor | 59.06 | 42.94 | 0.773 |
| Tuned XGBoost Regressor | 58.95 | 42.31 | 0.773 |
| Tuned XGBoost Regressor + `evo_stage` | 52.54 | 37.89 | 0.820 |

The expanded tree-based feature set improved performance across the ensemble models, though the gains were initially modest. Adding `evo_stage` provided meaningful additional signal: after the tuned XGBoost model plateaued at `R^2 = 0.773`, incorporating evolutionary stage improved performance to `R^2 = 0.820` while reducing RMSE from `58.95` to `52.54` and MAE from `42.31` to `37.89`. This suggests that evolutionary progression captures important information about overall Pokemon strength that was not fully represented by typing and special-form indicators alone.

The stronger performance of the tree-based ensemble models relative to the linear models suggests that the current feature set contains nonlinear patterns and interactions that simple linear models are not fully capturing.
