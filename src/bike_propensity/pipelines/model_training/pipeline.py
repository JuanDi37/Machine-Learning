"""Pipeline de particion de datos, baselines, ajuste de hiperparametros y entrenamiento."""

from kedro.pipeline import Pipeline, node

from .nodes import fit_baselines, fit_final_model, split_data, tune_lightgbm


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``model_training``."""
    return Pipeline(
        [
            node(
                split_data,
                inputs={
                    "model_input": "model_input",
                    "group_col": "params:features.group",
                    "test_size": "params:split.test_size",
                    "random_state": "params:split.random_state",
                },
                outputs=["train_set", "test_set"],
                name="split_data_node",
            ),
            node(
                fit_baselines,
                inputs={
                    "train": "train_set",
                    "features": "params:features",
                    "n_splits": "params:cross_validation.n_splits",
                },
                outputs=["baseline_dummy", "baseline_logreg", "cv_baseline_scores"],
                name="fit_baselines_node",
            ),
            node(
                tune_lightgbm,
                inputs={
                    "train": "train_set",
                    "features": "params:features",
                    "n_splits": "params:cross_validation.n_splits",
                    "tuning": "params:tuning",
                },
                outputs=["best_params", "tuning_trials"],
                name="tune_lightgbm_node",
            ),
            node(
                fit_final_model,
                inputs={
                    "train": "train_set",
                    "features": "params:features",
                    "best_params": "best_params",
                    "random_state": "params:tuning.random_state",
                },
                outputs="final_model",
                name="fit_final_model_node",
            ),
        ],
        tags="model_training",
    )
