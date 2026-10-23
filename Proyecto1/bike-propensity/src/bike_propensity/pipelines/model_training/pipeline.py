"""Pipeline de particion de datos, baselines y entrenamiento."""

from kedro.pipeline import Pipeline, node

from .nodes import fit_baselines, split_data, tune_gradient_boosting


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
                tune_gradient_boosting,
                inputs={
                    "train": "train_set",
                    "features": "params:features",
                    "n_splits": "params:cross_validation.n_splits",
                    "tuning": "params:tuning",
                },
                outputs=["final_model", "cv_results"],
                name="tune_gradient_boosting_node",
            ),
        ],
        tags="model_training",
    )
