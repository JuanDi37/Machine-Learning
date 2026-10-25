"""Pipeline de evaluacion, interpretacion e impacto de negocio."""

from kedro.pipeline import Pipeline, node

from .nodes import (
    compute_business_impact,
    compute_lift_table,
    compute_permutation_importance,
    compute_segment_summary,
    evaluate_models,
    make_diagnostic_plots,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``model_evaluation``."""
    return Pipeline(
        [
            node(
                evaluate_models,
                inputs={
                    "test": "test_set",
                    "dummy": "baseline_dummy",
                    "logreg": "baseline_logreg",
                    "final_model": "final_model",
                    "features": "params:features",
                    "evaluation": "params:evaluation",
                },
                outputs=["metrics", "test_predictions"],
                name="evaluate_models_node",
            ),
            node(
                compute_lift_table,
                inputs={
                    "preds": "test_predictions",
                    "target": "params:features.target",
                    "n_bins": "params:evaluation.n_bins",
                },
                outputs="lift_table",
                name="compute_lift_table_node",
            ),
            node(
                compute_business_impact,
                inputs={
                    "preds": "test_predictions",
                    "target": "params:features.target",
                    "fractions": "params:evaluation.contact_fractions",
                },
                outputs="business_impact",
                name="compute_business_impact_node",
            ),
            node(
                compute_permutation_importance,
                inputs={
                    "final_model": "final_model",
                    "test": "test_set",
                    "features": "params:features",
                    "evaluation": "params:evaluation",
                },
                outputs="permutation_importance",
                name="compute_permutation_importance_node",
            ),
            node(
                compute_segment_summary,
                inputs={
                    "model_input": "model_input",
                    "target": "params:features.target",
                    "min_orders": "params:evaluation.min_segment_orders",
                },
                outputs="segment_summary",
                name="compute_segment_summary_node",
            ),
            node(
                make_diagnostic_plots,
                inputs={
                    "preds": "test_predictions",
                    "lift": "lift_table",
                    "impact": "business_impact",
                    "importance": "permutation_importance",
                    "target": "params:features.target",
                },
                outputs="diagnostic_plots",
                name="make_diagnostic_plots_node",
            ),
        ],
        tags="model_evaluation",
    )
