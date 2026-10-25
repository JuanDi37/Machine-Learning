"""Pipeline de carga, limpieza y transformacion."""

from kedro.pipeline import Pipeline, node

from .nodes import build_model_input, clean_sales


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``data_processing``."""
    return Pipeline(
        [
            node(
                clean_sales,
                inputs={
                    "raw": "raw_sales",
                    "stable_catalog_only": "params:data_processing.stable_catalog_only",
                },
                outputs=["clean_sales", "cleaning_report"],
                name="clean_sales_node",
            ),
            node(
                build_model_input,
                inputs={
                    "clean": "clean_sales",
                    "target_category": "params:data_processing.target_category",
                    "group_columns": "params:data_processing.group_columns",
                },
                outputs="model_input",
                name="build_model_input_node",
            ),
        ],
        tags="data_processing",
    )
