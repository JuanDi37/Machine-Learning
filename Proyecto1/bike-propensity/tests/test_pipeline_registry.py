from pathlib import Path

from kedro.framework.startup import bootstrap_project

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pipelines_are_registered():
    bootstrap_project(PROJECT_ROOT)
    from kedro.framework.project import pipelines

    assert {"data_processing", "model_training", "model_evaluation", "__default__"} <= set(
        pipelines
    )
    assert len(pipelines["__default__"].nodes) == 10
