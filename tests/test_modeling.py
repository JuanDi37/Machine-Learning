import numpy as np
import pandas as pd
import pytest

from bike_propensity.pipelines.data_processing.nodes import build_model_input, clean_sales
from bike_propensity.pipelines.model_evaluation.nodes import (
    compute_business_impact,
    compute_lift_table,
    compute_permutation_importance,
    compute_segment_summary,
    evaluate_models,
)
from bike_propensity.pipelines.model_training.nodes import (
    fit_baselines,
    fit_final_model,
    split_data,
    tune_lightgbm,
)

FEATURES = {
    "categorical": ["genero", "pais", "estado"],
    "numeric": ["edad", "mes"],
    "baseline_onehot_extra": ["mes"],
    "target": "es_bicicleta",
    "group": "grupo",
}
GROUP = ["Day", "Month", "Customer_Age", "Customer_Gender", "State"]
TUNING = {
    "n_trials": 2,
    "random_state": 0,
    "cv_seed": 1,
    "fixed_params": {"subsample_freq": 1},
    "search_space": {
        "n_estimators": {"type": "int", "low": 20, "high": 40},
        "learning_rate": {"type": "loguniform", "low": 0.05, "high": 0.2},
        "num_leaves": {"type": "int", "low": 4, "high": 8},
        "min_child_samples": {"type": "int", "low": 5, "high": 20},
        "min_data_per_group": {"type": "int", "low": 5, "high": 20},
    },
}
EVALUATION = {"n_bootstrap": 20, "random_state": 0, "permutation_repeats": 2}


@pytest.fixture
def model_input(raw_sales) -> pd.DataFrame:
    clean, _ = clean_sales(raw_sales, stable_catalog_only=False)
    return build_model_input(clean, "Bikes", GROUP)


def test_split_keeps_groups_disjoint(model_input):
    train, test = split_data(model_input, "grupo", 0.25, 1)
    assert set(train["grupo"]).isdisjoint(test["grupo"])
    assert len(train) + len(test) == len(model_input)


def test_split_is_reproducible(model_input):
    a, _ = split_data(model_input, "grupo", 0.25, 1)
    b, _ = split_data(model_input, "grupo", 0.25, 1)
    pd.testing.assert_frame_equal(a, b)


@pytest.fixture
def trained(model_input):
    train, test = split_data(model_input, "grupo", 0.25, 1)
    dummy, logreg, scores = fit_baselines(train, FEATURES, n_splits=3)
    best, trials = tune_lightgbm(train, FEATURES, 3, TUNING)
    model = fit_final_model(train, FEATURES, best, random_state=0)
    return train, test, dummy, logreg, scores, best, trials, model


def test_baselines_report_cv_scores(trained):
    scores = trained[4]
    assert set(scores) == {"dummy", "logreg"}
    assert scores["dummy"]["roc_auc_mean"] == pytest.approx(0.5)


def test_tuning_returns_params_and_all_trials(trained):
    best, trials = trained[5], trained[6]
    assert trials.shape[0] == TUNING["n_trials"]
    assert best["subsample_freq"] == 1
    assert 20 <= best["n_estimators"] <= 40


def test_final_model_predicts_probabilities_and_handles_unseen_state(trained):
    test, model = trained[1], trained[7]
    cols = FEATURES["categorical"] + FEATURES["numeric"]
    unseen = test[cols].head(5).copy()
    unseen["estado"] = "Estado inexistente"
    p = model.predict_proba(unseen)[:, 1]
    assert p.shape == (5,)
    assert ((p >= 0) & (p <= 1)).all()


def test_evaluate_models_returns_metrics_with_ci(trained):
    _, test, dummy, logreg, _, _, _, model = trained
    metrics, preds = evaluate_models(test, dummy, logreg, model, FEATURES, EVALUATION)
    assert metrics["baseline"]["roc_auc"] == pytest.approx(0.5)
    low, high = metrics["lightgbm_final"]["roc_auc_ci95"]
    assert low <= high
    assert {"score", "score_logreg", "ganancia"} <= set(preds.columns)


def test_lift_and_business_impact(trained):
    _, test, dummy, logreg, _, _, _, model = trained
    _, preds = evaluate_models(test, dummy, logreg, model, FEATURES, EVALUATION)
    lift = compute_lift_table(preds, "es_bicicleta", n_bins=5)
    assert len(lift) == 5
    assert lift["orders"].sum() == len(preds)
    impact = compute_business_impact(preds, "es_bicicleta", [0.5, 1.0])
    assert impact["bikes_captured"].iloc[-1] == pytest.approx(1.0)
    assert np.all(np.diff(impact["bikes_captured"]) >= 0)


def test_importance_and_segments(trained, model_input):
    test, model = trained[1], trained[7]
    imp = compute_permutation_importance(model, test, FEATURES, EVALUATION)
    assert set(imp["feature"]) == set(FEATURES["categorical"] + FEATURES["numeric"])
    seg = compute_segment_summary(model_input, "es_bicicleta", min_orders=1)
    assert seg["orders"].sum() == len(model_input)
    assert seg["bike_rate"].between(0, 1).all()
