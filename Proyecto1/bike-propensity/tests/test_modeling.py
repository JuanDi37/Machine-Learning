import numpy as np
import pandas as pd
import pytest

from bike_propensity.pipelines.data_processing.nodes import build_model_input, clean_sales
from bike_propensity.pipelines.model_evaluation.nodes import (
    compute_business_impact,
    compute_lift_table,
)
from bike_propensity.pipelines.model_training.nodes import (
    fit_baselines,
    make_hgb_pipeline,
    split_data,
    tune_gradient_boosting,
)

FEATURES = {
    "categorical": ["Customer_Gender", "Country", "State"],
    "numeric": ["Customer_Age", "month"],
    "baseline_onehot_extra": ["month"],
    "target": "is_bike",
    "group": "split_group",
}
GROUP = ["Day", "Month", "Customer_Age", "Customer_Gender", "State"]


@pytest.fixture
def model_input(raw_sales) -> pd.DataFrame:
    clean, _ = clean_sales(raw_sales, stable_catalog_only=False)
    return build_model_input(clean, "Bikes", GROUP)


def test_split_keeps_groups_disjoint(model_input):
    train, test = split_data(model_input, "split_group", 0.25, 1)
    assert set(train["split_group"]).isdisjoint(test["split_group"])
    assert len(train) + len(test) == len(model_input)


def test_split_is_reproducible(model_input):
    a, _ = split_data(model_input, "split_group", 0.25, 1)
    b, _ = split_data(model_input, "split_group", 0.25, 1)
    pd.testing.assert_frame_equal(a, b)


def test_baselines_and_tuning_predict_probabilities(model_input):
    train, test = split_data(model_input, "split_group", 0.25, 1)
    dummy, logreg, scores = fit_baselines(train, FEATURES, n_splits=3)
    tuning = {
        "n_iter": 2,
        "random_state": 0,
        "scoring": "average_precision",
        "n_jobs": 1,
        "search_space": {"learning_rate": {"type": "loguniform", "low": 0.05, "high": 0.2}},
    }
    best, cv = tune_gradient_boosting(train, FEATURES, 3, tuning)
    cols = FEATURES["categorical"] + FEATURES["numeric"]
    for model in (dummy, logreg, best):
        p = model.predict_proba(test[cols])[:, 1]
        assert p.shape == (len(test),)
        assert ((p >= 0) & (p <= 1)).all()
    assert set(scores) == {"dummy", "logreg"}
    assert len(cv) == 2


def test_hgb_handles_unseen_category(model_input):
    cols = FEATURES["categorical"] + FEATURES["numeric"]
    model = make_hgb_pipeline(FEATURES["categorical"], FEATURES["numeric"], 0)
    model.fit(model_input[cols], model_input["is_bike"])
    new = model_input[cols].head(3).copy()
    new["State"] = "Estado nunca visto"
    assert np.isfinite(model.predict_proba(new)).all()


def test_lift_and_gains_tables():
    rng = np.random.default_rng(0)
    score = rng.random(1000)
    y = (rng.random(1000) < score).astype(int)  # el puntaje es informativo
    preds = pd.DataFrame({"is_bike": y, "Profit": rng.integers(0, 100, 1000), "p_hgb": score})
    lift = compute_lift_table(preds, "is_bike", 10)
    assert len(lift) == 10
    assert lift["lift"].iloc[0] > 1 > lift["lift"].iloc[-1]
    impact = compute_business_impact(preds, "is_bike", [0.1, 0.5, 1.0])
    assert (impact["bike_capture"] >= impact["random_capture"] - 0.05).all()
    assert impact["bike_capture"].iloc[-1] == pytest.approx(1.0)


def test_evaluation_nodes_end_to_end(model_input):
    from bike_propensity.pipelines.model_evaluation.nodes import (
        compute_permutation_importance,
        evaluate_models,
        make_diagnostic_plots,
    )

    train, test = split_data(model_input, "split_group", 0.3, 1)
    dummy, logreg, _ = fit_baselines(train, FEATURES, n_splits=3)
    tuning = {
        "n_iter": 2,
        "random_state": 0,
        "scoring": "average_precision",
        "n_jobs": 1,
        "search_space": {"learning_rate": {"type": "loguniform", "low": 0.05, "high": 0.2}},
    }
    best, _ = tune_gradient_boosting(train, FEATURES, 3, tuning)
    ev = {"random_state": 0, "permutation_repeats": 2}
    metrics, preds = evaluate_models(test, dummy, logreg, best, FEATURES, ev)
    assert metrics["dummy"]["roc_auc"] == pytest.approx(0.5)
    lo, hi = metrics["hgb"]["roc_auc_ci95"]
    assert lo <= hi
    imp = compute_permutation_importance(best, test, FEATURES, ev)
    assert set(imp["feature"]) == set(FEATURES["categorical"] + FEATURES["numeric"])
    lift = compute_lift_table(preds, "is_bike", 5)
    impact = compute_business_impact(preds, "is_bike", [0.2, 0.5])
    figs = make_diagnostic_plots(preds, lift, impact, imp, "is_bike")
    assert set(figs) == {"roc_pr_calibration.png", "lift_gains.png", "permutation_importance.png"}
