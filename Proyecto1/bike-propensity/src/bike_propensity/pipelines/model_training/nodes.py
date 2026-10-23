"""Nodos de particion, modelos base y entrenamiento con ajuste de hiperparametros."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


def split_data(
    model_input: pd.DataFrame, group_col: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide en entrenamiento y prueba manteniendo juntos los registros del mismo grupo.

    Se usa ``GroupShuffleSplit`` para que los registros "gemelos" (repetidos entre
    anios) no queden repartidos entre ambos conjuntos, lo que inflaria las metricas.

    Args:
        model_input: Tabla de modelado.
        group_col: Columna con el identificador de grupo.
        test_size: Fraccion aproximada de grupos para prueba.
        random_state: Semilla.

    Returns:
        Conjuntos de entrenamiento y prueba.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(model_input, groups=model_input[group_col]))
    return (
        model_input.iloc[train_idx].reset_index(drop=True),
        model_input.iloc[test_idx].reset_index(drop=True),
    )


def make_logreg_pipeline(
    categorical: list[str], numeric: list[str], onehot_extra: list[str]
) -> Pipeline:
    """Regresion logistica con one-hot de categoricas y escalado de numericas (baseline)."""
    onehot_cols = categorical + onehot_extra
    scaled_cols = [c for c in numeric if c not in onehot_extra]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), onehot_cols),
            ("num", StandardScaler(), scaled_cols),
        ]
    )
    return Pipeline([("prep", pre), ("model", LogisticRegression(max_iter=1000))])


def make_hgb_pipeline(categorical: list[str], numeric: list[str], random_state: int) -> Pipeline:
    """Gradient boosting de histogramas con categoricas nativas (Ke et al., 2017).

    Las categoricas se codifican ordinalmente y se declaran como tales en el modelo;
    categorias no vistas en entrenamiento se tratan como valores faltantes.
    """
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                ),
                categorical,
            ),
            ("num", "passthrough", numeric),
        ]
    )
    mask = [True] * len(categorical) + [False] * len(numeric)
    model = HistGradientBoostingClassifier(
        categorical_features=mask,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state,
    )
    return Pipeline([("prep", pre), ("model", model)])


def _distribution(spec: dict):
    """Convierte la especificacion YAML de un hiperparametro en una distribucion."""
    kind = spec["type"]
    if kind == "loguniform":
        return loguniform(spec["low"], spec["high"])
    if kind == "randint":
        return randint(spec["low"], spec["high"])
    if kind == "choice":
        return list(spec["values"])
    raise ValueError(f"Tipo de distribucion no soportado: {kind}")


def fit_baselines(
    train: pd.DataFrame, features: dict, n_splits: int
) -> tuple[Pipeline, Pipeline, dict]:
    """Entrena la linea base trivial (prevalencia) y la regresion logistica.

    Tambien calcula su validacion cruzada agrupada sobre el entrenamiento.

    Args:
        train: Conjunto de entrenamiento.
        features: Bloque ``features`` de parametros.
        n_splits: Numero de pliegues de la validacion cruzada.

    Returns:
        Modelo dummy, regresion logistica y puntajes de validacion cruzada.
    """
    cols = features["categorical"] + features["numeric"]
    x, y, groups = train[cols], train[features["target"]], train[features["group"]]
    dummy = DummyClassifier(strategy="prior")
    logreg = make_logreg_pipeline(
        features["categorical"], features["numeric"], features["baseline_onehot_extra"]
    )
    cv = GroupKFold(n_splits=n_splits)
    scoring = ["roc_auc", "average_precision", "neg_brier_score"]
    scores = {}
    for name, est in [("dummy", dummy), ("logreg", logreg)]:
        res = cross_validate(est, x, y, groups=groups, cv=cv, scoring=scoring)
        scores[name] = {
            m: [float(np.mean(res[f"test_{m}"])), float(np.std(res[f"test_{m}"]))] for m in scoring
        }
    dummy.fit(x, y)
    logreg.fit(x, y)
    return dummy, logreg, scores


def tune_gradient_boosting(
    train: pd.DataFrame, features: dict, n_splits: int, tuning: dict
) -> tuple[Pipeline, pd.DataFrame]:
    """Ajusta hiperparametros del gradient boosting con busqueda aleatoria agrupada.

    Args:
        train: Conjunto de entrenamiento.
        features: Bloque ``features`` de parametros.
        n_splits: Numero de pliegues (``GroupKFold``).
        tuning: Bloque ``tuning`` (n_iter, scoring, espacio de busqueda, semilla).

    Returns:
        El mejor pipeline reentrenado con todo el entrenamiento y la tabla de resultados.
    """
    cols = features["categorical"] + features["numeric"]
    base = make_hgb_pipeline(features["categorical"], features["numeric"], tuning["random_state"])
    space = {f"model__{k}": _distribution(v) for k, v in tuning["search_space"].items()}
    scoring = {"average_precision": "average_precision", "roc_auc": "roc_auc"}
    search = RandomizedSearchCV(
        base,
        space,
        n_iter=tuning["n_iter"],
        scoring=scoring,
        refit=tuning["scoring"],
        cv=GroupKFold(n_splits=n_splits),
        n_jobs=tuning["n_jobs"],
        random_state=tuning["random_state"],
    )
    search.fit(train[cols], train[features["target"]], groups=train[features["group"]])
    logger.info("Mejores hiperparametros: %s", search.best_params_)
    results = pd.DataFrame(search.cv_results_)
    keep = [
        c for c in results.columns if c.startswith(("param_", "mean_test", "std_test", "rank_test"))
    ]
    return search.best_estimator_, results[keep].sort_values("rank_test_average_precision")
