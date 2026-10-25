"""Nodos de particion, lineas base, ajuste de hiperparametros y entrenamiento final."""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def split_data(
    model_input: pd.DataFrame, group_col: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte la tabla en entrenamiento y prueba sin repartir grupos de registros gemelos.

    Args:
        model_input: Tabla de modelado.
        group_col: Columna que identifica el grupo de cada fila.
        test_size: Fraccion de grupos reservada para prueba.
        random_state: Semilla de la particion.

    Returns:
        Los conjuntos de entrenamiento y prueba.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_train, idx_test = next(splitter.split(model_input, groups=model_input[group_col]))
    train = model_input.iloc[idx_train].reset_index(drop=True)
    test = model_input.iloc[idx_test].reset_index(drop=True)
    return train, test


def build_lightgbm_pipeline(
    categorical: list[str], numeric: list[str], params: dict, random_state: int
) -> Pipeline:
    """Construye el pipeline LightGBM con categoricas nativas.

    Las columnas numericas van primero y las categoricas, codificadas como enteros (las
    categorias no vistas pasan a faltante), despues; se declaran categoricas para LightGBM
    mediante ``categorical_feature``.

    Args:
        categorical: Columnas categoricas.
        numeric: Columnas numericas.
        params: Hiperparametros de ``LGBMClassifier``.
        random_state: Semilla del modelo.

    Returns:
        Un ``Pipeline`` sin entrenar. Debe ajustarse con ``fit_lightgbm``.
    """
    encoder = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                categorical,
            ),
        ]
    )
    model = LGBMClassifier(
        **params,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
        deterministic=True,
        force_row_wise=True,
    )
    return Pipeline([("encoder", encoder), ("lgbm", model)])


def fit_lightgbm(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, features: dict) -> Pipeline:
    """Ajusta el pipeline declarando como categoricas las columnas categoricas codificadas.

    Args:
        pipeline: Pipeline creado con ``build_lightgbm_pipeline``.
        X: Predictores.
        y: Objetivo binario.
        features: Diccionario de variables (``categorical`` y ``numeric``).

    Returns:
        El mismo pipeline, ajustado.
    """
    n_num = len(features["numeric"])
    categorical_idx = list(range(n_num, n_num + len(features["categorical"])))
    return pipeline.fit(X, y, lgbm__categorical_feature=categorical_idx)


def _logit_pipeline(features: dict) -> Pipeline:
    """Regresion logistica con one-hot y spline de edad (linea base interpretable)."""
    transformer = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                features["categorical"] + features["baseline_onehot_extra"],
            ),
            ("edad", SplineTransformer(n_knots=6, degree=3), ["edad"]),
        ]
    )
    return make_pipeline(transformer, LogisticRegression(C=0.2, max_iter=3000))


def prepare_baseline_frame(X: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    """Convierte a texto las variables numericas que la logistica trata como categoricas."""
    X = X.copy()
    for col in extra:
        X[col] = X[col].astype(str)
    return X


def fit_baselines(
    train: pd.DataFrame, features: dict, n_splits: int
) -> tuple[DummyClassifier, Pipeline, dict]:
    """Ajusta la linea base de prevalencia y la regresion logistica, con CV agrupada.

    Args:
        train: Conjunto de entrenamiento.
        features: Diccionario con ``categorical``, ``numeric``, ``baseline_onehot_extra``,
            ``target`` y ``group``.
        n_splits: Numero de pliegues de la validacion cruzada.

    Returns:
        Modelo de prevalencia, regresion logistica y las metricas medias de validacion cruzada.
    """
    cols = features["categorical"] + features["numeric"]
    extra = features["baseline_onehot_extra"]
    X = prepare_baseline_frame(train[cols], extra)
    y = train[features["target"]].to_numpy()
    groups = train[features["group"]].to_numpy()

    cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=0)
    scores: dict[str, dict[str, float]] = {"dummy": {}, "logreg": {}}
    for name, factory in (
        ("dummy", lambda: DummyClassifier(strategy="prior")),
        ("logreg", lambda: _logit_pipeline(features)),
    ):
        roc, pr = [], []
        for a, b in cv.split(X, y, groups):
            p = factory().fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
            roc.append(roc_auc_score(y[b], p))
            pr.append(average_precision_score(y[b], p))
        scores[name] = {"roc_auc_mean": float(np.mean(roc)), "pr_auc_mean": float(np.mean(pr))}

    dummy = DummyClassifier(strategy="prior").fit(X, y)
    logreg = _logit_pipeline(features).fit(X, y)
    return dummy, logreg, scores


def _sample_params(trial: optuna.Trial, space: dict) -> dict:
    """Muestrea un conjunto de hiperparametros segun el espacio definido en parametros."""
    params: dict = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif kind == "loguniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        else:
            raise ValueError(f"Tipo de hiperparametro desconocido: {kind}")
    return params


def tune_lightgbm(
    train: pd.DataFrame, features: dict, n_splits: int, tuning: dict
) -> tuple[dict, pd.DataFrame]:
    """Ajusta los hiperparametros de LightGBM con Optuna (TPE) y CV agrupada.

    Args:
        train: Conjunto de entrenamiento.
        features: Diccionario de variables (ver ``fit_baselines``).
        n_splits: Pliegues de la validacion cruzada.
        tuning: Parametros de la busqueda: ``n_trials``, ``random_state``, ``cv_seed``,
            ``fixed_params`` y ``search_space``.

    Returns:
        Los mejores hiperparametros y la tabla con todas las pruebas.
    """
    cols = features["categorical"] + features["numeric"]
    X = train[cols]
    y = train[features["target"]].to_numpy()
    groups = train[features["group"]].to_numpy()
    folds = list(
        StratifiedGroupKFold(n_splits, shuffle=True, random_state=tuning["cv_seed"]).split(
            X, y, groups
        )
    )
    fixed = dict(tuning["fixed_params"])

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed, **_sample_params(trial, tuning["search_space"])}
        scores = []
        for a, b in folds:
            pipe = build_lightgbm_pipeline(
                features["categorical"], features["numeric"], params, tuning["random_state"]
            )
            fit_lightgbm(pipe, X.iloc[a], y[a], features)
            scores.append(average_precision_score(y[b], pipe.predict_proba(X.iloc[b])[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=tuning["random_state"])
    )
    study.optimize(objective, n_trials=tuning["n_trials"])
    best = {**fixed, **study.best_params}
    logger.info("Mejor PR-AUC en CV: %.4f", study.best_value)
    return best, study.trials_dataframe()


def fit_final_model(
    train: pd.DataFrame, features: dict, best_params: dict, random_state: int
) -> Pipeline:
    """Entrena el modelo final con todo el conjunto de entrenamiento.

    Args:
        train: Conjunto de entrenamiento.
        features: Diccionario de variables (ver ``fit_baselines``).
        best_params: Hiperparametros elegidos por ``tune_lightgbm``.
        random_state: Semilla del modelo.

    Returns:
        El pipeline LightGBM entrenado.
    """
    cols = features["categorical"] + features["numeric"]
    pipe = build_lightgbm_pipeline(
        features["categorical"], features["numeric"], best_params, random_state
    )
    return fit_lightgbm(pipe, train[cols], train[features["target"]].to_numpy(), features)
