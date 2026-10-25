"""Nodos de evaluacion, interpretacion e impacto de negocio."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla: necesario en CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from bike_propensity.pipelines.model_training.nodes import prepare_baseline_frame  # noqa: E402

AGE_BINS = [0, 24, 34, 44, 54, 120]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]


def _score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Calcula las metricas de clasificacion probabilistica usadas en el proyecto."""
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y, p)),
    }


def _group_bootstrap(
    y: np.ndarray, probs: dict[str, np.ndarray], groups: np.ndarray, n_boot: int, seed: int
) -> dict[str, dict[str, list[float]]]:
    """Intervalos de confianza del 95 % por bootstrap de grupos completos.

    Se remuestrean grupos (no filas) para respetar la dependencia entre registros gemelos.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    _, start = np.unique(groups[order], return_index=True)
    chunks = np.split(order, start[1:])
    draws: dict[str, list[tuple[float, float]]] = {name: [] for name in probs}
    for _ in range(n_boot):
        idx = np.concatenate([chunks[i] for i in rng.integers(0, len(chunks), len(chunks))])
        for name, p in probs.items():
            draws[name].append(
                (roc_auc_score(y[idx], p[idx]), average_precision_score(y[idx], p[idx]))
            )
    out: dict[str, dict[str, list[float]]] = {}
    for name, values in draws.items():
        arr = np.array(values)
        out[name] = {
            "roc_auc_ci95": np.percentile(arr[:, 0], [2.5, 97.5]).round(4).tolist(),
            "pr_auc_ci95": np.percentile(arr[:, 1], [2.5, 97.5]).round(4).tolist(),
        }
    return out


def evaluate_models(
    test: pd.DataFrame, dummy, logreg, final_model, features: dict, evaluation: dict
) -> tuple[dict, pd.DataFrame]:
    """Evalua en el conjunto de prueba el modelo final y las lineas base.

    Args:
        test: Conjunto de prueba.
        dummy: Linea base de prevalencia.
        logreg: Regresion logistica.
        final_model: Pipeline LightGBM entrenado.
        features: Diccionario de variables (``categorical``, ``numeric``, ...).
        evaluation: Parametros de evaluacion (``n_bootstrap``, ``random_state``).

    Returns:
        Un diccionario de metricas con intervalos de confianza y la tabla de predicciones.
    """
    cols = features["categorical"] + features["numeric"]
    y = test[features["target"]].to_numpy()
    X_base = prepare_baseline_frame(test[cols], features["baseline_onehot_extra"])
    probs = {
        "baseline": dummy.predict_proba(X_base)[:, 1],
        "logistic_regression": logreg.predict_proba(X_base)[:, 1],
        "lightgbm_final": final_model.predict_proba(test[cols])[:, 1],
    }
    metrics = {name: _score(y, p) for name, p in probs.items()}
    ci = _group_bootstrap(
        y,
        {k: probs[k] for k in ("logistic_regression", "lightgbm_final")},
        test[features["group"]].to_numpy(),
        evaluation["n_bootstrap"],
        evaluation["random_state"],
    )
    for name, bounds in ci.items():
        metrics[name].update(bounds)
    metrics["n_test"] = int(len(test))
    metrics["prevalence_test"] = float(y.mean())

    preds = test.copy()
    preds["score"] = probs["lightgbm_final"]
    preds["score_logreg"] = probs["logistic_regression"]
    return metrics, preds


def compute_lift_table(preds: pd.DataFrame, target: str, n_bins: int) -> pd.DataFrame:
    """Tabla de lift por decil de puntaje (1 = mayor propension).

    Args:
        preds: Predicciones con ``score`` y la columna objetivo.
        target: Nombre de la columna objetivo.
        n_bins: Numero de grupos (10 = deciles).

    Returns:
        Tasa observada, puntaje medio y lift de cada decil.
    """
    ordered = preds.sort_values("score", ascending=False).reset_index(drop=True)
    ordered["decile"] = pd.qcut(np.arange(len(ordered)), n_bins, labels=range(1, n_bins + 1))
    table = ordered.groupby("decile", observed=True).agg(
        orders=(target, "size"), bike_rate=(target, "mean"), mean_score=("score", "mean")
    )
    table["lift"] = table["bike_rate"] / ordered[target].mean()
    return table.reset_index()


def compute_business_impact(
    preds: pd.DataFrame, target: str, fractions: list[float]
) -> pd.DataFrame:
    """Calcula que fraccion de bicicletas y de ganancia se captura al contactar al top-k %.

    Args:
        preds: Predicciones con ``score``, ``ganancia`` y la columna objetivo.
        target: Nombre de la columna objetivo.
        fractions: Fracciones de la base a contactar (p. ej. 0.1, 0.2, 0.3).

    Returns:
        Una fila por fraccion con precision, lift y porcentajes capturados.
    """
    ordered = preds.sort_values("score", ascending=False).reset_index(drop=True)
    bikes_total = ordered[target].sum()
    bike_profit_total = ordered.loc[ordered[target] == 1, "ganancia"].sum()
    profit_total = ordered["ganancia"].sum()
    rows = []
    for frac in fractions:
        top = ordered.iloc[: int(round(len(ordered) * frac))]
        is_bike = top[target] == 1
        rows.append(
            {
                "contact_fraction": frac,
                "precision": float(top[target].mean()),
                "lift": float(top[target].mean() / ordered[target].mean()),
                "bikes_captured": float(top[target].sum() / bikes_total),
                "bike_profit_captured": float(
                    top.loc[is_bike, "ganancia"].sum() / bike_profit_total
                ),
                "total_profit_captured": float(top["ganancia"].sum() / profit_total),
            }
        )
    return pd.DataFrame(rows)


def compute_permutation_importance(
    final_model, test: pd.DataFrame, features: dict, evaluation: dict
) -> pd.DataFrame:
    """Importancia por permutacion (caida del ROC-AUC en prueba al desordenar cada variable).

    Args:
        final_model: Pipeline LightGBM entrenado.
        test: Conjunto de prueba.
        features: Diccionario de variables.
        evaluation: Parametros (``permutation_repeats``, ``random_state``).

    Returns:
        Importancia media y desviacion de cada predictor, ordenada de mayor a menor.
    """
    cols = features["categorical"] + features["numeric"]
    result = permutation_importance(
        final_model,
        test[cols],
        test[features["target"]].to_numpy(),
        scoring="roc_auc",
        n_repeats=evaluation["permutation_repeats"],
        random_state=evaluation["random_state"],
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def compute_segment_summary(
    model_input: pd.DataFrame, target: str, min_orders: int
) -> pd.DataFrame:
    """Resume la tasa de bicicleta y la ganancia por pais, tramo de edad y genero.

    Es la base de las recomendaciones comerciales por edad, genero y mercado.

    Args:
        model_input: Tabla de modelado completa.
        target: Nombre de la columna objetivo.
        min_orders: Numero minimo de pedidos para conservar un segmento.

    Returns:
        Una fila por segmento con pedidos, tasa de bicicleta, lift y ganancia media.
    """
    df = model_input.copy()
    df["tramo_edad"] = pd.cut(df["edad"], AGE_BINS, labels=AGE_LABELS)
    seg = (
        df.groupby(["pais", "tramo_edad", "genero"], observed=True)
        .agg(orders=(target, "size"), bike_rate=(target, "mean"), mean_profit=("ganancia", "mean"))
        .reset_index()
    )
    seg = seg[seg["orders"] >= min_orders].copy()
    seg["lift"] = seg["bike_rate"] / df[target].mean()
    return seg.sort_values("bike_rate", ascending=False).reset_index(drop=True)


def make_diagnostic_plots(
    preds: pd.DataFrame,
    lift: pd.DataFrame,
    impact: pd.DataFrame,
    importance: pd.DataFrame,
    target: str,
) -> dict[str, plt.Figure]:
    """Genera los graficos de diagnostico: ROC, calibracion, lift, ganancias e importancia.

    Args:
        preds: Predicciones con ``score`` y ``score_logreg``.
        lift: Tabla de lift por decil.
        impact: Tabla de impacto de negocio.
        importance: Importancia por permutacion.
        target: Nombre de la columna objetivo.

    Returns:
        Un diccionario ``nombre -> figura`` que Kedro guarda como PNG.
    """
    y = preds[target].to_numpy()
    figs: dict[str, plt.Figure] = {}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for col, label in (("score", "LightGBM"), ("score_logreg", "Regresion logistica")):
        fpr, tpr, _ = roc_curve(y, preds[col])
        axes[0].plot(fpr, tpr, label=f"{label} ({roc_auc_score(y, preds[col]):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(
        xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="Curva ROC"
    )
    axes[0].legend()
    frac_pos, mean_pred = calibration_curve(y, preds["score"], n_bins=10, strategy="quantile")
    axes[1].plot(mean_pred, frac_pos, marker="o")
    axes[1].plot([0, mean_pred.max()], [0, mean_pred.max()], "--", color="grey")
    axes[1].set(xlabel="Probabilidad predicha", ylabel="Frecuencia observada", title="Calibracion")
    fig.tight_layout()
    figs["roc_calibration"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(lift["decile"].astype(int), lift["lift"])
    axes[0].axhline(1.0, ls="--", color="grey")
    axes[0].set(
        xlabel="Decil de puntaje (1 = mayor propension)", ylabel="Lift", title="Lift por decil"
    )
    axes[1].plot(impact["contact_fraction"] * 100, impact["bikes_captured"] * 100, marker="o")
    axes[1].plot(
        [0, 100 * impact["contact_fraction"].max()],
        [0, 100 * impact["contact_fraction"].max()],
        "--",
        color="grey",
    )
    axes[1].set(
        xlabel="% de pedidos contactados",
        ylabel="% de bicicletas capturadas",
        title="Curva de ganancias",
    )
    fig.tight_layout()
    figs["lift_gains"] = fig

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ordered = importance.sort_values("importance_mean")
    ax.barh(ordered["feature"], ordered["importance_mean"], xerr=ordered["importance_std"])
    ax.set(xlabel="Caida del ROC-AUC al permutar", title="Importancia por permutacion")
    fig.tight_layout()
    figs["permutation_importance"] = fig
    return figs
