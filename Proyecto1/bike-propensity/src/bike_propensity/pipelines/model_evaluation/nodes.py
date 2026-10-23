"""Nodos de evaluacion, interpretacion e impacto de negocio."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

MODEL_LABELS = {
    "dummy": "Prevalencia (baseline trivial)",
    "logreg": "Regresion logistica (baseline)",
    "hgb": "Gradient boosting (final)",
}


def _group_bootstrap_ci(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_boot: int, seed: int
) -> dict[str, list[float]]:
    """Intervalo de confianza 95 % por bootstrap de grupos para ROC-AUC y PR-AUC."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.asarray)
    aucs, aps = [], []
    for _ in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in sample])
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        aps.append(average_precision_score(y[idx], p[idx]))
    return {
        "roc_auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "pr_auc_ci95": [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))],
    }


def evaluate_models(
    test: pd.DataFrame,
    dummy,
    logreg,
    final_model,
    features: dict,
    evaluation: dict,
) -> tuple[dict, pd.DataFrame]:
    """Calcula metricas en el conjunto de prueba para baselines y modelo final.

    Args:
        test: Conjunto de prueba (no usado en entrenamiento ni en ajuste).
        dummy: Baseline de prevalencia.
        logreg: Regresion logistica.
        final_model: Pipeline de gradient boosting ajustado.
        features: Bloque ``features``.
        evaluation: Bloque ``evaluation``.

    Returns:
        Diccionario de metricas y tabla de predicciones de prueba.
    """
    cols = features["categorical"] + features["numeric"]
    x, y = test[cols], test[features["target"]].to_numpy()
    groups = test[features["group"]].to_numpy()
    preds = test.copy()
    metrics: dict = {
        "n_test": int(len(test)),
        "prevalence_test": float(y.mean()),
    }
    for key, model in [("dummy", dummy), ("logreg", logreg), ("hgb", final_model)]:
        p = model.predict_proba(x)[:, 1]
        preds[f"p_{key}"] = p
        metrics[key] = {
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p)),
        }
    metrics["hgb"].update(
        _group_bootstrap_ci(y, preds["p_hgb"].to_numpy(), groups, 200, evaluation["random_state"])
    )
    metrics["hgb_best_params"] = {
        k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
        for k, v in final_model.get_params().items()
        if k
        in {
            f"model__{n}"
            for n in (
                "learning_rate",
                "max_iter",
                "max_leaf_nodes",
                "min_samples_leaf",
                "l2_regularization",
                "max_depth",
            )
        }  # fmt: skip
    }
    logger.info("Metricas de prueba: %s", metrics)
    return metrics, preds


def compute_lift_table(preds: pd.DataFrame, target: str, n_bins: int) -> pd.DataFrame:
    """Tabla de deciles: tasa de compra de bicicleta y lift por tramo de puntaje.

    El decil 1 contiene las observaciones con mayor probabilidad predicha.
    """
    df = preds[[target, "Profit", "p_hgb"]].copy()
    df["decile"] = (
        pd.qcut(df["p_hgb"].rank(method="first", ascending=False), n_bins, labels=False) + 1
    )
    prevalence = df[target].mean()
    table = (
        df.groupby("decile")
        .agg(n=(target, "size"), bike_rate=(target, "mean"), mean_profit=("Profit", "mean"),
             mean_score=("p_hgb", "mean"))
        .reset_index()
    )  # fmt: skip
    table["lift"] = table["bike_rate"] / prevalence
    return table


def compute_business_impact(
    preds: pd.DataFrame, target: str, fractions: list[float]
) -> pd.DataFrame:
    """Curva de ganancias: que se captura al contactar el top-f% segun el modelo.

    Compara contra contactar al azar (captura esperada = f). La ganancia (``Profit``)
    es la observada en las ordenes de prueba; no es un pronostico de campana.
    """
    df = preds.sort_values("p_hgb", ascending=False).reset_index(drop=True)
    total_bikes, total_profit = df[target].sum(), df["Profit"].sum()
    rows = []
    for f in fractions:
        top = df.head(int(round(f * len(df))))
        rows.append(
            {
                "contact_fraction": f,
                "bike_capture": top[target].sum() / total_bikes,
                "random_capture": f,
                "precision": top[target].mean(),
                "lift": top[target].mean() / df[target].mean(),
                "profit_capture": top["Profit"].sum() / total_profit,
                "mean_profit_selected": top["Profit"].mean(),
                "mean_profit_all": df["Profit"].mean(),
            }
        )
    return pd.DataFrame(rows)


def compute_permutation_importance(
    final_model, test: pd.DataFrame, features: dict, evaluation: dict
) -> pd.DataFrame:
    """Importancia por permutacion (caida de ROC-AUC) sobre el conjunto de prueba.

    Metodo model-agnostic de Breiman (2001), formalizado como *model reliance* por
    Fisher et al. (2019).
    """
    cols = features["categorical"] + features["numeric"]
    res = permutation_importance(
        final_model,
        test[cols],
        test[features["target"]],
        scoring="roc_auc",
        n_repeats=evaluation["permutation_repeats"],
        random_state=evaluation["random_state"],
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": cols,
                "auc_drop_mean": res.importances_mean,
                "auc_drop_std": res.importances_std,
            }
        )
        .sort_values("auc_drop_mean", ascending=False)
        .reset_index(drop=True)
    )


def make_diagnostic_plots(
    preds: pd.DataFrame,
    lift: pd.DataFrame,
    impact: pd.DataFrame,
    importance: pd.DataFrame,
    target: str,
) -> dict[str, plt.Figure]:
    """Genera las figuras de diagnostico (ROC, PR, calibracion, lift, ganancias, importancia)."""
    y = preds[target].to_numpy()
    figs: dict[str, plt.Figure] = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for key, label in MODEL_LABELS.items():
        p = preds[f"p_{key}"]
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, p):.3f})")
        if (
            key != "dummy"
        ):  # el baseline trivial tiene puntaje constante: su curva PR es un artefacto
            prec, rec, _ = precision_recall_curve(y, p)
            axes[1].plot(rec, prec, label=f"{label} (AP={average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0].set(
        xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="Curva ROC"
    )
    axes[1].axhline(y.mean(), color="k", ls=":", lw=1, label=f"Prevalencia ({y.mean():.3f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    for key in ("logreg", "hgb"):
        frac, mean_pred = calibration_curve(y, preds[f"p_{key}"], n_bins=10, strategy="quantile")
        axes[2].plot(mean_pred, frac, "o-", label=MODEL_LABELS[key])
    axes[2].plot([0, 0.6], [0, 0.6], "k:", lw=1)
    axes[2].set(xlabel="Probabilidad predicha", ylabel="Fraccion observada", title="Calibracion")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    figs["roc_pr_calibration.png"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(lift["decile"], lift["lift"], color="#2a6f97")
    axes[0].axhline(1, color="k", ls=":", lw=1)
    axes[0].set(
        xlabel="Decil de puntaje (1 = mayor probabilidad)",
        ylabel="Lift vs. promedio",
        title="Lift por decil",
    )
    axes[1].plot(impact["contact_fraction"], impact["bike_capture"], "o-", label="Modelo")
    axes[1].plot(impact["contact_fraction"], impact["random_capture"], "k:", label="Azar")
    axes[1].set(
        xlabel="Fraccion contactada",
        ylabel="Fraccion de compras de bicicleta capturadas",
        title="Curva de ganancias",
    )
    axes[1].legend()
    fig.tight_layout()
    figs["lift_gains.png"] = fig

    fig, ax = plt.subplots(figsize=(6, 3.6))
    imp = importance.sort_values("auc_drop_mean")
    ax.barh(imp["feature"], imp["auc_drop_mean"], xerr=imp["auc_drop_std"], color="#2a6f97")
    ax.set(xlabel="Caida de ROC-AUC al permutar", title="Importancia por permutacion (prueba)")
    fig.tight_layout()
    figs["permutation_importance.png"] = fig
    return figs
