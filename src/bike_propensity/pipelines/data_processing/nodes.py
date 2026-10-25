"""Nodos de carga, limpieza y construccion de la tabla de modelado."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Date", "Day", "Month", "Year", "Customer_Age", "Age_Group", "Customer_Gender",
    "Country", "State", "Product_Category", "Sub_Category", "Product",
    "Order_Quantity", "Unit_Cost", "Unit_Price", "Profit", "Cost", "Revenue",
]  # fmt: skip


def standardize_catalog_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Corrige categoria y subcategoria con el valor mas frecuente de cada producto.

    Cada producto tiene un unico precio y costo, pero unas pocas filas traen una
    categoria o subcategoria distinta a la del resto de sus ventas. Se asume que la
    etiqueta mayoritaria es la correcta y se reporta el numero de filas corregidas.

    Args:
        df: Tabla con ``Product``, ``Product_Category`` y ``Sub_Category``.

    Returns:
        La tabla corregida y el conteo de filas modificadas por columna.
    """
    modal = df.groupby("Product")[["Product_Category", "Sub_Category"]].agg(
        lambda s: s.value_counts().index[0]
    )
    df = df.copy()
    new_cat = df["Product"].map(modal["Product_Category"])
    new_sub = df["Product"].map(modal["Sub_Category"])
    report = {
        "category_labels_fixed_rows": int((df["Product_Category"] != new_cat).sum()),
        "subcategory_labels_fixed_rows": int((df["Sub_Category"] != new_sub).sum()),
    }
    df["Product_Category"], df["Sub_Category"] = new_cat, new_sub
    return df, report


def keep_stable_catalog_months(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo los meses con catalogo completo y estable.

    Un mes se conserva si contiene todas las categorias de producto **y** el mes
    calendario anterior tambien las contenia. Esto excluye de forma automatica el periodo
    en que solo se vendian bicicletas, el mes de lanzamiento de las demas categorias y los
    meses en que falta alguna categoria.

    Args:
        df: Tabla con ``Date`` y ``Product_Category``.

    Returns:
        Las filas pertenecientes a meses estables.
    """
    month = df["Date"].dt.to_period("M")
    calendar = pd.period_range(month.min(), month.max(), freq="M")
    n_categories = df["Product_Category"].nunique()
    full = (
        df.groupby(month)["Product_Category"].nunique().reindex(calendar, fill_value=0)
        == n_categories
    )
    stable = full & full.shift(1, fill_value=False)
    return df[month.map(stable).fillna(False).astype(bool)]


def clean_sales(raw: pd.DataFrame, stable_catalog_only: bool) -> tuple[pd.DataFrame, dict]:
    """Limpia la tabla cruda de ventas y genera un reporte de calidad de datos.

    Pasos: validacion de esquema, parseo de fecha, normalizacion de texto, conteo de
    faltantes, eliminacion de duplicados exactos, correccion de etiquetas de categoria,
    chequeo de consistencia aritmetica (``Cost = Unit_Cost * Qty``; ``Profit = Revenue -
    Cost``) y, opcionalmente, exclusion de meses con catalogo incompleto o en transicion.

    Args:
        raw: Datos tal como se leen del CSV.
        stable_catalog_only: Si es ``True``, solo se conservan meses de catalogo estable.

    Returns:
        La tabla limpia y un diccionario con los conteos de cada decision de limpieza.

    Raises:
        ValueError: Si faltan columnas esperadas.
    """
    missing_cols = sorted(set(EXPECTED_COLUMNS) - set(raw.columns))
    if missing_cols:
        raise ValueError(f"Faltan columnas en el dataset: {missing_cols}")

    df = raw[EXPECTED_COLUMNS].copy()
    report: dict = {"rows_raw": int(len(df))}

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    report["missing_values_total"] = int(df.isna().sum().sum())
    df = df.dropna()

    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    report["rows_after_dedup"] = int(len(df))

    df, label_report = standardize_catalog_labels(df)
    report.update(label_report)

    # La consistencia aritmetica se reporta pero no se corrige: no hay regla de negocio.
    report["cost_inconsistent_rows"] = int(
        (df["Cost"] != df["Unit_Cost"] * df["Order_Quantity"]).sum()
    )
    report["profit_inconsistent_rows"] = int((df["Profit"] != df["Revenue"] - df["Cost"]).sum())
    # Los valores atipicos son ventas reales: se conservan y solo se cuantifican.
    report["negative_profit_rows"] = int((df["Profit"] < 0).sum())
    q1, q3 = df["Customer_Age"].quantile([0.25, 0.75])
    iqr = q3 - q1
    report["age_outliers_iqr_rows"] = int(
        ((df["Customer_Age"] > q3 + 1.5 * iqr) | (df["Customer_Age"] < q1 - 1.5 * iqr)).sum()
    )

    report["rows_before_catalog_filter"] = int(len(df))
    report["stable_catalog_only"] = bool(stable_catalog_only)
    if stable_catalog_only:
        df = keep_stable_catalog_months(df)
        report["months_kept"] = sorted(df["Date"].dt.to_period("M").astype(str).unique().tolist())
    df = df.reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    logger.info("Limpieza: %s", report)
    return df, report


def build_model_input(
    clean: pd.DataFrame, target_category: str, group_columns: list[str]
) -> pd.DataFrame:
    """Construye la tabla de modelado con variables disponibles ANTES de la compra.

    Se excluyen producto, precios, costos, ingresos y cantidad, pues determinan o resultan
    de la categoria comprada (fuga de informacion). ``ganancia`` y ``categoria`` se
    conservan solo para valorar resultados de negocio, no como predictores.

    Args:
        clean: Tabla limpia.
        target_category: Categoria cuya compra se quiere predecir (p. ej. ``Bikes``).
        group_columns: Columnas que definen el grupo de registros "gemelos".

    Returns:
        Tabla con predictores, objetivo ``es_bicicleta``, ``ganancia``, ``categoria`` y ``grupo``.
    """
    out = pd.DataFrame(
        {
            "edad": clean["Customer_Age"].astype(int),
            "genero": clean["Customer_Gender"].astype(str),
            "pais": clean["Country"].astype(str),
            "estado": clean["State"].astype(str),
            "mes": clean["Date"].dt.month.astype(int),
            "es_bicicleta": (clean["Product_Category"] == target_category).astype(int),
            "ganancia": clean["Profit"],
            "categoria": clean["Product_Category"].astype(str),
        }
    )
    out["grupo"] = clean.groupby(group_columns, sort=False).ngroup()
    return out
