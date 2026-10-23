import pandas as pd
import pytest

from bike_propensity.pipelines.data_processing.nodes import (
    build_model_input,
    clean_sales,
    keep_stable_catalog_months,
    standardize_catalog_labels,
)

GROUP = ["Day", "Month", "Customer_Age", "Customer_Gender", "State"]


def test_clean_sales_removes_duplicates(raw_sales):
    clean, report = clean_sales(raw_sales, stable_catalog_only=False)
    assert report["duplicates_removed"] == 1
    assert report["rows_raw"] == len(raw_sales)
    assert report["rows_clean"] == len(clean) == len(raw_sales) - 1
    assert not clean.duplicated().any()


def test_clean_sales_reports_consistent_arithmetic(raw_sales):
    _, report = clean_sales(raw_sales, stable_catalog_only=False)
    assert report["cost_inconsistent_rows"] == 0
    assert report["profit_inconsistent_rows"] == 0


def test_clean_sales_raises_on_missing_column(raw_sales):
    with pytest.raises(ValueError, match="Faltan columnas"):
        clean_sales(raw_sales.drop(columns=["Revenue"]), stable_catalog_only=False)


def test_standardize_labels_uses_majority_label_per_product():
    df = pd.DataFrame(
        {
            "Product": ["Bici"] * 5 + ["Gorra"] * 4,
            "Product_Category": ["Bikes"] * 4 + ["Clothing"] + ["Clothing"] * 4,
            "Sub_Category": ["Touring Bikes"] * 4 + ["Vests"] + ["Caps"] * 4,
        }
    )
    fixed, report = standardize_catalog_labels(df)
    assert set(fixed.loc[fixed.Product == "Bici", "Product_Category"]) == {"Bikes"}
    assert set(fixed.loc[fixed.Product == "Bici", "Sub_Category"]) == {"Touring Bikes"}
    assert report == {"category_labels_fixed_rows": 1, "subcategory_labels_fixed_rows": 1}


def _monthly_frame(categories_by_month: dict[int, list[str]]) -> pd.DataFrame:
    rows = [
        {"Date": pd.Timestamp(2014, month, 15), "Product_Category": cat}
        for month, cats in categories_by_month.items()
        for cat in cats
    ]
    return pd.DataFrame(rows)


def test_keep_stable_catalog_months_drops_bike_only_launch_and_missing_category():
    three = ["Bikes", "Accessories", "Clothing"]
    df = _monthly_frame(
        {
            1: ["Bikes"], 2: ["Bikes"],  # solo bicicletas
            3: three,  # mes de lanzamiento: el mes previo no estaba completo
            4: three, 5: three,  # estables
            6: ["Accessories", "Clothing"],  # falta una categoria
            7: three,  # el mes previo estaba incompleto
            8: three,  # estable
        }
    )  # fmt: skip
    kept = keep_stable_catalog_months(df)
    assert sorted(kept["Date"].dt.month.unique()) == [4, 5, 8]


def test_clean_sales_stable_catalog_flag(raw_sales):
    clean_all, _ = clean_sales(raw_sales, stable_catalog_only=False)
    clean_stable, report = clean_sales(raw_sales, stable_catalog_only=True)
    assert len(clean_stable) <= len(clean_all)
    assert report["rows_clean"] == len(clean_stable)
    assert report["stable_catalog_only"] is True


def test_build_model_input_has_no_leaky_columns(raw_sales):
    clean, _ = clean_sales(raw_sales, stable_catalog_only=False)
    out = build_model_input(clean, "Bikes", GROUP)
    leaky = {
        "Product",
        "Product_Category",
        "Unit_Price",
        "Unit_Cost",
        "Revenue",
        "Cost",
        "Order_Quantity",
    }
    assert leaky.isdisjoint(out.columns)
    assert set(out["is_bike"].unique()) <= {0, 1}
    assert out["is_bike"].sum() == (clean["Product_Category"] == "Bikes").sum()
    assert len(out) == len(clean)


def test_build_model_input_calendar_features(raw_sales):
    clean, _ = clean_sales(raw_sales, stable_catalog_only=False)
    out = build_model_input(clean, "Bikes", GROUP)
    assert out["month"].between(1, 12).all()
    assert out["day_of_week"].between(0, 6).all()
