"""Fixtures compartidos: una tabla sintetica pequena con el esquema del dataset real."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_sales() -> pd.DataFrame:
    """Genera ventas sinteticas coherentes (Cost y Profit consistentes) con un duplicado."""
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.to_datetime("2012-06-01") + pd.to_timedelta(rng.integers(0, 1500, n), unit="D")
    qty = rng.integers(1, 10, n)
    unit_cost = rng.integers(2, 50, n)
    unit_price = unit_cost + rng.integers(1, 30, n)
    cost = qty * unit_cost
    revenue = (qty * unit_price * 0.9).astype(int)
    category = rng.choice(["Bikes", "Accessories", "Clothing"], n)
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Day": dates.day,
            "Month": dates.month_name(),
            "Year": dates.year,
            "Customer_Age": rng.integers(18, 70, n),
            "Age_Group": "Adults (35-64)",
            "Customer_Gender": rng.choice(["M", "F"], n),
            "Country": rng.choice(["Canada", "France"], n),
            "State": rng.choice(["British Columbia", "Nord", "Hessen"], n),
            "Product_Category": category,
            "Sub_Category": category,
            "Product": "producto_" + pd.Series(category),
            "Order_Quantity": qty,
            "Unit_Cost": unit_cost,
            "Unit_Price": unit_price,
            "Profit": revenue - cost,
            "Cost": cost,
            "Revenue": revenue,
        }
    )
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)  # un duplicado exacto
