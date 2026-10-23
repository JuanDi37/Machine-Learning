# bike_propensity — ¿A quién dirigir la campaña de bicicletas?

Proyecto 1 de *Machine Learning for Business* (2026). Proyecto [Kedro](https://docs.kedro.org) que responde:

> Dado el perfil de un cliente (edad, género, ubicación) y el mes, **¿qué probabilidad hay de que su compra sea una bicicleta** (producto de alto margen), y qué perfiles conviene priorizar en una campaña?

Dataset: *Sales Data* (Kaggle, <https://www.kaggle.com/datasets/jehanzaibbhatti/sales-data>), incluido en `data/01_raw/sales_data.csv`.
Modelo final: `HistGradientBoostingClassifier` (scikit-learn) con búsqueda aleatoria de hiperparámetros y validación cruzada agrupada.

## Resultados principales (conjunto de prueba, 19,108 pedidos)

| Modelo | ROC-AUC | PR-AUC |
|---|---|---|
| Línea base (prevalencia) | 0.500 | 0.162 |
| Regresión logística | 0.599 | 0.208 |
| **Gradient boosting (final)** | **0.664** (IC95 % 0.654–0.675) | **0.248** (IC95 % 0.237–0.260) |

Contactando al 30 % de los pedidos mejor puntuados se alcanza ≈ 46 % de las compras de bicicleta (30 % al azar). El detalle, las limitaciones y las referencias están en `notebooks/analisis_completo.ipynb` y en `docs/reporte.pdf`.
Estos números se regeneran con `kedro run`; si cambia la semilla, el entorno o las versiones de librerías pueden variar levemente.

## Instalación

Requiere Python 3.10+ (probado con 3.13 y 3.12).

```bash
cd Proyecto1/bike-propensity
python -m venv .venv
# Windows: .venv\Scripts\activate      |  Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

## Ejecución

```bash
python -m kedro run                    # pipeline completo (~1 min)
python -m kedro run --pipeline data_processing   # solo limpieza y features
python -m kedro run --params "tuning.n_iter=3"   # búsqueda reducida (lo que usa el CI)
```

No hay pasos manuales: el CSV crudo ya está en `data/01_raw/`. Todas las salidas quedan en `data/` (ver la tabla de abajo).

### Calidad y pruebas

```bash
ruff check . && ruff format --check .
pytest --cov=src
```

## Estructura

```
conf/base/catalog.yml        # todos los datasets y su ubicación
conf/base/parameters.yml     # parámetros configurables (año mínimo, features, CV, búsqueda, etc.)
src/bike_propensity/pipelines/
  data_processing/           # limpieza (duplicados, esquema, consistencia) y tabla de modelado
  model_training/            # partición agrupada, baselines, ajuste de hiperparámetros
  model_evaluation/          # métricas + IC bootstrap, lift, ganancias, importancia, gráficos
tests/                       # pruebas con datos sintéticos
notebooks/analisis_completo.ipynb   # EDA, justificación, evaluación, SHAP, limitaciones
notebooks/experimento_lightgbm.ipynb # experimento: LightGBM/CatBoost/Optuna/ensamble (sin mejora material)
docs/                        # reporte, figuras y póster
```

| Pipeline | Nodos | Salidas clave |
|---|---|---|
| `data_processing` | `clean_sales_node`, `build_model_input_node` | `02_intermediate/clean_sales.parquet`, `cleaning_report.json`, `03_primary/model_input.parquet` |
| `model_training` | `split_data_node`, `fit_baselines_node`, `tune_gradient_boosting_node` | `06_models/final_hgb.pkl`, baselines, `tuning_cv_results.csv` |
| `model_evaluation` | métricas, lift, ganancias, importancia, gráficos | `08_reporting/metrics.json`, `lift_table.csv`, `business_impact.csv`, `plots/` |

## Decisiones importantes

- **Sin fuga de información.** Solo se usan variables conocidas antes de la compra (edad, género, país, estado, mes). Producto, precio, costo, cantidad, ingreso y ganancia están excluidos.
- **Partición por grupos.** El dataset contiene registros "gemelos" repetidos entre años (misma fecha día/mes, edad, género, estado y producto); `GroupShuffleSplit`/`GroupKFold` los mantienen del mismo lado para no inflar métricas.
- **Solo meses de catálogo estable** (22 meses, 96,359 pedidos): antes de jul-2013 solo se vendían bicicletas, jul-2013 es el lanzamiento y en jul-2014/2016 no hay bicicletas (otro régimen, no comportamiento del cliente). La regla es automática (`stable_catalog_only` en `parameters.yml`).
- **Etiquetas de producto corregidas:** 24 filas traían subcategoría distinta a la de su producto (10 también categoría); se usa la etiqueta mayoritaria de cada producto.
- **Se eliminan 1,000 duplicados exactos.**
- El margen realizado (`Profit/Revenue`) y la cantidad por pedido se evaluaron como alternativas y se descartaron (ver notebook, sección 2.6).

## Reproducibilidad

Semillas fijas en `parameters.yml`; entorno declarado en `requirements.txt`; CI en `.github/workflows/ci.yml` (raíz del repositorio) que en cada *push* y *pull request* instala dependencias, corre Ruff, pytest y `kedro run`.

> Si este proyecto se sube como repositorio propio con el proyecto en la raíz, mueva `.github/` a la raíz del proyecto y elimine `working-directory` y las rutas `Proyecto1/bike-propensity/` del workflow.

## Referencias

Ver la sección *Referencias* del notebook y del reporte (CRISP-DM: Chapman et al., 2000; gradient boosting: Friedman, 2001; Ke et al., 2017; scikit-learn: Pedregosa et al., 2011; entre otras). Herramientas: [Kedro](https://docs.kedro.org), [Ruff](https://docs.astral.sh/ruff/), [scikit-learn](https://scikit-learn.org), [SHAP](https://shap.readthedocs.io).
