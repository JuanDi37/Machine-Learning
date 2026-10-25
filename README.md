# Propensión de compra de bicicletas

Proyecto 1 de *Machine Learning for Business* (2026). Dataset 2: [*Sales Data*](https://www.kaggle.com/datasets/jehanzaibbhatti/sales-data) (Kaggle).

**Pregunta de negocio.** Conociendo solo la edad, el género, el país, el estado y el mes, ¿qué probabilidad hay de que un pedido sea una bicicleta y, por tanto, a qué perfiles conviene dirigir las ventas de cada línea de producto?

Producto, precio, costo, cantidad, ingreso y ganancia se excluyen como predictores para evitar fuga de información; la ganancia solo se usa para valorar el resultado.

## Resultados (conjunto de prueba, 19,108 pedidos)

| Modelo | ROC-AUC | PR-AUC |
|---|---|---|
| Línea base (prevalencia) | 0.500 | 0.162 |
| Regresión logística | 0.612 | 0.215 |
| **LightGBM + Optuna (final)** | **0.665** (IC 95 %: 0.654–0.676) | **0.251** (IC 95 %: 0.240–0.262) |

Contactando al 30 % de los pedidos mejor puntuados se captura ≈ 46 % de las compras de bicicleta (30 % al azar). El modelo se eligió entre nueve candidatos (regresión logística, HistGradientBoosting, XGBoost, LightGBM, CatBoost y una mezcla) con validación cruzada agrupada; el detalle, las limitaciones y las recomendaciones están en el notebook y en el informe.

Las cifras exactas se regeneran con `kedro run` (se guardan en `data/08_reporting/`); pueden variar levemente con otras versiones de librerías.

## Contenido

```
data/01_raw/sales_data.csv                         # dataset original (único dato versionado)
notebooks/Proyecto_1_Propension_Bicicletas.ipynb   # análisis completo (CRISP-DM), ejecutado
docs/Informe_Propension_Bicicletas.pdf             # informe para audiencia no técnica
docs/Poster_Propension_Bicicletas.pptx             # diapositiva de presentación (infografía)
conf/base/catalog.yml, parameters.yml              # catálogo de datos y parámetros
src/bike_propensity/pipelines/                     # data_processing, model_training, model_evaluation
tests/                                             # pruebas con datos sintéticos
.github/workflows/ci.yml                           # GitHub Actions
```

## Instalación

Requiere Python 3.10 o superior (probado con 3.12 y 3.13).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

## Ejecución

```bash
python -m kedro run                                   # pipeline completo (≈ 6 min, por las 60 pruebas de Optuna)
python -m kedro run --pipeline data_processing        # solo limpieza y tabla de modelado
python -m kedro run --params "tuning.n_trials=3"      # búsqueda reducida (la que usa el CI)
```

No hay pasos manuales: el CSV crudo ya está en `data/01_raw/`. Las salidas quedan en `data/` (no se versionan).

### Calidad y pruebas

```bash
ruff check . && ruff format --check .
pytest --cov=src
```

El notebook se abre con `jupyter lab notebooks/` y se ejecuta completo (≈ 12 min, por la búsqueda de hiperparámetros y CatBoost).

## Arquitectura

| Pipeline | Nodos | Salidas clave |
|---|---|---|
| `data_processing` | `clean_sales`, `build_model_input` | `clean_sales.parquet`, `cleaning_report.json`, `model_input.parquet` |
| `model_training` | `split_data`, `fit_baselines`, `tune_lightgbm`, `fit_final_model` | `final_lightgbm.pkl`, `best_params.json`, `tuning_trials.csv` |
| `model_evaluation` | métricas con IC, lift, ganancias, permutación, segmentos, gráficos | `metrics.json`, `lift_table.csv`, `business_impact.csv`, `segment_summary.csv`, `plots/` |

## Decisiones importantes

- **Sin fuga de información.** Solo edad, género, país, estado y mes.
- **Partición por grupos.** Los pedidos del mismo día, edad, género y estado forman grupos (19,320) que además reaparecen en un segundo año; `GroupShuffleSplit` y `StratifiedGroupKFold` los mantienen del mismo lado para no inflar las métricas.
- **Solo meses de catálogo estable** (22 meses, 96,359 pedidos): antes de julio de 2013 solo se vendían bicicletas, julio de 2013 es el lanzamiento de las demás líneas y en julio de 2014 y 2016 no hay bicicletas. Son otro régimen, no comportamiento del cliente (`stable_catalog_only` en `parameters.yml`).
- **Limpieza:** 1,000 duplicados exactos eliminados y 10 etiquetas de categoría corregidas con la mayoritaria de cada producto.
- **Selección del modelo:** mayor PR-AUC en validación cruzada; si un modelo más simple o rápido queda a menos de 0.002, se prefiere el más simple. Resultó LightGBM ajustado con Optuna (60 pruebas, TPE).
- **Uso responsable.** Edad y género orientan la oferta comercial; no deben usarse para excluir clientes ni fijar precios.

## Referencias

Chapman et al. (2000), CRISP-DM; Friedman (2001); Ke et al. (2017), LightGBM; Akiba et al. (2019), Optuna; Lundberg & Lee (2017), SHAP; Kaufman et al. (2012), fuga de información. La lista completa está en el notebook y en el informe.
