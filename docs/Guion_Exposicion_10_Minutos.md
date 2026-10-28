# Guion de exposición: propensión de compra de bicicletas

Duración estimada: 10 minutos, a unas 135–145 palabras por minuto y con pausas breves. Las indicaciones entre corchetes no se leen. Puede presentarse con el póster existente; los tiempos indican el avance acumulado.

**0:00–0:50 | Apertura: la decisión comercial**

[Señalar el problema de negocio en el póster.]

Buenos días. Nuestro proyecto parte de una decisión concreta: si Marketing tiene un presupuesto limitado, ¿a qué perfiles conviene ofrecerles bicicletas?

En la población analizada, las bicicletas representan aproximadamente el 17 % de los pedidos, pero generan el 51 % de la ganancia registrada. Un pedido de bicicleta deja, en promedio, 5,3 veces la ganancia de uno de otra categoría. Por eso, identificar dónde se concentra esa demanda tiene una relevancia económica clara.

El resultado principal es este: al seleccionar el 30 % de los pedidos con mayor puntaje, encontramos el 46,5 % de las compras históricas de bicicletas. Eso plantea una oportunidad para concentrar el esfuerzo comercial que después debemos validar en una campaña real.

**0:50–2:00 | Qué predice el modelo y qué datos utiliza**

[Señalar la pregunta y el bloque de datos.]

La pregunta analítica fue: conociendo únicamente edad, género, país, estado y mes, ¿qué probabilidad existe de que un pedido corresponda a una bicicleta?

El modelo produce un puntaje para ordenar los perfiles y decidir a quién contactar primero, según la capacidad del equipo comercial.

Partimos de 113.036 líneas de pedido, con 18 columnas, entre 2011 y 2016, en seis mercados: Estados Unidos, Canadá, Reino Unido, Alemania, Francia y Australia.

Hay una precisión importante: cada fila es una línea de pedido. Como no existe un identificador de cliente, las cifras describen pedidos y perfiles; todavía no permiten gestionar la propensión de cada cliente individual.

También excluimos producto, precio, costo, cantidad, ingreso y ganancia como predictores. Esos datos revelan qué se compró o son consecuencia de la compra. Incluirlos produciría fuga de información: una evaluación excelente con información que aún no existiría al tomar la decisión comercial. La ganancia se conserva únicamente para analizar el valor económico de los resultados.

**2:00–3:15 | La calidad de datos que sostiene el resultado**

La preparación de los datos tuvo tres decisiones centrales.

Primero, eliminamos mil duplicados exactos y corregimos diez etiquetas de categoría utilizando la clasificación mayoritaria de cada producto. No encontramos valores faltantes.

Segundo, detectamos cambios fuertes de catálogo. Antes de julio de 2013 solo aparecen bicicletas, y hay otros meses sin ventas de esa categoría. Mezclar esos periodos haría que el modelo confundiera disponibilidad de productos con preferencias del comprador.

Por eso conservamos los meses con las tres categorías y cuyo mes anterior también tenía el catálogo completo. Quedaron 96.359 pedidos en 22 meses comparables.

Tercero, encontramos registros que comparten día, mes, edad, género y estado, y reaparecen en distintos años. Los agrupamos para mantenerlos juntos al separar entrenamiento y prueba.

Desde negocio, esto importa porque una recomendación de inversión necesita una evaluación creíble. Si registros prácticamente repetidos aparecen a ambos lados, el desempeño puede parecer mejor de lo que sería ante perfiles nuevos.

**3:15–4:30 | Cómo seleccionamos la solución**

[Señalar la comparación de modelos.]

Reservamos aproximadamente el 20 % de los datos, respetando los grupos, para la evaluación final: 19.108 pedidos. En el entrenamiento comparamos nueve candidatos mediante validación cruzada agrupada.

Usamos PR-AUC como métrica principal porque las bicicletas son una minoría. Un modelo que siempre dijera “no es bicicleta” acertaría cerca del 83 % de las veces y no ayudaría a priorizar ninguna venta.

El modelo seleccionado fue LightGBM, ajustado con Optuna mediante 60 pruebas. LightGBM combina árboles pequeños: cada árbol nuevo corrige parte de los errores anteriores y el conjunto genera una probabilidad. Esto permite captar relaciones como que la edad tenga efectos distintos según el mercado.

La mezcla de LightGBM y CatBoost obtuvo una PR-AUC ligeramente superior: 0,2581 frente a 0,2562. La diferencia quedó dentro de la tolerancia de 0,002 definida para preferir una solución más sencilla.

Elegimos LightGBM por ese equilibrio entre desempeño, rapidez y mantenimiento. La decisión también considera el costo de operar la solución.

**4:30–6:10 | Resultados: de las métricas a la capacidad comercial**

[Señalar los indicadores y la curva de ganancias.]

En los pedidos de prueba, LightGBM obtuvo un ROC-AUC de 0,665, frente a 0,612 de la regresión logística y 0,500 de la referencia al azar. Su PR-AUC fue 0,251, frente a 0,162 de la línea base.

El ROC-AUC indica capacidad de ordenamiento: al tomar una compra de bicicleta y otra de distinta categoría, el modelo asigna el mayor puntaje a la bicicleta aproximadamente el 66,5 % de las veces. El poder predictivo es moderado, pero puede resultar útil cuando existe un límite de contactos.

La traducción comercial se aprecia mejor en la curva de ganancias. El 10 % mejor puntuado concentra el 17,3 % de las bicicletas; el 20 % concentra el 33 %; y el 30 % concentra el 46,5 %.

Con una selección aleatoria del 30 %, esperaríamos encontrar el 30 % de las bicicletas. Con el modelo encontramos 16,5 puntos porcentuales adicionales: una concentración 1,55 veces mayor.

Ese mismo grupo reúne el 46,6 % de la ganancia histórica de bicicletas. Por tanto, la priorización también concentra valor económico.

En esta simulación, para cubrir esa misma proporción de compras al azar necesitaríamos seleccionar el 46,5 % de los pedidos, frente al 30 % con el modelo. La diferencia equivale a cerca del 35 % menos de contactos, redondeada al 36 % en el informe.

Estas cifras miden concentración de compras históricas. El aumento de ventas y el ahorro efectivo de una campaña deben comprobarse experimentalmente.

**6:10–7:50 | Qué perfiles y mercados priorizar**

[Señalar el mapa de país y edad.]

La edad es la variable más influyente, seguida del estado y el mes. El análisis por permutación y la interpretación con SHAP en el notebook ayudan a entender qué información utiliza el modelo.

Los tramos de 25 a 34 y de 35 a 44 años forman el núcleo de la categoría: entre el 18 % y el 19 % de sus pedidos son bicicletas. Su ganancia media, considerando todas las categorías, es de aproximadamente 230 y 257 dólares por pedido, respectivamente.

Pero la edad debe leerse junto con la geografía. Alemania, Reino Unido y Francia presentan sus mayores tasas entre los 25 y los 34 años, alrededor del 23 % al 25 %. En Australia, el máximo está entre los 35 y los 44 años, cerca del 25 %.

Canadá presenta una tasa general del 7,4 %, frente al promedio del 16,6 %. Ahí conviene investigar precios, canales y oferta local, y evaluar campañas de accesorios y ropa. Estas causas todavía son hipótesis.

El género ofrece poca diferenciación: la distancia global es de 0,4 puntos porcentuales y no resulta significativa en el análisis presentado. No encontramos una base sólida para repartir el presupuesto por género.

Noviembre y diciembre, y también mayo y junio, muestran señales favorables. La cobertura temporal disponible exige tratarlas como hipótesis de calendario para validar.

**7:50–8:35 | Cómo se convierte en un proceso repetible**

[Señalar la arquitectura al pie del póster.]

El proyecto incluye una implementación reproducible en Kedro con tres etapas: preparación de datos, entrenamiento y evaluación.

El flujo genera el modelo, métricas, tablas de priorización, resúmenes de segmentos y gráficos. Sus parámetros se configuran por separado, lo que facilita repetir el análisis cuando llegan datos nuevos.

Además, el repositorio configura controles de calidad y pruebas automatizadas en GitHub Actions. Para la empresa, esto facilita revisar los cambios y mantener el proceso.

La comparación amplia de candidatos y SHAP están desarrollados en el notebook; Kedro automatiza el flujo del modelo final y sus evaluaciones principales.

**8:35–10:00 | Recomendación y siguiente decisión de negocio**

[Volver al bloque de recomendaciones.]

Nuestra recomendación es comenzar con un piloto comercial que priorice al 20 % o 30 % de perfiles con mayor puntaje, según el presupuesto disponible.

Para evaluar la estrategia, proponemos comparar grupos de contactos asignados aleatoriamente: uno gestionado con priorización por puntaje y otro con la estrategia habitual, manteniendo presupuesto y condiciones comparables. Dentro del piloto también conviene reservar un grupo sin contacto para medir cuánto cambia la compra por recibir la campaña.

Los indicadores de decisión serían conversión, costo por compra y ganancia incremental después de descuentos y costos de contacto. El dataset actual no contiene costos de campaña ni respuestas a campañas, así que todavía no permite calcular un retorno real.

En paralelo, la inversión de datos más útil sería incorporar identificador de cliente, historial de compras y canal. Los modelos comparados alcanzaron desempeños parecidos, lo que sugiere que añadir información puede aportar más que seguir aumentando la complejidad del algoritmo.

También debemos monitorear el lift y la calibración con datos recientes, porque las preferencias y el catálogo cambian. Edad y género servirían para orientar la oferta, manteniendo el acceso de todos los clientes a los productos.

La decisión que proponemos hoy es financiar ese piloto: comprobar si la concentración observada de demanda se convierte en una mejora de rentabilidad. Tenemos una herramienta que ordena oportunidades mejor que el azar y una forma concreta de medir su valor antes de escalarla.

---

## Notas de respaldo para preguntas — fuera de los diez minutos

- **Fuentes revisadas:** README; los tres PDF completos de `docs` (informe de nueve páginas, póster y explicación del pipeline); HTML del póster; estructura del PowerPoint; las 62 celdas del notebook y sus salidas textuales; código completo de `src`; configuración, dependencias, pruebas y workflow de CI. Se verificó además la limpieza ejecutando el código sobre el CSV original. No se reentrenaron los modelos: las métricas predictivas citadas corresponden a la ejecución documentada.
- **Verificación sobre datos originales:** 113.036 filas; cero faltantes; 1.000 duplicados; diez correcciones de categoría y 24 de subcategoría; 96.359 filas finales; 19.320 grupos. Bicicletas: 15.990 pedidos y 11.363.324 de ganancia registrada, equivalentes al 51,12 % de la ganancia de la población final.
- **Diferencias entre notebook y Kedro:** el notebook compara nueve candidatos con cinco pliegues y dos particiones (diez evaluaciones), e incluye SHAP y 600 remuestreos para los intervalos. Kedro entrena las dos líneas base y LightGBM con CV de cinco pliegues, realiza 60 pruebas de Optuna y usa 500 remuestreos; su interpretación automatizada es por permutación. El flujo no automatiza la comparación y elección entre los nueve candidatos.
- **Intervalos documentados del modelo final:** ROC-AUC 0,654–0,676; PR-AUC 0,240–0,262, al 95 %, remuestreando grupos completos.
- **Calibración documentada:** ECE aproximado de 0,007 en prueba. Es evidencia de esa muestra; requiere seguimiento con datos nuevos.
- **Ahorro de contactos:** con cifras redondeadas, `1 − 30/46,5 = 35,48 %`. El informe lo presenta como 36 % al calcular con sus valores internos. Siempre describirlo como equivalencia retrospectiva frente a selección aleatoria, no como ahorro observado en campañas.
- **Diseño del piloto:** comparar directamente perfiles de alto puntaje con otros de menor puntaje confunde la diferencia previa de propensión con el efecto del contacto. La asignación aleatoria de estrategias y un control sin contacto permiten evaluar preguntas distintas: valor de priorizar y efecto incremental de la campaña. Es una propuesta para el siguiente paso, no un experimento ya ejecutado.
- **Alcance del objetivo:** se estima la categoría entre pedidos observados. No hay registros de personas que no compraron; por eso no equivale a estimar conversión absoluta de una población de prospectos.
- **Recomendaciones por producto:** accesorios y ropa se proponen a partir de la mezcla descriptiva observada; no se entrenaron modelos separados para predecir la respuesta a campañas de esas categorías.
- **Estacionalidad y validación:** los meses se repiten en pares de años con registros correlacionados; queda esencialmente un ciclo independiente y no hay julio en la población final. La evaluación agrupada no sustituye una validación futura con datos recientes.
- **Ganancia:** corresponde al campo `Profit = Revenue − Cost` del dataset. No representa utilidad neta después de marketing y todos los gastos de operación.

## Material de consulta

- [Informe completo](Informe_Propension_Bicicletas.pdf), especialmente páginas 4–8.
- [Póster para acompañar la exposición](Poster_Propension_Bicicletas.pdf).
- [Explicación del pipeline](Explicacion_Pipeline.pdf).
- [Notebook](../notebooks/Proyecto_1_Propension_Bicicletas.ipynb), secciones 5–11.
- [Preparación de datos](../src/bike_propensity/pipelines/data_processing/nodes.py), [entrenamiento](../src/bike_propensity/pipelines/model_training/nodes.py) y [evaluación](../src/bike_propensity/pipelines/model_evaluation/nodes.py).
