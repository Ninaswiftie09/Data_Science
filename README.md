# Laboratorio 7. Spark MLlib

Nina Najera, 231088. Universidad del Valle de Guatemala, Data Science.

Análisis de las bases de Personas de la ENEIC: calidad, estadística descriptiva,
correlaciones y segmentación con KMeans. Se utiliza Spark 3.5.1 y Java 17.

## Ejecutar

1. Iniciar Docker Desktop.
2. Ejecutar `docker compose build`.
3. Descargar los datos con `docker compose run --rm pyspark python src/download_data.py`.
4. Ejecutar `docker compose up -d`.
5. Consultar `docker compose logs pyspark` y abrir el enlace local de Jupyter con su token.
6. Abrir `notebooks/Laboratorio_7_Spark_MLlib.ipynb` y ejecutar todas las celdas en orden.

También se puede ejecutar sin abrir Jupyter:

```sh
docker compose run --rm pyspark jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/Laboratorio_7_Spark_MLlib.ipynb
```

La primera ejecución convierte cada Excel a Parquet. Las siguientes reutilizan
esa conversión si la huella SHA-256 del archivo no cambió. La preparación
analítica y los resultados se vuelven a calcular en Spark.

## Archivos

- `notebooks/Laboratorio_7_Spark_MLlib.ipynb`: procedimiento, resultados e interpretación.
- `src/laboratorio.py`: funciones de preparación y análisis utilizadas por el notebook.
- `src/download_data.py`: descarga de las bases de Personas y sus diccionarios.
- `data/sources.json`: enlaces oficiales y huellas de los archivos utilizados.
- `data/raw/` y `data/dictionaries/`: descargas locales, excluidas de Git por tamaño.
- `processed/images/`: gráficos del análisis.
- `processed/tables/`: resultados agregados y auditorías en CSV.
- `processed/parquet/`: datos originales convertidos y conjuntos preparados separados por año; se regeneran.
- `processed/models/`: KMeans elegido y estandarizador; se regeneran.
- `notebooks/Spark_03_SparkML_Nina_Najera.ipynb`: ejercicio individual conservado como evidencia del requisito del curso. Sus datos NBA pertenecen al trabajo anterior y pueden recuperarse del historial de Git.

## Criterios

La población incluye registros de personas de 15 años o más, ocupadas,
asalariadas de categorías 1 a 4 y con salario positivo. Se validan edad,
antigüedad y horas. No se imputan salarios ni se eliminan extremos.
Se conservan los códigos originales ANIO y TRIMESTRE; el período calendario
se obtiene del archivo. El código educativo 0 significa ninguno.

La exploración y KMeans usan solo 2025. El archivo de 2026 se prepara por
separado. Las estadísticas no usan FACTOR y no son estimaciones oficiales
para la población. Una persona puede contribuir observaciones en varios períodos.

## Entrega

Subir el notebook ejecutado y el enlace del repositorio:
https://github.com/Ninaswiftie09/Data_Science/tree/Laboratorio7

El informe LaTeX de la raíz es opcional y está excluido de Git.
Los gráficos se encuentran en `processed/images/` con los mismos nombres
que utiliza el informe. No es necesario subir los Excel ni los Parquet a GitHub.

Fuente: https://www.ine.gob.gt/encuesta-nacional-de-empleo-e-ingresos/

## Resultados principales

Se analizaron 53,025 registros elegibles de 2025. El salario mediano fue
Q3,000 y la media Q3,421.68. No se encontraron claves duplicadas dentro
de cada período. La comparación de K entre 2 y 5 seleccionó dos grupos,
con silhouette de 0.5545. La base preparada de 2026 contiene 13,258 registros.

Las tablas se calculan sobre el conjunto completo. La revisión comprobó que
las exclusiones cierran con los conteos originales, que los grupos suman la
población analítica y que la matriz de correlación es simétrica. El notebook
se ejecutó de principio a fin en Docker.
