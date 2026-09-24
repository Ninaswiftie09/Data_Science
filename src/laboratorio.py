"""Preparación y análisis ENEIC con DataFrames y MLlib de Spark 3.5."""
from pathlib import Path
from functools import reduce
import hashlib
import json
import math
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
COLS = ['P05D01', 'P02A03', 'P05C07A', 'P05C07B', 'P05H01A',
        'P03A03A', 'P05C16', 'DOMINIO', 'OCUPADOS', 'NUM_HOGAR',
        'NUM_PERSONA', 'FACTOR', 'ANIO', 'TRIMESTRE']
NAMES = dict(zip(COLS[:9], ['salario_mensual', 'edad', 'antiguedad_anios',
             'antiguedad_meses', 'horas_semanales', 'nivel_educativo',
             'categoria_ocupacional', 'dominio', 'ocupado']))
NUM = ['salario_mensual', 'edad', 'antiguedad', 'horas_semanales']
LABELS = {'nivel_educativo': {'0': 'Ninguno', '1': 'Preprimaria', '2': 'Primaria',
          '3': 'Básico', '4': 'Diversificado', '5': 'Superior', '6': 'Maestría', '7': 'Doctorado'},
          'categoria_ocupacional': {'1': 'Gobierno', '2': 'Empresa privada',
          '3': 'Jornalero o peón', '4': 'Servicio doméstico'},
          'dominio': {'1': 'Urbano metropolitano', '2': 'Resto urbano', '3': 'Rural nacional'}}
FILES = {f'2025T{i}': f'2025T{i}.xlsx' for i in range(1, 5)}
FILES['2026T1'] = 'Base-de-datos-Personas-ENEIC-I-2026.xlsx'
EXPECTED = {'2025T1': (51588, 270), '2025T2': (51167, 270),
            '2025T3': (51583, 270), '2025T4': (49338, 302), '2026T1': (49843, 270)}

def session():
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 25)
    pd.set_option('display.float_format', lambda x: f'{x:,.3f}')
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12})
    spark = (SparkSession.builder.master('local[2]').appName('ENEIC Laboratorio 7')
             .config('spark.driver.memory', '2g').config('spark.sql.shuffle.partitions', '4')
             .config('spark.ui.showConsoleProgress', 'false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    assert spark.version.startswith('3.5.'), spark.version
    for folder in ['images', 'tables', 'parquet', 'models']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    return spark

def save_table(frame, name):
    frame.to_csv(OUT / 'tables' / f'{name}.csv', index=False)
    return frame

def finite(name):
    c = F.col(name)
    return c.isNotNull() & ~F.isnan(c) & (F.abs(c) < float('inf'))

def dictionary(period):
    book = openpyxl.load_workbook(ROOT / 'data/dictionaries' / f'{period}.xlsx',
                                 read_only=True, data_only=True)
    codes = {k: {} for k in ['P03A03A', 'P05C16', 'DOMINIO']}
    current = None
    for row in book.active.values:
        # Las etiquetas de códigos no tienen nivel de medición en la cuarta columna.
        if len(row) < 4 or row[3] is not None:
            current = None
            continue
        if row[0] is not None:
            current = str(row[0]).strip()
        if current in codes and row[1] is not None:
            try:
                codes[current][str(int(float(row[1])))] = str(row[2])
            except (ValueError, TypeError):
                pass
    book.close()
    assert all(codes.values()), f'Diccionario incompleto: {period}'
    return codes

def load_excel(spark, period, filename):
    path = ROOT / 'data/raw' / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = OUT / 'parquet' / f'original_{period}'
    meta = OUT / 'parquet' / f'original_{period}.json'
    if cache.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info['sha256'] == digest:
            return spark.read.parquet(str(cache)), info
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = book.active.values
    headers = [str(v).strip() for v in next(rows)]
    positions = [headers.index(c) for c in COLS]
    selected = []
    for row in rows:
        if all(v is None for v in row):
            continue
        values = [None if row[i] is None or str(row[i]).strip() == '' else str(row[i]).strip()
                  for i in positions]
        fingerprint = hashlib.sha256(json.dumps(row, default=str).encode()).hexdigest()
        selected.append(tuple(values + [fingerprint]))
    book.close()
    info = {'periodo_archivo': period, 'filas_originales': len(selected),
            'columnas_originales': len(headers), 'sha256': digest}
    assert (len(selected), len(headers)) == EXPECTED[period], info
    schema = T.StructType([T.StructField(c, T.StringType(), True) for c in COLS + ['huella_original']])
    df = spark.createDataFrame(selected, schema)
    df.write.mode('overwrite').parquet(str(cache))
    meta.write_text(json.dumps(info))
    return spark.read.parquet(str(cache)), info

def prepare(spark):
    summaries, missing, exclusions, duplicate_summary, invalid_codes, trimestres = [], [], [], [], [], []
    prepared = {}
    for period, filename in FILES.items():
        print('Preparando', period, flush=True)
        raw, info = load_excel(spark, period, filename)
        n = info['filas_originales']
        counts = raw.agg(*[F.sum(F.col(c).isNull().cast('int')).alias(c) for c in COLS]).first()
        for c in COLS:
            missing.append((period, c, int(counts[c]), counts[c] * 100 / n))
        # Los códigos enteros se normalizan después de comprobar que no son fraccionarios.
        df = raw
        for c in COLS:
            if c in ['NUM_HOGAR', 'NUM_PERSONA']:
                df = df.withColumn(c, F.regexp_replace(F.col(c), r'\.0+$', ''))
            else:
                df = df.withColumn(c, F.col(c).cast('double'))
        df = (df.withColumn('periodo_archivo', F.lit(period))
              .withColumn('anio_archivo', F.lit(int(period[:4])))
              .withColumn('trimestre_calendario', F.lit(int(period[-1])))
              .withColumn('archivo_origen', F.lit(filename)))
        for row in df.groupBy('TRIMESTRE').count().collect():
            trimestres.append((period, row['TRIMESTRE'], row['count']))
        keys = ['periodo_archivo', 'NUM_HOGAR', 'NUM_PERSONA']
        repeated = df.groupBy(keys).agg(F.count('*').alias('n'),
                      F.countDistinct('huella_original').alias('versiones')).filter('n > 1')
        duplicate_rows = repeated.collect()
        null_keys = df.filter(F.col('NUM_HOGAR').isNull() | F.col('NUM_PERSONA').isNull()).count()
        duplicate_summary.append((period, len(duplicate_rows),
            sum(r['versiones'] == 1 for r in duplicate_rows),
            sum(r['versiones'] > 1 for r in duplicate_rows), null_keys))
        if duplicate_rows:
            save_table(repeated.toPandas(), f'claves_duplicadas_{period}')
            raise ValueError(f'Claves duplicadas en {period}; revisar tabla antes de continuar.')
        assert null_keys == 0, f'Claves ausentes: {period}'
        codes = dictionary(period)
        for source, allowed in codes.items():
            canonical = F.col(source).cast('long').cast('string')
            valid = finite(source) & (F.col(source) == F.floor(source)) & canonical.isin(list(allowed))
            invalid_codes.append((period, source, df.filter(~F.coalesce(valid, F.lit(False))).count()))
            df = df.withColumn(source, F.when(valid, canonical).otherwise('DESCONOCIDO'))
        for old, new in NAMES.items():
            df = df.withColumnRenamed(old, new)
        df = df.withColumn('antiguedad', F.col('antiguedad_anios') + F.col('antiguedad_meses') / 12)
        rules = [
            ('Edad finita y al menos 15', finite('edad') & (F.col('edad') >= 15)),
            ('Ocupado', F.col('ocupado') == 1),
            ('Asalariado: categorías 1 a 4', F.col('categoria_ocupacional').isin('1', '2', '3', '4')),
            ('Salario finito y positivo', finite('salario_mensual') & (F.col('salario_mensual') > 0)),
            ('Años de antigüedad finitos y no negativos', finite('antiguedad_anios') & (F.col('antiguedad_anios') >= 0)),
            ('Meses enteros entre 0 y 11', finite('antiguedad_meses') & F.col('antiguedad_meses').between(0, 11) & (F.col('antiguedad_meses') == F.floor('antiguedad_meses'))),
            ('Antigüedad no mayor a edad', finite('antiguedad') & (F.col('antiguedad') <= F.col('edad'))),
            ('Horas finitas en (0,168]', finite('horas_semanales') & (F.col('horas_semanales') > 0) & (F.col('horas_semanales') <= 168))]
        # Una sola agregación cuenta los filtros acumulativos en su orden fijo.
        condition = F.lit(True)
        expressions = []
        for i, (label, rule) in enumerate(rules):
            condition = condition & F.coalesce(rule, F.lit(False))
            expressions.append(F.sum(condition.cast('long')).alias(f's{i}'))
        remaining = df.agg(*expressions).first()
        before = n
        for i, (label, _) in enumerate(rules):
            after = int(remaining[f's{i}'])
            exclusions.append((period, i + 1, label, before, before - after, after))
            before = after
        df = df.filter(condition).drop('huella_original')
        prepared[period] = df
        summaries.append({**info, 'filas_elegibles': before})
    train = reduce(lambda a, b: a.unionByName(b), [prepared[p] for p in FILES if p.startswith('2025')])
    test = prepared['2026T1']
    for name, df in [('personas_2025', train), ('personas_2026', test)]:
        df.write.mode('overwrite').parquet(str(OUT / 'parquet' / name))
    tables = {
        'archivos': pd.DataFrame(summaries).drop(columns='sha256'),
        'faltantes': pd.DataFrame(missing, columns=['periodo', 'variable', 'faltantes', 'porcentaje']),
        'exclusiones': pd.DataFrame(exclusions, columns=['periodo', 'paso', 'criterio', 'antes', 'excluidos', 'despues']),
        'unicidad': pd.DataFrame(duplicate_summary, columns=['periodo', 'claves_duplicadas', 'exactas', 'en_conflicto', 'claves_nulas']),
        'codigos_desconocidos': pd.DataFrame(invalid_codes, columns=['periodo', 'variable', 'desconocidos_antes_filtros']),
        'trimestres_originales': pd.DataFrame(trimestres, columns=['periodo', 'TRIMESTRE', 'n'])}
    for name, table in tables.items():
        save_table(table, name)
    return spark.read.parquet(str(OUT / 'parquet/personas_2025')).cache(), tables

def figure(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / 'images' / filename, dpi=160, bbox_inches='tight')
    plt.close(fig)

def descriptive(df):
    records = []
    for c in NUM:
        row = df.agg(F.count(c).alias('n'), F.mean(c).alias('media'),
                    F.stddev_samp(c).alias('desviacion'), F.min(c).alias('minimo'),
                    F.max(c).alias('maximo'),
                    F.expr(f'percentile({c}, array(0.25, 0.5, 0.75, 0.95))').alias('p')).first()
        records.append({'variable': c, **{k: row[k] for k in ['n', 'media', 'desviacion', 'minimo', 'maximo']},
                        'p25': row.p[0], 'mediana': row.p[1], 'p75': row.p[2], 'p95': row.p[3]})
    result = {'descriptivos': save_table(pd.DataFrame(records), 'descriptivos')}
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    for ax, (c, labels) in zip(axes, LABELS.items()):
        table = df.groupBy(c).agg(F.count('*').alias('n'),
                   F.expr('percentile(salario_mensual, 0.5)').alias('mediana')).orderBy(c).toPandas()
        table['etiqueta'] = table[c].map(labels).fillna('Desconocido')
        result[c] = save_table(table, c)
        ax.barh(table.etiqueta, table.n, color='#367896')
        ax.set_xlabel('Registros'); ax.set_title(c.replace('_', ' ').capitalize())
    figure(fig, 'distribucion_categorias.png')
    # El histograma usa conteos del conjunto completo en intervalos logarítmicos.
    limits = df.agg(F.min('salario_mensual'), F.max('salario_mensual')).first()
    edges = np.linspace(math.log10(limits[0]), math.log10(limits[1]) + 1e-8, 31)
    width = edges[1] - edges[0]
    hist = (df.withColumn('bin', F.floor((F.log10('salario_mensual') - edges[0]) / width))
            .groupBy('bin').count().orderBy('bin').toPandas())
    save_table(hist, 'histograma_salarios')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(10 ** (edges[0] + hist.bin * width), hist['count'],
           width=10 ** (edges[0] + (hist.bin + 1) * width) - 10 ** (edges[0] + hist.bin * width), align='edge')
    ax.set_xscale('log'); ax.set_xlabel('Salario mensual (Q), escala logarítmica')
    ax.set_ylabel('Registros'); ax.set_title('Distribución del salario mensual, 2025')
    figure(fig, 'distribucion_salarios.png')
    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    for ax, c in zip(axes, ['nivel_educativo', 'categoria_ocupacional']):
        t = result[c]
        ax.barh(t.etiqueta, t.mediana, color='#367896')
        ax.set_xlabel('Salario mediano (Q)'); ax.set_title(c.replace('_', ' ').capitalize())
    figure(fig, 'salario_por_grupo.png')
    quarter = df.groupBy('periodo_archivo').agg(F.count('*').alias('n'),
              F.expr('percentile(salario_mensual, 0.5)').alias('mediana')).orderBy('periodo_archivo').toPandas()
    result['trimestres'] = save_table(quarter, 'resumen_trimestres')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(quarter.periodo_archivo, quarter.n); axes[0].set_ylabel('Registros elegibles')
    axes[1].plot(quarter.periodo_archivo, quarter.mediana, marker='o'); axes[1].set_ylabel('Salario mediano (Q)')
    for ax in axes: ax.set_xlabel('Período del archivo')
    figure(fig, 'comparacion_trimestres.png')
    return result

def correlations(df):
    vectors = VectorAssembler(inputCols=NUM, outputCol='vector_correlacion').transform(df)
    matrix = Correlation.corr(vectors, 'vector_correlacion', 'pearson').first()[0].toArray()
    table = pd.DataFrame(matrix, index=NUM, columns=NUM)
    save_table(table.rename_axis('variable').reset_index(), 'correlaciones')
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(4), ['Salario', 'Edad', 'Antigüedad', 'Horas'])
    ax.set_yticks(range(4), ['Salario', 'Edad', 'Antigüedad', 'Horas'])
    for i in range(4):
        for j in range(4): ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center')
    fig.colorbar(im, ax=ax, label='Correlación de Pearson')
    figure(fig, 'correlaciones.png')
    return table

def clusters(df):
    cols = ['edad', 'antiguedad', 'horas_semanales']
    assembler = VectorAssembler(inputCols=cols, outputCol='vector_perfil')
    vectors = assembler.transform(df)
    scaler = StandardScaler(inputCol='vector_perfil', outputCol='features', withMean=True, withStd=True).fit(vectors)
    scaled = scaler.transform(vectors).cache()
    scaled.count()
    evaluator = ClusteringEvaluator(featuresCol='features', metricName='silhouette', distanceMeasure='squaredEuclidean')
    scores, models = [], {}
    for k in [2, 3, 4, 5]:
        model = KMeans(k=k, seed=231088, featuresCol='features', maxIter=100, tol=1e-4).fit(scaled)
        pred = model.transform(scaled)
        score = evaluator.evaluate(pred)
        scores.append({'k': k, 'silhouette': score, 'costo': model.summary.trainingCost})
        models[k] = model
    scores = save_table(pd.DataFrame(scores), 'seleccion_k')
    best = int(scores.sort_values(['silhouette', 'k'], ascending=[False, True]).iloc[0].k)
    model = models[best]
    profiles = model.transform(scaled).groupBy('prediction').agg(F.count('*').alias('n'),
        *[F.mean(c).alias(c) for c in cols],
        F.expr('percentile(salario_mensual, 0.5)').alias('salario_mediano')).orderBy('prediction').toPandas()
    save_table(profiles, 'perfiles_clusters')
    model.write().overwrite().save(str(OUT / 'models/kmeans'))
    scaler.write().overwrite().save(str(OUT / 'models/scaler'))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scores.k, scores.silhouette, marker='o'); ax.set_xticks([2, 3, 4, 5])
    ax.set_xlabel('Número de grupos (K)'); ax.set_ylabel('Silhouette (distancia euclidiana al cuadrado)')
    figure(fig, 'seleccion_k.png')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, c in zip(axes, cols):
        ax.bar(profiles.prediction.astype(str), profiles[c]); ax.set_title(c.replace('_', ' '))
        ax.set_xlabel('Grupo'); ax.set_ylabel('Promedio en años' if c != 'horas_semanales' else 'Promedio en horas/semana')
    figure(fig, 'perfiles_clusters.png')
    scaled.unpersist()
    return scores, profiles, best
