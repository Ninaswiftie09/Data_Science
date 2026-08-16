from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    LAGOS,
    MAX_FECHAS,
    NUBOSIDAD_OFICIAL,
    PROJECT_ROOT,
    RESOLUCION_METROS,
    RESULTADOS_DIR
)

from funciones import (
    analizar_correlacion,
    analizar_estacionalidad,
    analizar_temporal,
    boxplot_fechas,
    calcular_persistencia,
    conectar_copernicus,
    crear_comparacion_lagos,
    descargar_bandas_fecha,
    grafica_comparacion_lagos,
    grafica_extension_floracion,
    grafica_temporal_lago,
    leer_y_calcular_indices,
    mapa_comparativo_fechas,
    mapa_diferencia,
    mapa_fecha_pico,
    mostrar_mapa,
    nombre_seguro
)


# aca se conecta con copernicus
conn = conectar_copernicus()


# aca se revisan las fechas oficiales
nubosidad = pd.DataFrame(
    NUBOSIDAD_OFICIAL,
    columns=[
        "lago",
        "fecha",
        "nubosidad_pct",
        "satelite"
    ]
)

nubosidad["fecha"] = pd.to_datetime(
    nubosidad["fecha"]
)

print(
    "\nFechas oficiales del laboratorio:"
)

print(
    nubosidad.to_string(
        index=False
    )
)


# aca se prueba una fecha
lago_prueba = "Atitlán"

fecha_prueba = (
    LAGOS[
        lago_prueba
    ]["fechas"][0]
)

archivo_prueba = descargar_bandas_fecha(
    conn=conn,
    lago=lago_prueba,
    bbox=LAGOS[
        lago_prueba
    ]["bbox"],
    fecha=fecha_prueba,
    resolucion=RESOLUCION_METROS
)

indices_prueba = leer_y_calcular_indices(
    archivo_prueba
)

print(
    "\nPrueba de índices:"
)

print(
    "NDVI promedio:",
    np.nanmean(
        indices_prueba["ndvi"]
    )
)

print(
    "NDWI promedio:",
    np.nanmean(
        indices_prueba["ndwi"]
    )
)

print(
    "Cianobacteria promedio:",
    np.nanmean(
        indices_prueba["cyano"]
    )
)

print(
    "Píxeles válidos de agua:",
    np.sum(
        indices_prueba[
            "mascara_valida"
        ]
    )
)

mostrar_mapa(
    indices_prueba["cyano"],
    (
        f"Índice de cianobacteria - "
        f"{lago_prueba} - "
        f"{fecha_prueba}"
    ),
    (
        f"cyano_prueba_"
        f"{nombre_seguro(lago_prueba)}_"
        f"{fecha_prueba}.png"
    ),
    etiqueta="Bloom Index"
)

mostrar_mapa(
    indices_prueba["ndvi"],
    (
        f"NDVI - "
        f"{lago_prueba} - "
        f"{fecha_prueba}"
    ),
    (
        f"ndvi_prueba_"
        f"{nombre_seguro(lago_prueba)}_"
        f"{fecha_prueba}.png"
    ),
    etiqueta="NDVI"
)

mostrar_mapa(
    indices_prueba["ndwi"],
    (
        f"NDWI - "
        f"{lago_prueba} - "
        f"{fecha_prueba}"
    ),
    (
        f"ndwi_prueba_"
        f"{nombre_seguro(lago_prueba)}_"
        f"{fecha_prueba}.png"
    ),
    etiqueta="NDWI"
)


# aca se procesan todas las fechas
registros = []

for lago, info in LAGOS.items():
    fechas = info["fechas"]

    if MAX_FECHAS is not None:
        fechas = fechas[
            :MAX_FECHAS
        ]

    print(
        f"\nProcesando {lago}..."
    )

    for fecha in tqdm(fechas):
        try:
            ruta = descargar_bandas_fecha(
                conn=conn,
                lago=lago,
                bbox=info["bbox"],
                fecha=fecha,
                resolucion=RESOLUCION_METROS
            )

            indices = leer_y_calcular_indices(
                ruta
            )

            cyano = indices["cyano"]
            ndvi = indices["ndvi"]
            ndwi = indices["ndwi"]

            validos_cyano = cyano[
                np.isfinite(cyano)
            ]

            if validos_cyano.size > 0:
                porcentaje_alto = (
                    np.mean(
                        validos_cyano
                        > 0.05
                    )
                    * 100
                )
            else:
                porcentaje_alto = np.nan

            registros.append({
                "lago":
                    lago,
                "fecha":
                    pd.Timestamp(
                        fecha
                    ),
                "cyano_promedio":
                    np.nanmean(
                        cyano
                    ),
                "cyano_mediana":
                    np.nanmedian(
                        cyano
                    ),
                "ndvi_promedio":
                    np.nanmean(
                        ndvi
                    ),
                "ndwi_promedio":
                    np.nanmean(
                        ndwi
                    ),
                "porcentaje_cyano_alto":
                    porcentaje_alto,
                "pixeles_validos":
                    int(
                        np.sum(
                            np.isfinite(
                                cyano
                            )
                        )
                    ),
                "archivo":
                    str(ruta)
            })

        except Exception as error:
            print(
                f"Error en "
                f"{lago} - "
                f"{fecha}: "
                f"{error}"
            )


# aca se guarda la tabla base
resultados = pd.DataFrame(
    registros
)

if resultados.empty:
    raise RuntimeError(
        "No se pudo procesar "
        "ninguna fecha."
    )

resultados = (
    resultados
    .sort_values(
        [
            "lago",
            "fecha"
        ]
    )
    .reset_index(
        drop=True
    )
)

ruta_csv = (
    RESULTADOS_DIR
    / "resumen_indices.csv"
)

resultados_exportar = (
    resultados.copy()
)

resultados_exportar[
    "archivo"
] = resultados_exportar[
    "archivo"
].apply(
    lambda ruta: str(
        Path(ruta).relative_to(
            PROJECT_ROOT
        )
    )
)

resultados_exportar.to_csv(
    ruta_csv,
    index=False
)

print(
    "\nResultados guardados en:",
    ruta_csv
)


# aca se hace el analisis temporal
grafica_temporal_lago(
    resultados,
    "Atitlán"
)

grafica_temporal_lago(
    resultados,
    "Amatitlán"
)

grafica_comparacion_lagos(
    resultados
)

for texto in analizar_temporal(
    resultados
):
    print(
        "- " + texto
    )

mapa_fecha_pico(
    resultados,
    "Atitlán"
)

mapa_fecha_pico(
    resultados,
    "Amatitlán"
)


# aca se hace el analisis espacial
mapa_comparativo_fechas(
    resultados,
    "Atitlán"
)

mapa_comparativo_fechas(
    resultados,
    "Amatitlán"
)

persistencia = pd.DataFrame([
    calcular_persistencia(
        resultados,
        "Atitlán"
    ),
    calcular_persistencia(
        resultados,
        "Amatitlán"
    )
])

persistencia.to_csv(
    RESULTADOS_DIR
    / "resumen_persistencia.csv",
    index=False
)

print(
    "\nPersistencia:"
)

print(
    persistencia.to_string(
        index=False
    )
)


# aca se hacen las correlaciones
correlaciones = pd.DataFrame([
    analizar_correlacion(
        resultados,
        "Atitlán"
    ),
    analizar_correlacion(
        resultados,
        "Amatitlán"
    )
])

correlaciones.to_csv(
    RESULTADOS_DIR
    / "correlaciones_indices.csv",
    index=False
)

print(
    "\nCorrelaciones:"
)

print(
    correlaciones.to_string(
        index=False
    )
)


# aca se comparan los lagos
comparacion = crear_comparacion_lagos(
    resultados
)

comparacion.to_csv(
    RESULTADOS_DIR
    / "comparacion_lagos.csv",
    index=False
)

print(
    "\nComparación de lagos:"
)

print(
    comparacion.to_string(
        index=False
    )
)


# aca se hace el analisis adicional
grafica_extension_floracion(
    resultados
)

boxplot_fechas(
    resultados,
    "Atitlán"
)

boxplot_fechas(
    resultados,
    "Amatitlán"
)

diferencias = pd.DataFrame([
    mapa_diferencia(
        resultados,
        "Atitlán"
    ),
    mapa_diferencia(
        resultados,
        "Amatitlán"
    )
])

diferencias.to_csv(
    RESULTADOS_DIR
    / "diferencias_espaciales.csv",
    index=False
)

resumen_estacional = (
    analizar_estacionalidad(
        resultados
    )
)

resumen_estacional.to_csv(
    RESULTADOS_DIR
    / "resumen_estacional.csv",
    index=False
)

print(
    "\nDiferencias espaciales:"
)

print(
    diferencias.to_string(
        index=False
    )
)

print(
    "\nResumen estacional:"
)

print(
    resumen_estacional.to_string(
        index=False
    )
)

print(
    "\nLaboratorio completo."
)