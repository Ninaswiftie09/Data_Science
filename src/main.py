import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    LAGOS,
    MAX_FECHAS,
    NUBOSIDAD_OFICIAL,
    RESOLUCION_METROS,
    RESULTADOS_DIR
)

from funciones import (
    analizar_temporal,
    conectar_copernicus,
    descargar_bandas_fecha,
    grafica_comparacion_lagos,
    grafica_temporal_lago,
    leer_y_calcular_indices,
    mapa_fecha_pico,
    mostrar_mapa,
    nombre_seguro
)


# ============================================================
# 1. CONEXIÓN CON COPERNICUS / OPENEO
# ============================================================

conn = conectar_copernicus()


# ============================================================
# 2. REVISIÓN DE FECHAS OFICIALES
# ============================================================

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


# ============================================================
# 3. PRUEBA CON UNA SOLA FECHA
# ============================================================

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


# ============================================================
# MAPA DE CIANOBACTERIA DE PRUEBA
# ============================================================

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


# ============================================================
# MAPA NDVI DE PRUEBA
# ============================================================

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


# ============================================================
# MAPA NDWI DE PRUEBA
# ============================================================

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


# ============================================================
# 4. PROCESAMIENTO DE TODAS LAS FECHAS
# ============================================================

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


            registro = {
                "lago": lago,

                "fecha": pd.Timestamp(
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
            }


            registros.append(
                registro
            )


        except Exception as error:

            print(
                f"Error en "
                f"{lago} - "
                f"{fecha}: "
                f"{error}"
            )


# ============================================================
# 5. TABLA DE RESULTADOS
# ============================================================

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


resultados.to_csv(
    ruta_csv,
    index=False
)


print(
    "\nResumen de resultados:"
)


print(
    resultados.to_string(
        index=False
    )
)


print(
    "\nCSV guardado en:",
    ruta_csv
)


# ============================================================
# 6. ANÁLISIS TEMPORAL
# ============================================================

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


# ============================================================
# 7. RESUMEN AUTOMÁTICO
# ============================================================

print(
    "\nHallazgos temporales:"
)


for texto in analizar_temporal(
    resultados
):

    print(
        "- " + texto
    )


# ============================================================
# 8. MAPAS DE LAS FECHAS PICO
# ============================================================

mapa_fecha_pico(
    resultados,
    "Atitlán"
)


mapa_fecha_pico(
    resultados,
    "Amatitlán"
)


# ============================================================
# 9. TABLA FINAL DEL AVANCE
# ============================================================

tabla_avance = resultados[
    [
        "lago",
        "fecha",
        "cyano_promedio",
        "cyano_mediana",
        "ndvi_promedio",
        "ndwi_promedio",
        "porcentaje_cyano_alto",
        "pixeles_validos"
    ]
].copy()


tabla_avance["fecha"] = (
    tabla_avance["fecha"]
    .dt.strftime(
        "%Y-%m-%d"
    )
)


ruta_tabla = (
    RESULTADOS_DIR
    / "tabla_avance.csv"
)


tabla_avance.to_csv(
    ruta_tabla,
    index=False
)


print(
    "\nTabla final del avance:"
)


print(
    tabla_avance.to_string(
        index=False
    )
)


print(
    "\nTabla guardada en:",
    ruta_tabla
)


print(
    "\nListo. El avance de los "
    "ejercicios 1 al 4 "
    "quedó procesado."
)