import pandas as pd

from config import (
    DATASET_ML_PATH,
    LAGOS,
    MUESTRA_ML_POR_FECHA,
    NUBOSIDAD_OFICIAL,
    RANDOM_STATE,
    UMBRAL_CYANO_ALTO
)

from funciones import (
    construir_dataset_ml,
    graficar_distribucion_objetivo,
    graficar_eda_ml,
    resumir_dataset_ml
)


# aca se construye o reutiliza el conjunto de datos
datos = construir_dataset_ml(
    lagos=LAGOS,
    nubosidad_oficial=NUBOSIDAD_OFICIAL,
    salida_csv=DATASET_ML_PATH,
    umbral_cyano=UMBRAL_CYANO_ALTO,
    max_observaciones_fecha=MUESTRA_ML_POR_FECHA,
    random_state=RANDOM_STATE,
    reutilizar=True
)


# aca se muestran los resultados de preparacion
resumen = resumir_dataset_ml(
    datos
)

print(
    "\nTotal de observaciones:",
    f"{resumen['total']:,}"
)

print(
    "\nObservaciones por lago:"
)
print(
    resumen["por_lago"].to_string(
        index=False
    )
)

print(
    "\nObservaciones por fecha:"
)
print(
    resumen["por_fecha"].to_string(
        index=False
    )
)

print(
    "\nTipos y valores faltantes:"
)
print(
    resumen["faltantes"].to_string(
        index=False
    )
)


# aca se revisa la variable respuesta
respuesta_global = (
    datos["cyano_alta"]
    .value_counts()
    .sort_index()
    .rename_axis("cyano_alta")
    .reset_index(name="observaciones")
)

respuesta_global["porcentaje"] = (
    respuesta_global["observaciones"]
    / len(datos)
    * 100
)

print(
    "\nDistribución global de la respuesta:"
)
print(
    respuesta_global.to_string(
        index=False
    )
)

respuesta_lago = (
    datos
    .groupby([
        "lago",
        "cyano_alta"
    ])
    .size()
    .rename("observaciones")
    .reset_index()
)

respuesta_lago["porcentaje"] = (
    respuesta_lago["observaciones"]
    / respuesta_lago.groupby(
        "lago"
    )["observaciones"].transform("sum")
    * 100
)

print(
    "\nDistribución por lago:"
)
print(
    respuesta_lago.to_string(
        index=False
    )
)

respuesta_fecha = (
    datos
    .groupby([
        "lago",
        "fecha",
        "cyano_alta"
    ])
    .size()
    .rename("observaciones")
    .reset_index()
)

respuesta_fecha["porcentaje"] = (
    respuesta_fecha["observaciones"]
    / respuesta_fecha.groupby([
        "lago",
        "fecha"
    ])["observaciones"].transform("sum")
    * 100
)

print(
    "\nDistribución por fecha:"
)
print(
    respuesta_fecha.to_string(
        index=False
    )
)


# aca se generan las graficas del avance
graficar_distribucion_objetivo(
    datos
)

graficar_eda_ml(
    datos
)


# aca se muestran las variables predictoras sin fuga de informacion
predictoras = pd.DataFrame([
    ["longitud", "Espacial", "Ubicación este-oeste de la observación"],
    ["latitud", "Espacial", "Ubicación norte-sur de la observación"],
    ["B03", "Banda espectral", "Reflectancia verde que no se usa para construir cyano"],
    ["nubosidad_pct", "Ambiental", "Nubosidad reportada para la fecha"],
    ["mes_sin", "Temporal", "Componente cíclico del mes"],
    ["mes_cos", "Temporal", "Componente cíclico del mes"],
    ["dia_anio_sin", "Temporal", "Componente cíclico del día del año"],
    ["dia_anio_cos", "Temporal", "Componente cíclico del día del año"]
], columns=[
    "variable",
    "tipo",
    "descripcion"
])

print(
    "\nVariables predictoras propuestas:"
)
print(
    predictoras.to_string(
        index=False
    )
)

print(
    "\nVariables excluidas por fuga de información:"
)
print(
    "B04, B05, B06, B08, NDVI, NDWI y cyano."
)
