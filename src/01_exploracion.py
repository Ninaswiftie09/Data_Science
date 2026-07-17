# Librerías
import numpy as np
import pandas as pd

# Funciones del proyecto
from utils import (
    cargar_datos,
    crear_carpeta_processed,
    PROCESSED_DIR,
    TEMPERATURE_COLUMNS
)


# Cargar los datos
df = cargar_datos()

# Crear la carpeta de resultados
crear_carpeta_processed()


# Exploración general
print("\n--- EXPLORACIÓN GENERAL ---")

print("\nPrimeras filas:")
print(df.head())

print("\nÚltimas filas:")
print(df.tail())

print("\nDimensiones:")
print(df.shape)

print("\nColumnas:")
print(df.columns.tolist())

print("\nTipos de datos:")
print(df.dtypes)

print("\nValores nulos:")
print(df.isnull().sum())


# Rango de fechas
print("\n--- RANGO DE FECHAS ---")

fecha_inicial = df["month"].min()
fecha_final = df["month"].max()

print("Fecha inicial:", fecha_inicial)
print("Fecha final:", fecha_final)


# Extremos de temperatura
print("\n--- TEMPERATURA A 2 METROS ---")

indice_minimo = df["temperature_2m_c"].idxmin()
indice_maximo = df["temperature_2m_c"].idxmax()

temperatura_minima = df.loc[indice_minimo, "temperature_2m_c"]
temperatura_maxima = df.loc[indice_maximo, "temperature_2m_c"]

fecha_minima = df.loc[indice_minimo, "month"]
fecha_maxima = df.loc[indice_maximo, "month"]

print("Temperatura mínima:", round(temperatura_minima, 2), "°C")
print("Fecha de la mínima:", fecha_minima)

print("Temperatura máxima:", round(temperatura_maxima, 2), "°C")
print("Fecha de la máxima:", fecha_maxima)


# Extremos por capa
print("\n--- EXTREMOS POR CAPA ---")

resultados_extremos = []

for columna in TEMPERATURE_COLUMNS:

    # Encontrar los índices
    indice_min = df[columna].idxmin()
    indice_max = df[columna].idxmax()

    # Guardar los resultados
    resultados_extremos.append({
        "capa": columna,
        "temperatura_minima": df.loc[indice_min, columna],
        "fecha_minima": df.loc[indice_min, "month"],
        "temperatura_maxima": df.loc[indice_max, columna],
        "fecha_maxima": df.loc[indice_max, "month"]
    })


# Crear tabla de extremos
extremos_df = pd.DataFrame(resultados_extremos)

print(extremos_df)


# Promedio anual
print("\n--- PROMEDIO ANUAL ---")

promedio_anual = (
    df.groupby("year")[TEMPERATURE_COLUMNS]
    .mean()
    .reset_index()
)

print(promedio_anual.head())


# Tendencia
print("\n--- TENDENCIA ---")

# Crear una secuencia de meses
meses = np.arange(len(df))

# Calcular la tendencia
pendiente_mensual = np.polyfit(
    meses,
    df["temperature_2m_c"],
    1
)[0]

# Convertir la tendencia a años
pendiente_anual = pendiente_mensual * 12

print(
    "Cambio aproximado por año:",
    round(pendiente_anual, 4),
    "°C"
)

if pendiente_anual > 0:
    print("La temperatura presenta una tendencia ascendente.")
elif pendiente_anual < 0:
    print("La temperatura presenta una tendencia descendente.")
else:
    print("La temperatura se mantiene constante.")


# Guardar resultados
extremos_df.to_csv(
    PROCESSED_DIR / "extremos_por_capa.csv",
    index=False
)

promedio_anual.to_csv(
    PROCESSED_DIR / "promedios_anuales.csv",
    index=False
)

print("\nResultados guardados correctamente.")