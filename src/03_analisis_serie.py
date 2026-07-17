# Librerías
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import levene
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

# Funciones del proyecto
from utils import (
    cargar_datos,
    crear_carpeta_processed,
    PROCESSED_DIR
)


# Ocultar advertencias de KPSS
warnings.simplefilter("ignore", InterpolationWarning)


# Configuración
COLUMNA = "temperature_2m_c"
MESES_PRUEBA = 36
PERIODO_ESTACIONAL = 12


# Cargar los datos
df = cargar_datos()

# Crear carpeta de resultados
crear_carpeta_processed()


# Separar entrenamiento
train = df.iloc[:-MESES_PRUEBA].copy()


# Crear la serie temporal
serie = train.set_index("month")[COLUMNA]

# Indicar frecuencia mensual
serie = serie.asfreq("MS")


# Completar posibles valores faltantes
if serie.isnull().sum() > 0:
    serie = serie.interpolate(method="time")


# Información de la serie
print("\n--- SERIE TEMPORAL ---")

print("Variable:", COLUMNA)
print("Cantidad de meses:", len(serie))
print("Fecha inicial:", serie.index.min())
print("Fecha final:", serie.index.max())
print("Valores nulos:", serie.isnull().sum())


# Gráfica de la serie
plt.figure(figsize=(12, 5))

plt.plot(
    serie.index,
    serie.values,
    label="Temperatura"
)

plt.title("Temperatura promedio mensual en Guatemala")
plt.xlabel("Fecha")
plt.ylabel("Temperatura °C")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Calcular la tendencia
meses = np.arange(len(serie))

pendiente_mensual, intercepto = np.polyfit(
    meses,
    serie.values,
    1
)

tendencia_lineal = (
    pendiente_mensual * meses
    + intercepto
)

pendiente_anual = pendiente_mensual * 12
pendiente_decada = pendiente_mensual * 120


# Mostrar tendencia
print("\n--- TENDENCIA ---")

print(
    "Cambio aproximado por año:",
    round(pendiente_anual, 4),
    "°C"
)

print(
    "Cambio aproximado por década:",
    round(pendiente_decada, 4),
    "°C"
)

if pendiente_anual > 0:
    print("La serie presenta una tendencia ascendente.")
elif pendiente_anual < 0:
    print("La serie presenta una tendencia descendente.")
else:
    print("La serie no presenta una tendencia clara.")


# Gráfica de tendencia
plt.figure(figsize=(12, 5))

plt.plot(
    serie.index,
    serie.values,
    label="Temperatura"
)

plt.plot(
    serie.index,
    tendencia_lineal,
    label="Tendencia lineal"
)

plt.title("Tendencia de la temperatura")
plt.xlabel("Fecha")
plt.ylabel("Temperatura °C")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Media móvil
media_movil = serie.rolling(
    window=12
).mean()


# Gráfica de media móvil
plt.figure(figsize=(12, 5))

plt.plot(
    serie.index,
    serie.values,
    label="Temperatura"
)

plt.plot(
    media_movil.index,
    media_movil.values,
    label="Media móvil de 12 meses"
)

plt.title("Media móvil de la serie")
plt.xlabel("Fecha")
plt.ylabel("Temperatura °C")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Varianza móvil
varianza_movil = serie.rolling(
    window=12
).var()


# Gráfica de varianza móvil
plt.figure(figsize=(12, 5))

plt.plot(
    varianza_movil.index,
    varianza_movil.values
)

plt.title("Varianza móvil de 12 meses")
plt.xlabel("Fecha")
plt.ylabel("Varianza")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Descomponer la serie
descomposicion = seasonal_decompose(
    serie,
    model="additive",
    period=PERIODO_ESTACIONAL,
    extrapolate_trend="freq"
)


# Mostrar componentes
figura = descomposicion.plot()

figura.set_size_inches(12, 9)
figura.suptitle(
    "Descomposición de la serie",
    y=1.02
)

plt.tight_layout()
plt.show()


# Prueba ADF
print("\n--- PRUEBA ADF ---")

resultado_adf = adfuller(
    serie.dropna(),
    autolag="AIC"
)

estadistico_adf = resultado_adf[0]
p_valor_adf = resultado_adf[1]

print(
    "Estadístico ADF:",
    round(estadistico_adf, 4)
)

print(
    "Valor p:",
    round(p_valor_adf, 6)
)

if p_valor_adf < 0.05:
    print("ADF indica que la serie es estacionaria.")
else:
    print("ADF indica que la serie no es estacionaria.")


# Prueba KPSS en media
print("\n--- PRUEBA KPSS EN MEDIA ---")

resultado_kpss_media = kpss(
    serie.dropna(),
    regression="c",
    nlags="auto"
)

estadistico_kpss_media = resultado_kpss_media[0]
p_valor_kpss_media = resultado_kpss_media[1]

print(
    "Estadístico KPSS:",
    round(estadistico_kpss_media, 4)
)

print(
    "Valor p:",
    round(p_valor_kpss_media, 6)
)

if p_valor_kpss_media < 0.05:
    print("KPSS indica que la media no es estacionaria.")
else:
    print("KPSS indica que la media es estacionaria.")


# Prueba KPSS con tendencia
print("\n--- PRUEBA KPSS CON TENDENCIA ---")

resultado_kpss_tendencia = kpss(
    serie.dropna(),
    regression="ct",
    nlags="auto"
)

estadistico_kpss_tendencia = resultado_kpss_tendencia[0]
p_valor_kpss_tendencia = resultado_kpss_tendencia[1]

print(
    "Estadístico KPSS:",
    round(estadistico_kpss_tendencia, 4)
)

print(
    "Valor p:",
    round(p_valor_kpss_tendencia, 6)
)

if p_valor_kpss_tendencia < 0.05:
    print("La serie no es estacionaria alrededor de una tendencia.")
else:
    print("La serie puede ser estacionaria alrededor de una tendencia.")


# Separar la serie en dos partes
mitad = len(serie) // 2

primera_mitad = serie.iloc[:mitad]
segunda_mitad = serie.iloc[mitad:]


# Comparar varianzas
varianza_primera = primera_mitad.var()
varianza_segunda = segunda_mitad.var()

resultado_levene = levene(
    primera_mitad,
    segunda_mitad
)

p_valor_levene = resultado_levene.pvalue


# Mostrar comparación
print("\n--- ESTACIONARIEDAD EN VARIANZA ---")

print(
    "Varianza de la primera mitad:",
    round(varianza_primera, 4)
)

print(
    "Varianza de la segunda mitad:",
    round(varianza_segunda, 4)
)

print(
    "Valor p de Levene:",
    round(p_valor_levene, 6)
)

if p_valor_levene < 0.05:
    print("Las varianzas presentan diferencias significativas.")
else:
    print("No se encontraron diferencias significativas entre las varianzas.")


# Guardar los componentes
componentes = pd.DataFrame({
    "month": serie.index,
    "observado": descomposicion.observed.values,
    "tendencia": descomposicion.trend.values,
    "estacionalidad": descomposicion.seasonal.values,
    "residuo": descomposicion.resid.values
})


# Guardar resultados
componentes.to_csv(
    PROCESSED_DIR / "componentes_serie.csv",
    index=False
)


# Guardar pruebas
pruebas = pd.DataFrame({
    "prueba": [
        "ADF",
        "KPSS media",
        "KPSS tendencia",
        "Levene"
    ],
    "estadistico": [
        estadistico_adf,
        estadistico_kpss_media,
        estadistico_kpss_tendencia,
        resultado_levene.statistic
    ],
    "p_valor": [
        p_valor_adf,
        p_valor_kpss_media,
        p_valor_kpss_tendencia,
        p_valor_levene
    ]
})


pruebas.to_csv(
    PROCESSED_DIR / "pruebas_estacionariedad.csv",
    index=False
)


print("\nResultados guardados correctamente.")