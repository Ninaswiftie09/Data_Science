# Librerías
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import (
    adfuller,
    kpss
)

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)

from statsmodels.tools.sm_exceptions import (
    InterpolationWarning
)


# Funciones del proyecto
from utils import (
    cargar_datos,
    crear_carpeta_processed,
    crear_serie,
    entrenar_modelos_sarima,
    PROCESSED_DIR
)


# Ocultar advertencias
warnings.simplefilter(
    "ignore",
    InterpolationWarning
)


# Configuración
COLUMNA = "temperature_2m_c"
MESES_PRUEBA = 36
PERIODO = 12


# Cargar datos
df = cargar_datos()

# Crear carpeta
crear_carpeta_processed()


# Separar entrenamiento
train = df.iloc[
    :-MESES_PRUEBA
].copy()


# Crear serie temporal
serie = crear_serie(
    train,
    COLUMNA
)


# Diferenciación estacional
serie_transformada = (
    serie
    .diff(PERIODO)
    .dropna()
)


# Información
print(
    "\n--- TRANSFORMACIÓN DE LA SERIE ---"
)

print(
    "Datos originales:",
    len(serie)
)

print(
    "Datos transformados:",
    len(serie_transformada)
)

print(
    "Diferenciación aplicada:",
    PERIODO,
    "meses"
)


# Gráfica transformada
plt.figure(
    figsize=(12, 5)
)

plt.plot(
    serie_transformada.index,
    serie_transformada.values
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title(
    "Serie con diferenciación estacional"
)

plt.xlabel("Fecha")
plt.ylabel("Diferencia de temperatura")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Prueba ADF
resultado_adf = adfuller(
    serie_transformada,
    autolag="AIC"
)

estadistico_adf = resultado_adf[0]
p_valor_adf = resultado_adf[1]


print(
    "\n--- ADF DESPUÉS DE LA TRANSFORMACIÓN ---"
)

print(
    "Estadístico:",
    round(estadistico_adf, 4)
)

print(
    "Valor p:",
    round(p_valor_adf, 6)
)

if p_valor_adf < 0.05:
    print(
        "La serie es estacionaria según ADF."
    )
else:
    print(
        "La serie no es estacionaria según ADF."
    )


# Prueba KPSS
resultado_kpss = kpss(
    serie_transformada,
    regression="c",
    nlags="auto"
)

estadistico_kpss = resultado_kpss[0]
p_valor_kpss = resultado_kpss[1]


print(
    "\n--- KPSS DESPUÉS DE LA TRANSFORMACIÓN ---"
)

print(
    "Estadístico:",
    round(estadistico_kpss, 4)
)

print(
    "Valor p:",
    round(p_valor_kpss, 6)
)

if p_valor_kpss < 0.05:
    print(
        "La serie no es estacionaria según KPSS."
    )
else:
    print(
        "La serie es estacionaria según KPSS."
    )


# Gráfica ACF
plot_acf(
    serie_transformada,
    lags=36
)

plt.title(
    "Función de autocorrelación"
)

plt.xlabel("Rezagos")
plt.ylabel("Autocorrelación")
plt.tight_layout()
plt.show()


# Gráfica PACF
plot_pacf(
    serie_transformada,
    lags=36,
    method="ywm"
)

plt.title(
    "Autocorrelación parcial"
)

plt.xlabel("Rezagos")
plt.ylabel(
    "Autocorrelación parcial"
)

plt.tight_layout()
plt.show()


# Entrenar modelos
print(
    "\n--- ENTRENAMIENTO DE MODELOS ---"
)

ajustes, metricas = (
    entrenar_modelos_sarima(
        serie
    )
)


# Mostrar resultados
for _, fila in metricas.iterrows():

    print(
        "\nModelo:",
        fila["modelo"]
    )

    print(
        "Orden:",
        fila["order"]
    )

    print(
        "Orden estacional:",
        fila["seasonal_order"]
    )

    print(
        "AIC:",
        round(fila["aic"], 4)
    )

    print(
        "BIC:",
        round(fila["bic"], 4)
    )

    print(
        "Convergencia:",
        fila["convergencia"]
    )


# Mejor modelo
mejor_nombre = (
    metricas
    .iloc[0]["modelo"]
)

mejor_ajuste = ajustes[
    mejor_nombre
]


print(
    "\n--- MEJOR MODELO ---"
)

print(
    "Modelo:",
    mejor_nombre
)

print(
    "AIC:",
    round(
        mejor_ajuste.aic,
        4
    )
)

print(
    "BIC:",
    round(
        mejor_ajuste.bic,
        4
    )
)


# Guardar resúmenes
for nombre, ajuste in ajustes.items():

    ruta = (
        PROCESSED_DIR
        / f"resumen_{nombre}.txt"
    )

    with open(
        ruta,
        "w",
        encoding="utf-8"
    ) as archivo:

        archivo.write(
            ajuste
            .summary()
            .as_text()
        )


# Guardar serie
serie_transformada.to_csv(
    PROCESSED_DIR
    / "serie_transformada.csv",
    header=[
        "temperature_diff_12"
    ]
)


# Guardar métricas
metricas.to_csv(
    PROCESSED_DIR
    / "metricas_arima.csv",
    index=False
)


# Guardar pruebas
pruebas = pd.DataFrame({
    "prueba": [
        "ADF",
        "KPSS"
    ],
    "estadistico": [
        estadistico_adf,
        estadistico_kpss
    ],
    "p_valor": [
        p_valor_adf,
        p_valor_kpss
    ]
})


pruebas.to_csv(
    PROCESSED_DIR
    / "pruebas_transformacion.csv",
    index=False
)


print(
    "\nResultados guardados correctamente."
)