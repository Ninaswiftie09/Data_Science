# Librerías
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    InterpolationWarning
)

# Funciones del proyecto
from utils import (
    cargar_datos,
    crear_carpeta_processed,
    PROCESSED_DIR
)


# Ocultar advertencias
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", InterpolationWarning)


# Configuración
COLUMNA = "temperature_2m_c"
MESES_PRUEBA = 36
PERIODO_ESTACIONAL = 12


# Cargar datos
df = cargar_datos()

# Crear carpeta de resultados
crear_carpeta_processed()


# Separar entrenamiento
train = df.iloc[:-MESES_PRUEBA].copy()


# Crear serie temporal
serie = train.set_index("month")[COLUMNA]

# Indicar frecuencia mensual
serie = serie.asfreq("MS")


# Diferenciación estacional
serie_estacionaria = serie.diff(
    PERIODO_ESTACIONAL
).dropna()


# Mostrar información
print("\n--- TRANSFORMACIÓN DE LA SERIE ---")

print("Datos originales:", len(serie))
print("Datos transformados:", len(serie_estacionaria))
print("Diferenciación aplicada: 12 meses")


# Gráfica transformada
plt.figure(figsize=(12, 5))

plt.plot(
    serie_estacionaria.index,
    serie_estacionaria.values
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Serie con diferenciación estacional")
plt.xlabel("Fecha")
plt.ylabel("Diferencia de temperatura")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Prueba ADF
resultado_adf = adfuller(
    serie_estacionaria,
    autolag="AIC"
)

estadistico_adf = resultado_adf[0]
p_valor_adf = resultado_adf[1]


print("\n--- ADF DESPUÉS DE LA TRANSFORMACIÓN ---")

print(
    "Estadístico:",
    round(estadistico_adf, 4)
)

print(
    "Valor p:",
    round(p_valor_adf, 6)
)

if p_valor_adf < 0.05:
    print("La serie transformada es estacionaria según ADF.")
else:
    print("La serie transformada no es estacionaria según ADF.")


# Prueba KPSS
resultado_kpss = kpss(
    serie_estacionaria,
    regression="c",
    nlags="auto"
)

estadistico_kpss = resultado_kpss[0]
p_valor_kpss = resultado_kpss[1]


print("\n--- KPSS DESPUÉS DE LA TRANSFORMACIÓN ---")

print(
    "Estadístico:",
    round(estadistico_kpss, 4)
)

print(
    "Valor p:",
    round(p_valor_kpss, 6)
)

if p_valor_kpss < 0.05:
    print("La serie transformada no es estacionaria según KPSS.")
else:
    print("La serie transformada es estacionaria según KPSS.")


# Gráfica ACF
plot_acf(
    serie_estacionaria,
    lags=36
)

plt.title("Función de autocorrelación")
plt.xlabel("Rezagos")
plt.ylabel("Autocorrelación")
plt.tight_layout()
plt.show()


# Gráfica PACF
plot_pacf(
    serie_estacionaria,
    lags=36,
    method="ywm"
)

plt.title("Función de autocorrelación parcial")
plt.xlabel("Rezagos")
plt.ylabel("Autocorrelación parcial")
plt.tight_layout()
plt.show()


# Configuración de modelos
modelos = {
    "modelo_1": {
        "order": (1, 0, 1),
        "seasonal_order": (0, 1, 1, 12)
    },

    "modelo_2": {
        "order": (2, 0, 0),
        "seasonal_order": (0, 1, 1, 12)
    },

    "modelo_3": {
        "order": (0, 0, 2),
        "seasonal_order": (0, 1, 1, 12)
    }
}


# Lista de resultados
resultados_modelos = []


# Entrenar los modelos
print("\n--- ENTRENAMIENTO DE MODELOS ---")

for nombre, parametros in modelos.items():

    print("\nEntrenando:", nombre)

    # Crear modelo
    modelo = SARIMAX(
        serie,
        order=parametros["order"],
        seasonal_order=parametros["seasonal_order"],
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Entrenar modelo
    ajuste = modelo.fit(
        disp=False,
        maxiter=300
    )

    # Guardar resultados
    resultados_modelos.append({
        "modelo": nombre,
        "order": str(parametros["order"]),
        "seasonal_order": str(
            parametros["seasonal_order"]
        ),
        "aic": ajuste.aic,
        "bic": ajuste.bic,
        "convergencia": ajuste.mle_retvals.get(
            "converged",
            False
        )
    })

    # Mostrar métricas
    print("AIC:", round(ajuste.aic, 4))
    print("BIC:", round(ajuste.bic, 4))

    # Guardar resumen
    ruta_resumen = (
        PROCESSED_DIR
        / f"resumen_{nombre}.txt"
    )

    with open(
        ruta_resumen,
        "w",
        encoding="utf-8"
    ) as archivo:

        archivo.write(
            ajuste.summary().as_text()
        )


# Crear tabla
metricas = pd.DataFrame(
    resultados_modelos
)


# Ordenar por AIC
metricas = metricas.sort_values(
    by="aic"
)


# Mostrar comparación
print("\n--- COMPARACIÓN DE MODELOS ---")

print(
    metricas.to_string(index=False)
)


# Mostrar mejor modelo
mejor_modelo = metricas.iloc[0]

print("\nMejor modelo según AIC:")
print(mejor_modelo["modelo"])

print(
    "AIC:",
    round(mejor_modelo["aic"], 4)
)

print(
    "BIC:",
    round(mejor_modelo["bic"], 4)
)


# Guardar serie transformada
serie_estacionaria.to_csv(
    PROCESSED_DIR
    / "serie_transformada.csv",
    header=["temperature_diff_12"]
)


# Guardar métricas
metricas.to_csv(
    PROCESSED_DIR
    / "metricas_arima.csv",
    index=False
)


print("\nResultados guardados correctamente.")