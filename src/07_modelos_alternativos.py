# Librerías
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    SimpleExpSmoothing
)


# Funciones del proyecto
from utils import (
    cargar_datos,
    crear_carpeta_processed,
    crear_serie,
    entrenar_modelos_sarima,
    calcular_metricas,
    PROCESSED_DIR
)


# Ocultar advertencias
warnings.simplefilter(
    "ignore"
)


# Configuración
COLUMNA = "temperature_2m_c"
MESES_PRUEBA = 36
PERIODO = 12


# Cargar datos
df = cargar_datos()

# Crear carpeta
crear_carpeta_processed()


# Separar datos
train_df = df.iloc[
    :-MESES_PRUEBA
].copy()

test_df = df.iloc[
    -MESES_PRUEBA:
].copy()


# Crear series
train = crear_serie(
    train_df,
    COLUMNA
)

test = crear_serie(
    test_df,
    COLUMNA
)


# Modelo SARIMA
ajustes, metricas_aic = (
    entrenar_modelos_sarima(
        train
    )
)

mejor_nombre = (
    metricas_aic
    .iloc[0]["modelo"]
)

mejor_sarima = ajustes[
    mejor_nombre
]

pred_sarima = (
    mejor_sarima
    .forecast(
        steps=len(test)
    )
)

pred_sarima.index = test.index


# Modelo Holt-Winters
modelo_hw = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=PERIODO,
    initialization_method="estimated"
)


ajuste_hw = modelo_hw.fit(
    optimized=True
)


pred_hw = ajuste_hw.forecast(
    len(test)
)

pred_hw.index = test.index


# Suavizamiento exponencial
modelo_ses = SimpleExpSmoothing(
    train,
    initialization_method="estimated"
)


ajuste_ses = modelo_ses.fit(
    optimized=True
)


pred_ses = ajuste_ses.forecast(
    len(test)
)

pred_ses.index = test.index


# Seasonal naive
ultimos_meses = (
    train
    .iloc[-PERIODO:]
    .values
)


repeticiones = int(
    np.ceil(
        len(test)
        / PERIODO
    )
)


pred_naive_valores = np.tile(
    ultimos_meses,
    repeticiones
)[:len(test)]


pred_naive = pd.Series(
    pred_naive_valores,
    index=test.index
)


# Guardar predicciones
predicciones = {
    f"SARIMA {mejor_nombre}": pred_sarima,
    "Holt-Winters": pred_hw,
    "Suavizamiento exponencial": pred_ses,
    "Seasonal naive": pred_naive
}


# Calcular métricas
filas_metricas = []


for nombre, prediccion in predicciones.items():

    resultado = calcular_metricas(
        test.values,
        prediccion.values
    )

    filas_metricas.append({
        "modelo": nombre,
        "MAE": resultado["MAE"],
        "RMSE": resultado["RMSE"],
        "MAPE": resultado["MAPE"]
    })


# Crear tabla
metricas = pd.DataFrame(
    filas_metricas
)


# Ordenar por RMSE
metricas = metricas.sort_values(
    by="RMSE"
)

metricas = metricas.reset_index(
    drop=True
)


print(
    "\n--- COMPARACIÓN DE MODELOS ---"
)

print(
    metricas.to_string(
        index=False
    )
)


# Mejor modelo
mejor_modelo = (
    metricas
    .iloc[0]["modelo"]
)


print(
    "\nMejor modelo según RMSE:",
    mejor_modelo
)


# Crear tabla de predicciones
tabla_predicciones = pd.DataFrame({
    "month": test.index,
    "real": test.values,
    "sarima": pred_sarima.values,
    "holt_winters": pred_hw.values,
    "suavizamiento_exponencial": (
        pred_ses.values
    ),
    "seasonal_naive": pred_naive.values
})


# Gráfica
plt.figure(
    figsize=(13, 7)
)

plt.plot(
    test.index,
    test.values,
    label="Valores reales",
    linewidth=2
)

plt.plot(
    test.index,
    pred_sarima.values,
    label="SARIMA"
)

plt.plot(
    test.index,
    pred_hw.values,
    label="Holt-Winters"
)

plt.plot(
    test.index,
    pred_ses.values,
    label="Suavizamiento exponencial"
)

plt.plot(
    test.index,
    pred_naive.values,
    label="Seasonal naive"
)

plt.title(
    "Comparación de predicciones"
)

plt.xlabel("Fecha")
plt.ylabel("Temperatura °C")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Guardar resultados
tabla_predicciones.to_csv(
    PROCESSED_DIR
    / "predicciones_modelos.csv",
    index=False
)


metricas.to_csv(
    PROCESSED_DIR
    / "metricas_modelos.csv",
    index=False
)


print(
    "\nResultados guardados correctamente."
)