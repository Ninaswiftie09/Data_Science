# Librerías
import pandas as pd
import matplotlib.pyplot as plt


# Funciones del proyecto
from utils import (
    cargar_datos,
    crear_carpeta_processed,
    crear_serie,
    entrenar_modelos_sarima,
    calcular_metricas,
    PROCESSED_DIR
)


# Configuración
COLUMNA = "temperature_2m_c"
MESES_PRUEBA = 36


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


# Entrenar modelos
ajustes, metricas_aic = (
    entrenar_modelos_sarima(
        train
    )
)


# Seleccionar mejor modelo
mejor_nombre = (
    metricas_aic
    .iloc[0]["modelo"]
)

mejor_ajuste = ajustes[
    mejor_nombre
]


print(
    "\n--- PREDICCIÓN SARIMA ---"
)

print(
    "Modelo utilizado:",
    mejor_nombre
)


# Crear predicción
resultado_forecast = (
    mejor_ajuste
    .get_forecast(
        steps=len(test)
    )
)


# Valores predichos
prediccion = (
    resultado_forecast
    .predicted_mean
)


# Intervalo de confianza
intervalo = (
    resultado_forecast
    .conf_int(
        alpha=0.05
    )
)


# Ajustar índices
prediccion.index = test.index
intervalo.index = test.index


# Calcular métricas
metricas = calcular_metricas(
    test.values,
    prediccion.values
)


print(
    "MAE:",
    round(
        metricas["MAE"],
        4
    )
)

print(
    "RMSE:",
    round(
        metricas["RMSE"],
        4
    )
)

print(
    "MAPE:",
    round(
        metricas["MAPE"],
        4
    ),
    "%"
)


# Crear tabla
resultados = pd.DataFrame({
    "month": test.index,
    "real": test.values,
    "prediccion": prediccion.values,
    "limite_inferior": intervalo.iloc[
        :, 0
    ].values,
    "limite_superior": intervalo.iloc[
        :, 1
    ].values
})


# Gráfica de predicción
plt.figure(
    figsize=(13, 6)
)

plt.plot(
    train.index[-60:],
    train.values[-60:],
    label="Entrenamiento"
)

plt.plot(
    test.index,
    test.values,
    label="Valores reales"
)

plt.plot(
    prediccion.index,
    prediccion.values,
    label="Predicción SARIMA"
)

plt.fill_between(
    test.index,
    intervalo.iloc[:, 0],
    intervalo.iloc[:, 1],
    alpha=0.2,
    label="Intervalo de confianza"
)

plt.title(
    "Predicción del conjunto de prueba"
)

plt.xlabel("Fecha")
plt.ylabel("Temperatura °C")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Guardar predicción
resultados.to_csv(
    PROCESSED_DIR
    / "prediccion_sarima.csv",
    index=False
)


# Guardar métricas
tabla_metricas = pd.DataFrame({
    "modelo": [
        mejor_nombre
    ],
    "MAE": [
        metricas["MAE"]
    ],
    "RMSE": [
        metricas["RMSE"]
    ],
    "MAPE": [
        metricas["MAPE"]
    ]
})


tabla_metricas.to_csv(
    PROCESSED_DIR
    / "metricas_prediccion_sarima.csv",
    index=False
)


print(
    "\nResultados guardados correctamente."
)