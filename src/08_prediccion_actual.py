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
MESES_ACTUALES = 12


# Cargar datos
df = cargar_datos()

# Crear carpeta
crear_carpeta_processed()


# Crear serie completa
serie = crear_serie(
    df,
    COLUMNA
)


# Separar últimos 12 meses
entrenamiento = serie.iloc[
    :-MESES_ACTUALES
]

valores_actuales = serie.iloc[
    -MESES_ACTUALES:
]


print(
    "\n--- EVALUACIÓN DE VALORES ACTUALES ---"
)

print(
    "Entrenamiento hasta:",
    entrenamiento.index.max()
)

print(
    "Evaluación desde:",
    valores_actuales.index.min()
)

print(
    "Evaluación hasta:",
    valores_actuales.index.max()
)


# Entrenar modelos
ajustes, metricas_aic = (
    entrenar_modelos_sarima(
        entrenamiento
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
    "\nModelo utilizado:",
    mejor_nombre
)


# Crear predicción
resultado_forecast = (
    mejor_ajuste
    .get_forecast(
        steps=MESES_ACTUALES
    )
)


prediccion = (
    resultado_forecast
    .predicted_mean
)


intervalo = (
    resultado_forecast
    .conf_int(
        alpha=0.05
    )
)


# Ajustar índices
prediccion.index = (
    valores_actuales.index
)

intervalo.index = (
    valores_actuales.index
)


# Calcular métricas
metricas = calcular_metricas(
    valores_actuales.values,
    prediccion.values
)


print(
    "\n--- MÉTRICAS ACTUALES ---"
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


# Verificar intervalo
dentro_intervalo = (
    (
        valores_actuales.values
        >= intervalo.iloc[:, 0].values
    )
    &
    (
        valores_actuales.values
        <= intervalo.iloc[:, 1].values
    )
)


cobertura = (
    dentro_intervalo.mean()
    * 100
)


print(
    "Cobertura del intervalo:",
    round(
        cobertura,
        2
    ),
    "%"
)


# Conclusión
print(
    "\n--- CONCLUSIÓN ---"
)


if (
    metricas["MAPE"] < 5
    and cobertura >= 80
):

    conclusion = (
        "El modelo tiene una buena capacidad "
        "para predecir los valores más recientes."
    )

elif metricas["MAPE"] < 10:

    conclusion = (
        "El modelo tiene una capacidad "
        "aceptable para predecir los valores "
        "más recientes."
    )

else:

    conclusion = (
        "El modelo presenta un error considerable "
        "al predecir los valores más recientes."
    )


print(conclusion)


# Crear tabla
resultados = pd.DataFrame({
    "month": valores_actuales.index,
    "real": valores_actuales.values,
    "prediccion": prediccion.values,
    "limite_inferior": intervalo.iloc[
        :, 0
    ].values,
    "limite_superior": intervalo.iloc[
        :, 1
    ].values,
    "dentro_intervalo": dentro_intervalo
})


# Gráfica
plt.figure(
    figsize=(12, 6)
)

plt.plot(
    entrenamiento.index[-36:],
    entrenamiento.values[-36:],
    label="Entrenamiento"
)

plt.plot(
    valores_actuales.index,
    valores_actuales.values,
    label="Valores actuales"
)

plt.plot(
    prediccion.index,
    prediccion.values,
    label="Predicción"
)

plt.fill_between(
    valores_actuales.index,
    intervalo.iloc[:, 0],
    intervalo.iloc[:, 1],
    alpha=0.2,
    label="Intervalo de confianza"
)

plt.title(
    "Predicción de los valores más recientes"
)

plt.xlabel("Fecha")
plt.ylabel("Temperatura °C")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Guardar predicciones
resultados.to_csv(
    PROCESSED_DIR
    / "prediccion_valores_actuales.csv",
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
    ],
    "cobertura_intervalo": [
        cobertura
    ],
    "conclusion": [
        conclusion
    ]
})


tabla_metricas.to_csv(
    PROCESSED_DIR
    / "metricas_valores_actuales.csv",
    index=False
)


print(
    "\nResultados guardados correctamente."
)