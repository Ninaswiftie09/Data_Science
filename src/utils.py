# Librerías
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.statespace.sarimax import SARIMAX


# Ruta principal
BASE_DIR = Path(__file__).resolve().parents[1]


# Rutas de los datos
RAW_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "guatemala_temperatura.csv"
)

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# Columnas de temperatura
TEMPERATURE_COLUMNS = [
    "temperature_2m_c",
    "skin_temperature_c",
    "soil_temperature_layer_1_c",
    "soil_temperature_layer_2_c",
    "soil_temperature_layer_3_c",
    "soil_temperature_layer_4_c"
]


# Modelos SARIMA
MODELOS_SARIMA = {
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


def cargar_datos():
    # Cargar CSV
    df = pd.read_csv(RAW_FILE)

    # Convertir fecha
    df["month"] = pd.to_datetime(
        df["month"]
    )

    # Ordenar los datos
    df = df.sort_values(
        "month"
    )

    # Reiniciar índices
    df = df.reset_index(
        drop=True
    )

    return df


def crear_carpeta_processed():
    # Crear carpeta
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def crear_serie(
    df,
    columna="temperature_2m_c"
):
    # Crear serie mensual
    serie = (
        df
        .set_index("month")[columna]
        .asfreq("MS")
    )

    # Completar valores faltantes
    if serie.isnull().sum() > 0:
        serie = serie.interpolate(
            method="time"
        )

    return serie


def calcular_metricas(
    reales,
    predicciones
):
    # Convertir a arreglos
    reales = np.asarray(
        reales,
        dtype=float
    )

    predicciones = np.asarray(
        predicciones,
        dtype=float
    )

    # Calcular errores
    mae = mean_absolute_error(
        reales,
        predicciones
    )

    mse = mean_squared_error(
        reales,
        predicciones
    )

    rmse = np.sqrt(mse)

    # Evitar divisiones entre cero
    mascara = reales != 0

    mape = np.mean(
        np.abs(
            (
                reales[mascara]
                - predicciones[mascara]
            )
            / reales[mascara]
        )
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


def ajustar_sarima(
    serie,
    order,
    seasonal_order
):
    # Ocultar advertencias
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        # Crear modelo
        modelo = SARIMAX(
            serie,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        # Entrenar modelo
        ajuste = modelo.fit(
            disp=False,
            maxiter=300
        )

    return ajuste


def entrenar_modelos_sarima(
    serie
):
    # Guardar modelos
    ajustes = {}

    # Guardar métricas
    resultados = []

    # Entrenar cada modelo
    for nombre, parametros in MODELOS_SARIMA.items():

        ajuste = ajustar_sarima(
            serie=serie,
            order=parametros["order"],
            seasonal_order=parametros[
                "seasonal_order"
            ]
        )

        ajustes[nombre] = ajuste

        resultados.append({
            "modelo": nombre,
            "order": str(
                parametros["order"]
            ),
            "seasonal_order": str(
                parametros[
                    "seasonal_order"
                ]
            ),
            "aic": ajuste.aic,
            "bic": ajuste.bic,
            "convergencia": ajuste.mle_retvals.get(
                "converged",
                False
            )
        })

    # Crear tabla
    metricas = pd.DataFrame(
        resultados
    )

    # Ordenar por AIC
    metricas = metricas.sort_values(
        by=["aic", "bic"]
    )

    metricas = metricas.reset_index(
        drop=True
    )

    return ajustes, metricas