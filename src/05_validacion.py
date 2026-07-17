# Librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.diagnostic import (
    acorr_ljungbox
)

from statsmodels.graphics.tsaplots import (
    plot_acf
)


# Funciones del proyecto
from utils import (
    cargar_datos,
    crear_carpeta_processed,
    crear_serie,
    entrenar_modelos_sarima,
    MODELOS_SARIMA,
    PROCESSED_DIR
)


# Configuración
COLUMNA = "temperature_2m_c"
MESES_PRUEBA = 36


# Cargar datos
df = cargar_datos()

# Crear carpeta
crear_carpeta_processed()


# Separar entrenamiento
train = df.iloc[
    :-MESES_PRUEBA
].copy()


# Crear serie
serie = crear_serie(
    train,
    COLUMNA
)


# Entrenar modelos
ajustes, metricas = (
    entrenar_modelos_sarima(
        serie
    )
)


# Seleccionar mejor modelo
mejor_nombre = (
    metricas
    .iloc[0]["modelo"]
)

mejor_ajuste = ajustes[
    mejor_nombre
]

parametros = MODELOS_SARIMA[
    mejor_nombre
]


print(
    "\n--- MODELO SELECCIONADO ---"
)

print(
    "Modelo:",
    mejor_nombre
)

print(
    "Orden:",
    parametros["order"]
)

print(
    "Orden estacional:",
    parametros[
        "seasonal_order"
    ]
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


# Coeficientes
coeficientes = pd.DataFrame({
    "coeficiente": mejor_ajuste.params.index,
    "valor": mejor_ajuste.params.values,
    "p_valor": mejor_ajuste.pvalues.values
})


# Determinar significación
coeficientes["significativo"] = (
    coeficientes["p_valor"]
    < 0.05
)


print(
    "\n--- COEFICIENTES ---"
)

print(
    coeficientes.to_string(
        index=False
    )
)


# Revisar coeficientes
no_significativos = (
    coeficientes[
        coeficientes[
            "significativo"
        ] == False
    ]
)


if len(no_significativos) == 0:
    print(
        "\nTodos los coeficientes son significativos."
    )
else:
    print(
        "\nHay coeficientes no significativos."
    )


# Obtener raíces
raices_ar = mejor_ajuste.arroots
raices_ma = mejor_ajuste.maroots


# Crear tabla de raíces
filas_raices = []


for raiz in raices_ar:

    filas_raices.append({
        "tipo": "AR",
        "parte_real": np.real(
            raiz
        ),
        "parte_imaginaria": np.imag(
            raiz
        ),
        "modulo": np.abs(
            raiz
        ),
        "fuera_circulo_unitario": (
            np.abs(raiz) > 1
        )
    })


for raiz in raices_ma:

    filas_raices.append({
        "tipo": "MA",
        "parte_real": np.real(
            raiz
        ),
        "parte_imaginaria": np.imag(
            raiz
        ),
        "modulo": np.abs(
            raiz
        ),
        "fuera_circulo_unitario": (
            np.abs(raiz) > 1
        )
    })


tabla_raices = pd.DataFrame(
    filas_raices
)


print(
    "\n--- RAÍCES DEL MODELO ---"
)

print(
    tabla_raices.to_string(
        index=False
    )
)


# Buscar raíces comunes
raices_comunes = []

TOLERANCIA = 0.05


for raiz_ar in raices_ar:

    for raiz_ma in raices_ma:

        distancia = np.abs(
            raiz_ar
            - raiz_ma
        )

        if distancia < TOLERANCIA:

            raices_comunes.append({
                "raiz_ar": str(
                    raiz_ar
                ),
                "raiz_ma": str(
                    raiz_ma
                ),
                "distancia": distancia
            })


tabla_comunes = pd.DataFrame(
    raices_comunes
)


print(
    "\n--- RAÍCES COMUNES ---"
)


if len(tabla_comunes) == 0:

    print(
        "No se encontraron raíces comunes."
    )

else:

    print(
        tabla_comunes.to_string(
            index=False
        )
    )


# Obtener residuos
residuos = (
    mejor_ajuste
    .resid
    .dropna()
)


# Eliminar valores iniciales
inicio = (
    mejor_ajuste
    .loglikelihood_burn
)

if inicio > 0:
    residuos = residuos.iloc[
        inicio:
    ]


# Gráfica de residuos
plt.figure(
    figsize=(12, 5)
)

plt.plot(
    residuos.index,
    residuos.values
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title(
    "Residuos del mejor modelo"
)

plt.xlabel("Fecha")
plt.ylabel("Residuo")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ACF de residuos
plot_acf(
    residuos,
    lags=36
)

plt.title(
    "Autocorrelación de los residuos"
)

plt.xlabel("Rezagos")
plt.ylabel("Autocorrelación")
plt.tight_layout()
plt.show()


# Diagnósticos
mejor_ajuste.plot_diagnostics(
    figsize=(12, 8),
    lags=36
)

plt.tight_layout()
plt.show()


# Grados del modelo
p, d, q = parametros[
    "order"
]

P, D, Q, s = parametros[
    "seasonal_order"
]

grados_modelo = (
    p
    + q
    + P
    + Q
)


# Prueba Ljung-Box
ljung_box = acorr_ljungbox(
    residuos,
    lags=[
        12,
        24,
        36
    ],
    model_df=grados_modelo,
    return_df=True
)


ljung_box = (
    ljung_box
    .reset_index()
    .rename(
        columns={
            "index": "rezago",
            "lb_stat": "estadistico",
            "lb_pvalue": "p_valor"
        }
    )
)


print(
    "\n--- PRUEBA LJUNG-BOX ---"
)

print(
    ljung_box.to_string(
        index=False
    )
)


# Conclusión de residuos
if (
    ljung_box["p_valor"]
    .dropna()
    .gt(0.05)
    .all()
):

    print(
        "\nLos residuos no presentan "
        "autocorrelación significativa."
    )

else:

    print(
        "\nAlgunos residuos presentan "
        "autocorrelación significativa."
    )


# Guardar resultados
coeficientes.to_csv(
    PROCESSED_DIR
    / "coeficientes_mejor_modelo.csv",
    index=False
)


tabla_raices.to_csv(
    PROCESSED_DIR
    / "raices_mejor_modelo.csv",
    index=False
)


tabla_comunes.to_csv(
    PROCESSED_DIR
    / "raices_comunes.csv",
    index=False
)


ljung_box.to_csv(
    PROCESSED_DIR
    / "prueba_ljung_box.csv",
    index=False
)


metricas.to_csv(
    PROCESSED_DIR
    / "comparacion_aic_bic.csv",
    index=False
)


# Guardar nombre del modelo
with open(
    PROCESSED_DIR
    / "mejor_modelo.txt",
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write(
        f"Modelo: {mejor_nombre}\n"
    )

    archivo.write(
        f"Orden: {parametros['order']}\n"
    )

    archivo.write(
        "Orden estacional: "
        f"{parametros['seasonal_order']}\n"
    )

    archivo.write(
        f"AIC: {mejor_ajuste.aic}\n"
    )

    archivo.write(
        f"BIC: {mejor_ajuste.bic}\n"
    )


print(
    "\nResultados guardados correctamente."
)