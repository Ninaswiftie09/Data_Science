from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openeo
import pandas as pd
import rasterio

from config import (
    BANDAS,
    CONEXION_URL,
    DATOS_DIR,
    GRAFICOS_DIR,
    RESOLUCION_METROS
)


# aca se conecta con copernicus
def conectar_copernicus():
    conn = openeo.connect(CONEXION_URL)
    conn.authenticate_oidc()

    print("Conexión realizada correctamente.")
    print("Backend:", CONEXION_URL)

    return conn


# aca se limpia el nombre para usarlo en archivos
def nombre_seguro(texto):
    return (
        texto.lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
        .replace(" ", "_")
    )


# aca se evita dividir entre cero
def division_segura(numerador, denominador):
    salida = np.full_like(
        numerador,
        np.nan,
        dtype="float32"
    )

    valido = (
        np.isfinite(denominador)
        & (np.abs(denominador) > 1e-8)
    )

    salida[valido] = (
        numerador[valido]
        / denominador[valido]
    )

    return salida


# aca se descargan solo las bandas necesarias
def descargar_bandas_fecha(
    conn,
    lago,
    bbox,
    fecha,
    resolucion=RESOLUCION_METROS
):
    slug = nombre_seguro(lago)
    salida = DATOS_DIR / f"{slug}_{fecha}.tif"

    if salida.exists():
        print(f"Ya existe: {salida}")
        return salida

    fecha_inicio = pd.Timestamp(fecha)

    fecha_fin = (
        fecha_inicio
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    print(f"Descargando {lago} - {fecha}...")

    cubo = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=[
            fecha,
            fecha_fin
        ],
        bands=BANDAS
    )

    cubo = cubo.resample_spatial(
        resolution=resolucion,
        method="near"
    )

    cubo = cubo.max_time()

    cubo.download(
        str(salida),
        format="GTiff"
    )

    print(f"Guardado: {salida}")

    return salida


# aca se calculan los indices
def leer_y_calcular_indices(ruta_tif):
    ruta_tif = Path(ruta_tif)

    with rasterio.open(ruta_tif) as src:
        data = src.read().astype("float32")
        perfil = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if data.shape[0] != len(BANDAS):
        raise ValueError(
            f"Se esperaban {len(BANDAS)} bandas, "
            f"pero el archivo tiene {data.shape[0]}."
        )

    valores = {
        banda: data[i]
        for i, banda in enumerate(BANDAS)
    }

    p99_rojo = np.nanpercentile(
        valores["B04"],
        99
    )

    escala = (
        0.0001
        if p99_rojo > 2
        else 1.0
    )

    for banda in [
        "B03",
        "B04",
        "B05",
        "B06",
        "B08"
    ]:
        valores[banda] = (
            valores[banda]
            * escala
        )

    b03 = valores["B03"]
    b04 = valores["B04"]
    b05 = valores["B05"]
    b06 = valores["B06"]
    b08 = valores["B08"]
    scl = valores["SCL"]

    ndvi = division_segura(
        b08 - b04,
        b08 + b04
    )

    ndwi = division_segura(
        b03 - b08,
        b03 + b08
    )

    mci = (
        b05
        - b04
        - (
            b06 - b04
        ) * (
            (705 - 665)
            / (740 - 665)
        )
    )

    fai = (
        b06
        - b04
        - (
            b08 - b04
        ) * (
            (740 - 665)
            / (842 - 665)
        )
    )

    cyano = np.maximum(
        mci,
        fai
    )

    clases_scl_no_validas = {
        1,
        3,
        8,
        9,
        10,
        11
    }

    mascara_calidad = ~np.isin(
        scl.astype("int16"),
        list(clases_scl_no_validas)
    )

    mascara_agua = (
        ndwi > 0
    )

    mascara_valida = (
        np.isfinite(b03)
        & np.isfinite(b04)
        & np.isfinite(b08)
        & (b03 > 0)
        & (b04 > 0)
        & (b08 > 0)
        & mascara_calidad
        & mascara_agua
    )

    ndvi_mask = np.where(
        mascara_valida,
        ndvi,
        np.nan
    ).astype("float32")

    ndwi_mask = np.where(
        mascara_valida,
        ndwi,
        np.nan
    ).astype("float32")

    cyano_mask = np.where(
        mascara_valida,
        cyano,
        np.nan
    ).astype("float32")

    return {
        "ndvi": ndvi_mask,
        "ndwi": ndwi_mask,
        "cyano": cyano_mask,
        "mascara_valida": mascara_valida,
        "bandas": valores,
        "transform": transform,
        "crs": crs,
        "perfil": perfil,
        "nodata": nodata
    }


# aca se muestra y guarda un mapa
def mostrar_mapa(
    matriz,
    titulo,
    nombre_archivo,
    etiqueta="Valor del índice",
    percentiles=(2, 98)
):
    valores = matriz[
        np.isfinite(matriz)
    ]

    if valores.size == 0:
        print(
            f"No hay valores válidos para: "
            f"{titulo}"
        )
        return

    vmin, vmax = np.nanpercentile(
        valores,
        percentiles
    )

    if np.isclose(vmin, vmax):
        vmin = np.nanmin(valores)
        vmax = np.nanmax(valores)

    plt.figure(
        figsize=(8, 6)
    )

    imagen = plt.imshow(
        matriz,
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(
        imagen,
        label=etiqueta
    )

    plt.title(titulo)
    plt.axis("off")
    plt.tight_layout()

    salida = (
        GRAFICOS_DIR
        / nombre_archivo
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        "Gráfica guardada en:",
        salida
    )


# aca se grafica la serie temporal de un lago
def grafica_temporal_lago(
    df,
    lago
):
    datos = (
        df[
            df["lago"] == lago
        ]
        .dropna(
            subset=["cyano_promedio"]
        )
        .sort_values("fecha")
        .copy()
    )

    if datos.empty:
        print(
            f"No hay datos para {lago}."
        )
        return

    pico = datos.loc[
        datos[
            "cyano_promedio"
        ].idxmax()
    ]

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        datos["fecha"],
        datos["cyano_promedio"],
        marker="o"
    )

    plt.scatter(
        [pico["fecha"]],
        [pico["cyano_promedio"]],
        s=90
    )

    plt.annotate(
        f"Pico: {pico['fecha'].date()}",
        (
            pico["fecha"],
            pico["cyano_promedio"]
        ),
        xytext=(8, 10),
        textcoords="offset points"
    )

    plt.title(
        f"Evolución temporal del índice "
        f"de cianobacteria - {lago}"
    )

    plt.xlabel("Fecha")
    plt.ylabel(
        "Índice promedio de cianobacteria"
    )

    plt.grid(alpha=0.25)
    plt.xticks(rotation=45)
    plt.tight_layout()

    salida = (
        GRAFICOS_DIR
        / (
            f"serie_temporal_"
            f"{nombre_seguro(lago)}.png"
        )
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# aca se comparan los dos lagos en el tiempo
def grafica_comparacion_lagos(df):
    plt.figure(
        figsize=(10, 5)
    )

    for lago in [
        "Atitlán",
        "Amatitlán"
    ]:
        datos = (
            df[
                df["lago"] == lago
            ]
            .dropna(
                subset=["cyano_promedio"]
            )
            .sort_values("fecha")
        )

        if datos.empty:
            continue

        plt.plot(
            datos["fecha"],
            datos["cyano_promedio"],
            marker="o",
            label=lago
        )

    plt.title(
        "Comparación temporal del "
        "índice de cianobacteria"
    )

    plt.xlabel("Fecha")
    plt.ylabel(
        "Índice promedio de cianobacteria"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.xticks(rotation=45)
    plt.tight_layout()

    salida = (
        GRAFICOS_DIR
        / "comparacion_temporal_lagos.png"
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# aca se crea el resumen temporal
def analizar_temporal(df):
    textos = []

    for lago in [
        "Atitlán",
        "Amatitlán"
    ]:
        datos = (
            df[
                df["lago"] == lago
            ]
            .dropna(
                subset=["cyano_promedio"]
            )
            .sort_values("fecha")
            .copy()
        )

        if len(datos) < 2:
            textos.append(
                f"{lago}: no hay suficientes "
                f"datos para analizar la tendencia."
            )
            continue

        pico = datos.loc[
            datos[
                "cyano_promedio"
            ].idxmax()
        ]

        minimo = datos.loc[
            datos[
                "cyano_promedio"
            ].idxmin()
        ]

        texto = (
            f"{lago}: el valor promedio más alto "
            f"aparece el {pico['fecha'].date()} "
            f"con {pico['cyano_promedio']:.5f}. "
            f"El valor más bajo aparece el "
            f"{minimo['fecha'].date()} "
            f"con {minimo['cyano_promedio']:.5f}. "
            f"El porcentaje de agua con valores altos "
            f"en la fecha del pico fue de "
            f"{pico['porcentaje_cyano_alto']:.2f}%."
        )

        textos.append(texto)

    return textos


# aca se crea el mapa de la fecha pico
def mapa_fecha_pico(
    df,
    lago
):
    datos = (
        df[
            df["lago"] == lago
        ]
        .dropna(
            subset=["cyano_promedio"]
        )
    )

    if datos.empty:
        print(
            f"No hay datos para {lago}."
        )
        return

    fila = datos.loc[
        datos[
            "cyano_promedio"
        ].idxmax()
    ]

    indices = leer_y_calcular_indices(
        fila["archivo"]
    )

    mostrar_mapa(
        indices["cyano"],
        (
            f"Mayor promedio de cianobacteria - "
            f"{lago} - "
            f"{fila['fecha'].date()}"
        ),
        (
            f"mapa_pico_cyano_"
            f"{nombre_seguro(lago)}.png"
        ),
        etiqueta="Bloom Index"
    )


# aca se comparan la fecha minima y la fecha pico
def mapa_comparativo_fechas(
    df,
    lago
):
    datos = (
        df[
            df["lago"] == lago
        ]
        .dropna(
            subset=["cyano_promedio"]
        )
        .copy()
    )

    fila_min = datos.loc[
        datos[
            "cyano_promedio"
        ].idxmin()
    ]

    fila_max = datos.loc[
        datos[
            "cyano_promedio"
        ].idxmax()
    ]

    indice_min = (
        leer_y_calcular_indices(
            fila_min["archivo"]
        )["cyano"]
    )

    indice_max = (
        leer_y_calcular_indices(
            fila_max["archivo"]
        )["cyano"]
    )

    valores = np.concatenate([
        indice_min[
            np.isfinite(indice_min)
        ],
        indice_max[
            np.isfinite(indice_max)
        ]
    ])

    vmin, vmax = np.nanpercentile(
        valores,
        [2, 98]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True
    )

    imagen = axes[0].imshow(
        indice_min,
        vmin=vmin,
        vmax=vmax
    )

    axes[0].set_title(
        f"{fila_min['fecha'].date()} "
        f"- menor promedio"
    )
    axes[0].axis("off")

    axes[1].imshow(
        indice_max,
        vmin=vmin,
        vmax=vmax
    )

    axes[1].set_title(
        f"{fila_max['fecha'].date()} "
        f"- mayor promedio"
    )
    axes[1].axis("off")

    fig.colorbar(
        imagen,
        ax=axes.ravel().tolist(),
        label="Bloom Index",
        shrink=0.8
    )

    fig.suptitle(
        f"Comparación espacial de "
        f"cianobacteria - {lago}"
    )

    salida = (
        GRAFICOS_DIR
        / (
            f"comparacion_espacial_"
            f"{nombre_seguro(lago)}.png"
        )
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return fila_min, fila_max


# aca se revisan zonas con valores altos repetidos
def calcular_persistencia(
    df,
    lago,
    umbral=0.05
):
    datos = (
        df[
            df["lago"] == lago
        ]
        .sort_values("fecha")
    )

    conteo_alto = None
    conteo_valido = None

    for _, fila in datos.iterrows():
        indices = leer_y_calcular_indices(
            fila["archivo"]
        )

        cyano = indices["cyano"]

        valido = np.isfinite(cyano)

        alto = (
            valido
            & (cyano > umbral)
        )

        if conteo_alto is None:
            conteo_alto = np.zeros_like(
                cyano,
                dtype="int16"
            )

            conteo_valido = np.zeros_like(
                cyano,
                dtype="int16"
            )

        conteo_alto += alto.astype(
            "int16"
        )

        conteo_valido += valido.astype(
            "int16"
        )

    persistencia = np.full_like(
        conteo_alto,
        np.nan,
        dtype="float32"
    )

    suficiente = (
        conteo_valido >= 6
    )

    persistencia[suficiente] = (
        conteo_alto[suficiente]
        / conteo_valido[suficiente]
        * 100
    )

    persistente = (
        suficiente
        & (conteo_alto >= 3)
    )

    alto, ancho = persistente.shape

    zonas = {
        "noroeste": int(
            persistente[
                :alto // 2,
                :ancho // 2
            ].sum()
        ),
        "noreste": int(
            persistente[
                :alto // 2,
                ancho // 2:
            ].sum()
        ),
        "suroeste": int(
            persistente[
                alto // 2:,
                :ancho // 2
            ].sum()
        ),
        "sureste": int(
            persistente[
                alto // 2:,
                ancho // 2:
            ].sum()
        )
    }

    zona_principal = max(
        zonas,
        key=zonas.get
    )

    plt.figure(
        figsize=(8, 6)
    )

    imagen = plt.imshow(
        persistencia,
        vmin=0,
        vmax=100
    )

    plt.colorbar(
        imagen,
        label=(
            "Porcentaje de fechas "
            "con valores altos"
        )
    )

    plt.title(
        f"Persistencia de valores "
        f"altos - {lago}"
    )

    plt.axis("off")
    plt.tight_layout()

    salida = (
        GRAFICOS_DIR
        / (
            f"persistencia_cyano_"
            f"{nombre_seguro(lago)}.png"
        )
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return {
        "lago": lago,
        "pixeles_persistentes": int(
            persistente.sum()
        ),
        "zona_principal":
            zona_principal,
        "max_persistencia": float(
            np.nanmax(persistencia)
        )
    }


# aca se toman pixeles para calcular correlaciones
def obtener_muestra_correlacion(
    df,
    lago,
    muestra_por_fecha=15000
):
    rng = np.random.default_rng(42)

    muestras = []

    datos = (
        df[
            df["lago"] == lago
        ]
        .sort_values("fecha")
    )

    for _, fila in datos.iterrows():
        indices = leer_y_calcular_indices(
            fila["archivo"]
        )

        cyano = indices["cyano"]
        ndvi = indices["ndvi"]
        ndwi = indices["ndwi"]

        valido = (
            np.isfinite(cyano)
            & np.isfinite(ndvi)
            & np.isfinite(ndwi)
        )

        posiciones = np.flatnonzero(
            valido
        )

        if (
            len(posiciones)
            > muestra_por_fecha
        ):
            posiciones = rng.choice(
                posiciones,
                size=muestra_por_fecha,
                replace=False
            )

        muestras.append(
            pd.DataFrame({
                "cyano":
                    cyano.ravel()[
                        posiciones
                    ],
                "ndvi":
                    ndvi.ravel()[
                        posiciones
                    ],
                "ndwi":
                    ndwi.ravel()[
                        posiciones
                    ]
            })
        )

    return pd.concat(
        muestras,
        ignore_index=True
    )


# aca se calcula y grafica la correlacion
def analizar_correlacion(
    df,
    lago
):
    muestra = obtener_muestra_correlacion(
        df,
        lago
    )

    correlaciones = muestra[
        [
            "cyano",
            "ndvi",
            "ndwi"
        ]
    ].corr()

    corr_ndvi = correlaciones.loc[
        "cyano",
        "ndvi"
    ]

    corr_ndwi = correlaciones.loc[
        "cyano",
        "ndwi"
    ]

    grafica = muestra.sample(
        min(
            12000,
            len(muestra)
        ),
        random_state=42
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].scatter(
        grafica["ndvi"],
        grafica["cyano"],
        s=5,
        alpha=0.25
    )

    axes[0].set_title(
        f"NDVI y cianobacteria "
        f"- {lago}"
    )

    axes[0].set_xlabel("NDVI")
    axes[0].set_ylabel(
        "Bloom Index"
    )

    axes[1].scatter(
        grafica["ndwi"],
        grafica["cyano"],
        s=5,
        alpha=0.25
    )

    axes[1].set_title(
        f"NDWI y cianobacteria "
        f"- {lago}"
    )

    axes[1].set_xlabel("NDWI")
    axes[1].set_ylabel(
        "Bloom Index"
    )

    plt.tight_layout()

    salida = (
        GRAFICOS_DIR
        / (
            f"correlacion_indices_"
            f"{nombre_seguro(lago)}.png"
        )
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return {
        "lago": lago,
        "corr_cyano_ndvi":
            corr_ndvi,
        "corr_cyano_ndwi":
            corr_ndwi
    }


# aca se crea la tabla de comparacion general
def crear_comparacion_lagos(df):
    comparacion = []

    for lago in [
        "Atitlán",
        "Amatitlán"
    ]:
        datos = (
            df[
                df["lago"] == lago
            ]
            .copy()
        )

        fila_pico = datos.loc[
            datos[
                "cyano_promedio"
            ].idxmax()
        ]

        comparacion.append({
            "lago":
                lago,
            "cyano_promedio_periodo":
                datos[
                    "cyano_promedio"
                ].mean(),
            "cyano_maximo":
                fila_pico[
                    "cyano_promedio"
                ],
            "fecha_pico":
                fila_pico[
                    "fecha"
                ].date(),
            "area_alta_promedio":
                datos[
                    "porcentaje_cyano_alto"
                ].mean(),
            "area_alta_maxima":
                datos[
                    "porcentaje_cyano_alto"
                ].max(),
            "fechas_area_mayor_1":
                int(
                    (
                        datos[
                            "porcentaje_cyano_alto"
                        ] > 1
                    ).sum()
                ),
            "fechas_area_mayor_5":
                int(
                    (
                        datos[
                            "porcentaje_cyano_alto"
                        ] > 5
                    ).sum()
                )
        })

    tabla = pd.DataFrame(
        comparacion
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5)
    )

    axes[0].bar(
        tabla["lago"],
        tabla[
            "cyano_promedio_periodo"
        ]
    )

    axes[0].set_title(
        "Promedio de cianobacteria "
        "por lago"
    )

    axes[0].set_ylabel(
        "Bloom Index"
    )

    axes[1].bar(
        tabla["lago"],
        tabla[
            "area_alta_promedio"
        ]
    )

    axes[1].set_title(
        "Área promedio con "
        "valores altos"
    )

    axes[1].set_ylabel(
        "Porcentaje"
    )

    plt.tight_layout()

    salida = (
        GRAFICOS_DIR
        / "comparacion_general_lagos.png"
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return tabla


# aca se grafica la extension de valores altos
def grafica_extension_floracion(df):
    plt.figure(
        figsize=(10, 5)
    )

    for lago in [
        "Atitlán",
        "Amatitlán"
    ]:
        datos = (
            df[
                df["lago"] == lago
            ]
            .sort_values("fecha")
        )

        plt.plot(
            datos["fecha"],
            datos[
                "porcentaje_cyano_alto"
            ],
            marker="o",
            label=lago
        )

    plt.title(
        "Porcentaje del lago con "
        "valores altos de cianobacteria"
    )

    plt.xlabel("Fecha")
    plt.ylabel(
        "Porcentaje de píxeles válidos"
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.xticks(rotation=45)
    plt.tight_layout()

    salida = (
        GRAFICOS_DIR
        / "extension_floracion_lagos.png"
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# aca se comparan las distribuciones por fecha
def boxplot_fechas(
    df,
    lago,
    muestra_por_fecha=4000
):
    rng = np.random.default_rng(42)

    datos = (
        df[
            df["lago"] == lago
        ]
        .sort_values("fecha")
    )

    valores_box = []
    etiquetas = []

    for _, fila in datos.iterrows():
        cyano = (
            leer_y_calcular_indices(
                fila["archivo"]
            )["cyano"]
        )

        valores = cyano[
            np.isfinite(cyano)
        ]

        if (
            len(valores)
            > muestra_por_fecha
        ):
            valores = rng.choice(
                valores,
                size=muestra_por_fecha,
                replace=False
            )

        valores_box.append(
            valores
        )

        etiquetas.append(
            fila[
                "fecha"
            ].strftime(
                "%Y-%m-%d"
            )
        )

    plt.figure(
        figsize=(12, 6)
    )

    plt.boxplot(
        valores_box,
        tick_labels=etiquetas,
        showfliers=False
    )

    plt.title(
        f"Distribución del índice "
        f"de cianobacteria - {lago}"
    )

    plt.xlabel("Fecha")
    plt.ylabel(
        "Bloom Index"
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    salida = (
        GRAFICOS_DIR
        / (
            f"boxplot_cyano_"
            f"{nombre_seguro(lago)}.png"
        )
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# aca se calcula el cambio entre la fecha minima y la fecha pico
def mapa_diferencia(
    df,
    lago
):
    datos = (
        df[
            df["lago"] == lago
        ]
        .copy()
    )

    fila_min = datos.loc[
        datos[
            "cyano_promedio"
        ].idxmin()
    ]

    fila_max = datos.loc[
        datos[
            "cyano_promedio"
        ].idxmax()
    ]

    cyano_min = (
        leer_y_calcular_indices(
            fila_min["archivo"]
        )["cyano"]
    )

    cyano_max = (
        leer_y_calcular_indices(
            fila_max["archivo"]
        )["cyano"]
    )

    diferencia = (
        cyano_max
        - cyano_min
    )

    valores = diferencia[
        np.isfinite(diferencia)
    ]

    limite = np.nanpercentile(
        np.abs(valores),
        98
    )

    plt.figure(
        figsize=(8, 6)
    )

    imagen = plt.imshow(
        diferencia,
        vmin=-limite,
        vmax=limite,
        cmap="coolwarm"
    )

    plt.colorbar(
        imagen,
        label=(
            "Cambio del Bloom Index"
        )
    )

    plt.title(
        f"Cambio entre menor y "
        f"mayor promedio - {lago}"
    )

    plt.axis("off")
    plt.tight_layout()

    salida = (
        GRAFICOS_DIR
        / (
            f"diferencia_cyano_"
            f"{nombre_seguro(lago)}.png"
        )
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    ambos = (
        np.isfinite(cyano_min)
        & np.isfinite(cyano_max)
    )

    return {
        "lago":
            lago,
        "fecha_menor":
            fila_min[
                "fecha"
            ].date(),
        "fecha_mayor":
            fila_max[
                "fecha"
            ].date(),
        "porcentaje_aumento":
            float(
                np.mean(
                    diferencia[
                        ambos
                    ] > 0
                )
                * 100
            )
    }


# aca se revisa un posible patron estacional
def analizar_estacionalidad(df):
    datos = df.copy()

    datos["mes"] = (
        datos[
            "fecha"
        ].dt.month
    )

    datos["temporada"] = np.where(
        datos[
            "mes"
        ].isin(
            [
                5,
                6,
                7,
                8,
                9,
                10
            ]
        ),
        "Lluviosa",
        "Seca"
    )

    resumen = (
        datos
        .groupby(
            [
                "lago",
                "temporada"
            ]
        )
        .agg(
            cyano_promedio=(
                "cyano_promedio",
                "mean"
            ),
            area_alta_promedio=(
                "porcentaje_cyano_alto",
                "mean"
            ),
            fechas=(
                "fecha",
                "count"
            )
        )
        .reset_index()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5)
    )

    for lago in [
        "Atitlán",
        "Amatitlán"
    ]:
        parte = resumen[
            resumen["lago"] == lago
        ]

        axes[0].bar(
            [
                f"{lago}\n{x}"
                for x
                in parte["temporada"]
            ],
            parte["cyano_promedio"]
        )

        axes[1].bar(
            [
                f"{lago}\n{x}"
                for x
                in parte["temporada"]
            ],
            parte[
                "area_alta_promedio"
            ]
        )

    axes[0].set_title(
        "Promedio de cianobacteria "
        "por temporada"
    )

    axes[0].set_ylabel(
        "Bloom Index"
    )

    axes[1].set_title(
        "Área alta promedio "
        "por temporada"
    )

    axes[1].set_ylabel(
        "Porcentaje"
    )

    plt.tight_layout()

    salida = (
        GRAFICOS_DIR
        / "comparacion_estacional.png"
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return resumen

# aca se convierten las coordenadas del raster a longitud y latitud
def obtener_coordenadas_geograficas(
    transform,
    crs,
    filas,
    columnas
):
    xs, ys = rasterio.transform.xy(
        transform,
        filas,
        columnas,
        offset="center"
    )

    xs = np.asarray(xs, dtype="float64")
    ys = np.asarray(ys, dtype="float64")

    if crs is not None:
        crs_texto = str(crs).upper()

        if "4326" not in crs_texto:
            from rasterio.warp import transform as transformar_coordenadas

            xs, ys = transformar_coordenadas(
                crs,
                "EPSG:4326",
                xs.tolist(),
                ys.tolist()
            )

            xs = np.asarray(xs, dtype="float64")
            ys = np.asarray(ys, dtype="float64")

    return xs, ys


# aca se toma una muestra que conserva la proporcion de las clases
def seleccionar_muestra_estratificada(
    posiciones,
    objetivo,
    max_observaciones,
    semilla
):
    if (
        max_observaciones is None
        or len(posiciones) <= max_observaciones
    ):
        return posiciones

    rng = np.random.default_rng(semilla)

    objetivo = np.asarray(
        objetivo,
        dtype="int8"
    )

    pos_alta = posiciones[
        objetivo == 1
    ]

    pos_baja = posiciones[
        objetivo == 0
    ]

    proporcion_alta = (
        len(pos_alta)
        / len(posiciones)
    )

    n_alta = int(
        round(
            max_observaciones
            * proporcion_alta
        )
    )

    n_alta = min(
        n_alta,
        len(pos_alta)
    )

    n_baja = (
        max_observaciones
        - n_alta
    )

    if n_baja > len(pos_baja):
        faltantes = n_baja - len(pos_baja)
        n_baja = len(pos_baja)
        n_alta = min(
            len(pos_alta),
            n_alta + faltantes
        )

    muestra_alta = rng.choice(
        pos_alta,
        size=n_alta,
        replace=False
    ) if n_alta > 0 else np.array([], dtype=posiciones.dtype)

    muestra_baja = rng.choice(
        pos_baja,
        size=n_baja,
        replace=False
    ) if n_baja > 0 else np.array([], dtype=posiciones.dtype)

    muestra = np.concatenate([
        muestra_alta,
        muestra_baja
    ])

    rng.shuffle(muestra)

    return muestra


# aca se construyen observaciones para una fecha
def construir_observaciones_ml_fecha(
    ruta_tif,
    lago,
    fecha,
    nubosidad_pct,
    umbral_cyano=0.05,
    max_observaciones=30000,
    random_state=42
):
    indices = leer_y_calcular_indices(
        ruta_tif
    )

    valores = indices["bandas"]

    mascara = (
        indices["mascara_valida"]
        & np.isfinite(indices["ndvi"])
        & np.isfinite(indices["ndwi"])
        & np.isfinite(indices["cyano"])
    )

    for banda in [
        "B03",
        "B04",
        "B05",
        "B06",
        "B08"
    ]:
        mascara &= np.isfinite(
            valores[banda]
        )

    posiciones = np.flatnonzero(
        mascara
    )

    if len(posiciones) == 0:
        return pd.DataFrame()

    objetivo_completo = (
        indices["cyano"].ravel()[
            posiciones
        ] > umbral_cyano
    ).astype("int8")

    semilla_fecha = (
        random_state
        + int(
            pd.Timestamp(fecha).strftime(
                "%Y%m%d"
            )
        )
        + sum(ord(letra) for letra in lago)
    )

    posiciones = seleccionar_muestra_estratificada(
        posiciones,
        objetivo_completo,
        max_observaciones,
        semilla_fecha
    )

    filas, columnas = np.unravel_index(
        posiciones,
        mascara.shape
    )

    longitud, latitud = obtener_coordenadas_geograficas(
        indices["transform"],
        indices["crs"],
        filas,
        columnas
    )

    fecha_dt = pd.Timestamp(fecha)

    datos = pd.DataFrame({
        "longitud": longitud,
        "latitud": latitud,
        "fecha": fecha_dt,
        "lago": lago,
        "B03": valores["B03"].ravel()[posiciones],
        "B04": valores["B04"].ravel()[posiciones],
        "B05": valores["B05"].ravel()[posiciones],
        "B06": valores["B06"].ravel()[posiciones],
        "B08": valores["B08"].ravel()[posiciones],
        "SCL": valores["SCL"].ravel()[posiciones].astype("int16"),
        "ndvi": indices["ndvi"].ravel()[posiciones],
        "ndwi": indices["ndwi"].ravel()[posiciones],
        "cyano": indices["cyano"].ravel()[posiciones],
        "cyano_alta": (
            indices["cyano"].ravel()[posiciones]
            > umbral_cyano
        ).astype("int8"),
        "nubosidad_pct": float(nubosidad_pct)
    })

    datos["mes"] = datos["fecha"].dt.month.astype("int8")
    datos["dia_anio"] = datos["fecha"].dt.dayofyear.astype("int16")

    datos["mes_sin"] = np.sin(
        2 * np.pi * datos["mes"] / 12
    ).astype("float32")

    datos["mes_cos"] = np.cos(
        2 * np.pi * datos["mes"] / 12
    ).astype("float32")

    datos["dia_anio_sin"] = np.sin(
        2 * np.pi * datos["dia_anio"] / 366
    ).astype("float32")

    datos["dia_anio_cos"] = np.cos(
        2 * np.pi * datos["dia_anio"] / 366
    ).astype("float32")

    return datos


# aca se construye el conjunto de datos de todos los lagos y fechas
def construir_dataset_ml(
    lagos,
    nubosidad_oficial,
    salida_csv,
    umbral_cyano=0.05,
    max_observaciones_fecha=30000,
    random_state=42,
    reutilizar=True
):
    salida_csv = Path(
        salida_csv
    )

    if reutilizar and salida_csv.exists():
        print(
            "Se reutiliza el conjunto de datos:",
            salida_csv
        )

        datos = pd.read_csv(
            salida_csv,
            parse_dates=["fecha"]
        )

        return datos

    nubosidad = pd.DataFrame(
        nubosidad_oficial,
        columns=[
            "lago",
            "fecha",
            "nubosidad_pct",
            "satelite"
        ]
    )

    nubosidad["fecha"] = pd.to_datetime(
        nubosidad["fecha"]
    )

    partes = []

    for lago, info in lagos.items():
        for fecha in info["fechas"]:
            ruta = (
                DATOS_DIR
                / f"{nombre_seguro(lago)}_{fecha}.tif"
            )

            if not ruta.exists():
                raise FileNotFoundError(
                    f"No se encontró el raster de {lago} para {fecha}: {ruta}"
                )

            fila_nubosidad = nubosidad[
                (nubosidad["lago"] == lago)
                & (
                    nubosidad["fecha"]
                    == pd.Timestamp(fecha)
                )
            ]

            nubosidad_pct = (
                float(
                    fila_nubosidad[
                        "nubosidad_pct"
                    ].iloc[0]
                )
                if not fila_nubosidad.empty
                else np.nan
            )

            parte = construir_observaciones_ml_fecha(
                ruta_tif=ruta,
                lago=lago,
                fecha=fecha,
                nubosidad_pct=nubosidad_pct,
                umbral_cyano=umbral_cyano,
                max_observaciones=max_observaciones_fecha,
                random_state=random_state
            )

            partes.append(
                parte
            )

            print(
                f"{lago} - {fecha}: "
                f"{len(parte):,} observaciones"
            )

    datos = pd.concat(
        partes,
        ignore_index=True
    )

    salida_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    datos.to_csv(
        salida_csv,
        index=False,
        compression="gzip"
    )

    print(
        "Conjunto de datos guardado en:",
        salida_csv
    )

    return datos


# aca se resume el conjunto de datos para el avance
def resumir_dataset_ml(datos):
    total = len(datos)

    por_lago = (
        datos
        .groupby("lago")
        .size()
        .rename("observaciones")
        .reset_index()
    )

    por_fecha = (
        datos
        .groupby([
            "lago",
            "fecha"
        ])
        .size()
        .rename("observaciones")
        .reset_index()
    )

    faltantes = pd.DataFrame({
        "variable": datos.columns,
        "tipo": [
            str(datos[columna].dtype)
            for columna in datos.columns
        ],
        "faltantes_pct": [
            datos[columna].isna().mean() * 100
            for columna in datos.columns
        ]
    })

    return {
        "total": total,
        "por_lago": por_lago,
        "por_fecha": por_fecha,
        "faltantes": faltantes
    }


# aca se grafica la distribucion de la variable respuesta
def graficar_distribucion_objetivo(datos):
    conteo = (
        datos["cyano_alta"]
        .value_counts()
        .sort_index()
    )

    plt.figure(
        figsize=(7, 4)
    )

    plt.bar(
        ["Baja o ausente", "Alta"],
        [
            conteo.get(0, 0),
            conteo.get(1, 0)
        ]
    )

    plt.title(
        "Distribución global de la variable respuesta"
    )
    plt.ylabel("Observaciones")
    plt.tight_layout()

    salida = (
        GRAFICOS_DIR
        / "distribucion_objetivo_global.png"
    )

    plt.savefig(
        salida,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# aca se hace un analisis exploratorio de las variables principales
def graficar_eda_ml(datos):
    muestra = datos.sample(
        min(80000, len(datos)),
        random_state=42
    )

    variables = [
        "B03",
        "ndvi",
        "ndwi",
        "cyano"
    ]

    for variable in variables:
        plt.figure(
            figsize=(7, 4)
        )

        plt.hist(
            muestra[variable],
            bins=60
        )

        plt.title(
            f"Distribución de {variable}"
        )
        plt.xlabel(variable)
        plt.ylabel("Frecuencia")
        plt.tight_layout()

        salida = (
            GRAFICOS_DIR
            / f"eda_{variable.lower()}.png"
        )

        plt.savefig(
            salida,
            dpi=180,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

    for lago in sorted(
        datos["lago"].unique()
    ):
        parte = datos[
            datos["lago"] == lago
        ]

        parte = parte.sample(
            min(25000, len(parte)),
            random_state=42
        )

        plt.figure(
            figsize=(7, 6)
        )

        imagen = plt.scatter(
            parte["longitud"],
            parte["latitud"],
            c=parte["cyano_alta"],
            s=4,
            alpha=0.45
        )

        plt.colorbar(
            imagen,
            label="Clase de cianobacteria"
        )
        plt.title(
            f"Observaciones geográficas - {lago}"
        )
        plt.xlabel("Longitud")
        plt.ylabel("Latitud")
        plt.tight_layout()

        salida = (
            GRAFICOS_DIR
            / (
                f"eda_mapa_observaciones_"
                f"{nombre_seguro(lago)}.png"
            )
        )

        plt.savefig(
            salida,
            dpi=180,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()


# aca se calculan las metricas de clasificacion
def calcular_metricas_modelo(
    y_real,
    probabilidades,
    umbral=0.5,
    modelo=None,
    validacion=None
):
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score
    )

    y_real = np.asarray(y_real, dtype="int8")
    probabilidades = np.asarray(probabilidades, dtype="float64")
    prediccion = (probabilidades >= umbral).astype("int8")

    tn, fp, fn, tp = confusion_matrix(
        y_real,
        prediccion,
        labels=[0, 1]
    ).ravel()

    salida = {
        "Accuracy": accuracy_score(y_real, prediccion),
        "Precision": precision_score(
            y_real,
            prediccion,
            zero_division=0
        ),
        "Recall": recall_score(
            y_real,
            prediccion,
            zero_division=0
        ),
        "F1": f1_score(
            y_real,
            prediccion,
            zero_division=0
        ),
        "F2": fbeta_score(
            y_real,
            prediccion,
            beta=2,
            zero_division=0
        ),
        "ROC_AUC": (
            roc_auc_score(y_real, probabilidades)
            if np.unique(y_real).size == 2
            else np.nan
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "umbral": float(umbral)
    }

    if modelo is not None:
        salida["modelo"] = modelo

    if validacion is not None:
        salida["validacion"] = validacion

    return salida


# aca se selecciona un umbral que da mas peso al recall
def seleccionar_umbral_f2(y_real, probabilidades):
    from sklearn.metrics import precision_recall_curve

    precision, recall, umbrales = precision_recall_curve(
        y_real,
        probabilidades
    )

    if len(umbrales) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    f2 = (
        5 * precision * recall
        / (4 * precision + recall + 1e-12)
    )

    return float(
        umbrales[int(np.nanargmax(f2))]
    )


# aca se construyen los modelos y sus valores de ajuste
def crear_candidatos_modelos(random_state=42):
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Regresion logistica": [
            (
                {"C": valor_c},
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=valor_c,
                        class_weight="balanced",
                        max_iter=500,
                        random_state=random_state
                    )
                )
            )
            for valor_c in [0.1, 1.0, 10.0]
        ],
        "Random Forest": [
            (
                {
                    "max_depth": profundidad,
                    "min_samples_leaf": 5
                },
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=profundidad,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    max_samples=0.7,
                    n_jobs=-1,
                    random_state=random_state
                )
            )
            for profundidad in [12, 18]
        ],
        "Gradient Boosting": [
            (
                {
                    "learning_rate": tasa,
                    "max_leaf_nodes": hojas
                },
                HistGradientBoostingClassifier(
                    learning_rate=tasa,
                    max_leaf_nodes=hojas,
                    max_iter=180,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    early_stopping=True,
                    random_state=random_state
                )
            )
            for tasa, hojas in [
                (0.05, 31),
                (0.08, 31),
                (0.08, 63)
            ]
        ]
    }


# aca se ajustan hiperparametros sin usar el conjunto de prueba
def entrenar_modelos_aleatorios(
    datos,
    predictores,
    objetivo="cyano_alta",
    random_state=42,
    max_ajuste=180000
):
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(datos))

    indice_entreno, indice_prueba = train_test_split(
        indices,
        test_size=0.30,
        stratify=datos[objetivo],
        random_state=random_state
    )

    indice_ajuste, indice_validacion = train_test_split(
        indice_entreno,
        test_size=0.20,
        stratify=datos.iloc[indice_entreno][objetivo],
        random_state=random_state
    )

    if len(indice_ajuste) > max_ajuste:
        indice_ajuste, _ = train_test_split(
            indice_ajuste,
            train_size=max_ajuste,
            stratify=datos.iloc[indice_ajuste][objetivo],
            random_state=random_state
        )

    X_ajuste = datos.iloc[indice_ajuste][predictores]
    y_ajuste = datos.iloc[indice_ajuste][objetivo]
    X_validacion = datos.iloc[indice_validacion][predictores]
    y_validacion = datos.iloc[indice_validacion][objetivo]

    mejores_modelos = {}
    umbrales = {}
    filas_ajuste = []

    for nombre, candidatos in crear_candidatos_modelos(
        random_state
    ).items():
        mejor_f2 = -np.inf
        mejor_modelo = None
        mejor_umbral = 0.5
        mejores_parametros = None

        for parametros, candidato in candidatos:
            modelo = clone(candidato)
            modelo.fit(X_ajuste, y_ajuste)

            probabilidades = modelo.predict_proba(
                X_validacion
            )[:, 1]

            umbral = seleccionar_umbral_f2(
                y_validacion,
                probabilidades
            )

            metricas = calcular_metricas_modelo(
                y_validacion,
                probabilidades,
                umbral=umbral,
                modelo=nombre,
                validacion="Ajuste interno"
            )

            filas_ajuste.append({
                **metricas,
                "parametros": str(parametros)
            })

            if metricas["F2"] > mejor_f2:
                mejor_f2 = metricas["F2"]
                mejor_modelo = clone(candidato)
                mejor_umbral = umbral
                mejores_parametros = parametros

        mejor_modelo.fit(
            datos.iloc[indice_entreno][predictores],
            datos.iloc[indice_entreno][objetivo]
        )

        mejores_modelos[nombre] = mejor_modelo
        umbrales[nombre] = mejor_umbral

        print(
            f"{nombre}: {mejores_parametros}, "
            f"umbral={mejor_umbral:.4f}"
        )

    filas_prueba = []
    predicciones = {}
    X_prueba = datos.iloc[indice_prueba][predictores]
    y_prueba = datos.iloc[indice_prueba][objetivo]

    for nombre, modelo in mejores_modelos.items():
        probabilidades = modelo.predict_proba(
            X_prueba
        )[:, 1]

        predicciones[nombre] = probabilidades
        filas_prueba.append(
            calcular_metricas_modelo(
                y_prueba,
                probabilidades,
                umbral=umbrales[nombre],
                modelo=nombre,
                validacion="Aleatoria 70/30"
            )
        )

    return {
        "modelos": mejores_modelos,
        "umbrales": umbrales,
        "ajuste": pd.DataFrame(filas_ajuste),
        "metricas": pd.DataFrame(filas_prueba),
        "indice_entreno": indice_entreno,
        "indice_prueba": indice_prueba,
        "predicciones_prueba": predicciones
    }


# aca se crean bloques de un kilometro en UTM 15N
def crear_bloques_espaciales(datos, tamano_metros=1000):
    from pyproj import Transformer

    salida = datos.copy()

    transformador = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32615",
        always_xy=True
    )

    x_utm, y_utm = transformador.transform(
        salida["longitud"].to_numpy(),
        salida["latitud"].to_numpy()
    )

    salida["x_utm"] = x_utm
    salida["y_utm"] = y_utm
    salida["bloque_x"] = np.floor(
        x_utm / tamano_metros
    ).astype("int32")
    salida["bloque_y"] = np.floor(
        y_utm / tamano_metros
    ).astype("int32")

    salida["bloque_espacial"] = (
        salida["lago"].astype(str)
        + "_"
        + salida["bloque_x"].astype(str)
        + "_"
        + salida["bloque_y"].astype(str)
    )

    return salida


# aca se resume y grafica la cuadricula espacial
def resumir_y_graficar_bloques(datos_bloques):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    resumen = (
        datos_bloques
        .groupby(["lago", "bloque_espacial"])
        .size()
        .rename("observaciones")
        .reset_index()
    )

    estadisticas = (
        resumen
        .groupby("lago")["observaciones"]
        .agg(
            bloques="count",
            minimo="min",
            mediana="median",
            promedio="mean",
            maximo="max"
        )
        .reset_index()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5)
    )

    for eje, lago in zip(
        axes,
        sorted(datos_bloques["lago"].unique())
    ):
        parte = (
            datos_bloques[
                datos_bloques["lago"] == lago
            ]
            .drop_duplicates("bloque_espacial")
        )

        rectangulos = [
            Rectangle(
                (fila["bloque_x"] * 1000, fila["bloque_y"] * 1000),
                1000,
                1000
            )
            for _, fila in parte.iterrows()
        ]

        coleccion = PatchCollection(
            rectangulos,
            cmap="tab20",
            edgecolor="black",
            linewidth=0.25,
            alpha=0.75
        )
        coleccion.set_array(np.arange(len(rectangulos)))
        eje.add_collection(coleccion)
        eje.autoscale_view()

        eje.set_title(f"Bloques espaciales de 1 km - {lago}")
        eje.set_xlabel("Este UTM (m)")
        eje.set_ylabel("Norte UTM (m)")
        eje.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    salida = GRAFICOS_DIR / "bloques_espaciales_1km.png"
    plt.savefig(salida, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()

    return estadisticas, resumen


# aca se valida sin separar observaciones del mismo bloque
def validar_modelos_espacialmente(
    datos_bloques,
    predictores,
    modelos,
    umbrales,
    objetivo="cyano_alta",
    n_splits=5
):
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    X = datos_bloques[predictores]
    y = datos_bloques[objetivo].to_numpy()
    grupos = datos_bloques["bloque_espacial"]

    division = GroupKFold(n_splits=n_splits)
    filas = []

    for nombre, modelo_base in modelos.items():
        probabilidades = np.full(len(datos_bloques), np.nan)

        for numero, (entreno, validacion) in enumerate(
            division.split(X, y, grupos),
            start=1
        ):
            modelo = clone(modelo_base)
            modelo.fit(X.iloc[entreno], y[entreno])
            probabilidades[validacion] = modelo.predict_proba(
                X.iloc[validacion]
            )[:, 1]

            print(
                f"{nombre} - bloque espacial {numero}/{n_splits}"
            )

        filas.append(
            calcular_metricas_modelo(
                y,
                probabilidades,
                umbral=umbrales[nombre],
                modelo=nombre,
                validacion="Espacial GroupKFold"
            )
        )

    return pd.DataFrame(filas)


# aca se evalua con las ultimas tres fechas de cada lago
def validar_modelos_temporalmente(
    datos,
    predictores,
    modelos,
    umbrales,
    objetivo="cyano_alta",
    fechas_prueba_por_lago=3
):
    from sklearn.base import clone

    fechas_prueba = {}
    mascara_prueba = np.zeros(len(datos), dtype=bool)

    for lago, parte in datos.groupby("lago"):
        fechas = sorted(pd.to_datetime(parte["fecha"]).unique())
        seleccionadas = fechas[-fechas_prueba_por_lago:]
        fechas_prueba[lago] = [
            pd.Timestamp(fecha).date().isoformat()
            for fecha in seleccionadas
        ]
        mascara_prueba |= (
            (datos["lago"] == lago)
            & pd.to_datetime(datos["fecha"]).isin(seleccionadas)
        ).to_numpy()

    entreno = np.flatnonzero(~mascara_prueba)
    prueba = np.flatnonzero(mascara_prueba)
    filas = []

    for nombre, modelo_base in modelos.items():
        modelo = clone(modelo_base)
        modelo.fit(
            datos.iloc[entreno][predictores],
            datos.iloc[entreno][objetivo]
        )
        probabilidades = modelo.predict_proba(
            datos.iloc[prueba][predictores]
        )[:, 1]
        filas.append(
            calcular_metricas_modelo(
                datos.iloc[prueba][objetivo],
                probabilidades,
                umbral=umbrales[nombre],
                modelo=nombre,
                validacion="Temporal"
            )
        )

    return pd.DataFrame(filas), fechas_prueba


# aca se entrena en un lago y se evalua en el otro
def evaluar_generalizacion_lagos(
    datos,
    predictores,
    modelos,
    umbrales,
    objetivo="cyano_alta"
):
    from sklearn.base import clone

    lagos = sorted(datos["lago"].unique())
    filas = []

    for lago_entreno in lagos:
        lago_prueba = next(
            lago for lago in lagos
            if lago != lago_entreno
        )

        entreno = datos["lago"] == lago_entreno
        prueba = datos["lago"] == lago_prueba

        for nombre, modelo_base in modelos.items():
            modelo = clone(modelo_base)
            modelo.fit(
                datos.loc[entreno, predictores],
                datos.loc[entreno, objetivo]
            )
            probabilidades = modelo.predict_proba(
                datos.loc[prueba, predictores]
            )[:, 1]

            metricas = calcular_metricas_modelo(
                datos.loc[prueba, objetivo],
                probabilidades,
                umbral=umbrales[nombre],
                modelo=nombre,
                validacion="Entre lagos"
            )
            metricas["entrenamiento"] = lago_entreno
            metricas["prueba"] = lago_prueba
            filas.append(metricas)

    return pd.DataFrame(filas)


# aca se muestran las matrices de confusion de los tres modelos
def graficar_matrices_confusion(metricas, titulo, nombre_archivo):
    modelos = metricas["modelo"].tolist()
    fig, axes = plt.subplots(
        1,
        len(modelos),
        figsize=(5 * len(modelos), 4)
    )

    if len(modelos) == 1:
        axes = [axes]

    for eje, (_, fila) in zip(axes, metricas.iterrows()):
        matriz = np.array([
            [fila["TN"], fila["FP"]],
            [fila["FN"], fila["TP"]]
        ])
        imagen = eje.imshow(matriz, cmap="Blues")

        for i in range(2):
            for j in range(2):
                eje.text(
                    j,
                    i,
                    f"{int(matriz[i, j]):,}",
                    ha="center",
                    va="center"
                )

        eje.set_title(fila["modelo"])
        eje.set_xlabel("Prediccion")
        eje.set_ylabel("Valor real")
        eje.set_xticks([0, 1])
        eje.set_yticks([0, 1])

    fig.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(
        GRAFICOS_DIR / nombre_archivo,
        dpi=180,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()


# aca se calcula importancia global y un resumen SHAP
def explicar_modelo(
    modelo,
    X,
    y,
    predictores,
    nombre_modelo,
    random_state=42
):
    import shap
    from sklearn.inspection import permutation_importance

    muestra = pd.concat([
        X.loc[y == 1].sample(
            min(1200, int((y == 1).sum())),
            random_state=random_state
        ),
        X.loc[y == 0].sample(
            min(2400, int((y == 0).sum())),
            random_state=random_state
        )
    ]).sample(frac=1, random_state=random_state)

    y_muestra = y.loc[muestra.index]

    permutacion = permutation_importance(
        modelo,
        muestra,
        y_muestra,
        scoring="roc_auc",
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1
    )

    importancia = pd.DataFrame({
        "variable": predictores,
        "importancia": permutacion.importances_mean
    }).sort_values("importancia", ascending=False)

    plt.figure(figsize=(8, 5))
    orden = importancia.sort_values("importancia")
    plt.barh(orden["variable"], orden["importancia"])
    plt.xlabel("Disminucion de ROC-AUC al permutar")
    plt.title(f"Importancia global - {nombre_modelo}")
    plt.tight_layout()
    plt.savefig(
        GRAFICOS_DIR / "importancia_global_mejor_modelo.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    if hasattr(modelo, "named_steps"):
        estimador = modelo.named_steps["logisticregression"]
        transformada = modelo.named_steps["standardscaler"].transform(
            muestra
        )
        explicador = shap.LinearExplainer(estimador, transformada)
        valores = explicador(transformada)
        datos_grafica = transformada
    else:
        explicador = shap.TreeExplainer(modelo)
        valores = explicador(muestra)
        datos_grafica = muestra

    if valores.values.ndim == 3:
        valores = shap.Explanation(
            values=valores.values[:, :, 1],
            base_values=valores.base_values[:, 1],
            data=valores.data,
            feature_names=predictores
        )

    shap.summary_plot(
        valores,
        datos_grafica,
        feature_names=predictores,
        show=False
    )
    plt.title(f"Resumen SHAP - {nombre_modelo}")
    plt.tight_layout()
    plt.savefig(
        GRAFICOS_DIR / "shap_summary_mejor_modelo.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    shap_medio = pd.DataFrame({
        "variable": predictores,
        "shap_abs_medio": np.abs(valores.values).mean(axis=0)
    }).sort_values("shap_abs_medio", ascending=False)

    return importancia, shap_medio


# aca se crean mapas de probabilidad y de errores
def generar_mapas_predictivos(
    datos,
    probabilidades,
    umbral,
    random_state=42
):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    salida = datos[[
        "longitud",
        "latitud",
        "lago",
        "cyano_alta"
    ]].copy()

    salida["probabilidad"] = probabilidades
    salida["prediccion"] = (
        salida["probabilidad"] >= umbral
    ).astype("int8")

    salida["error"] = np.select(
        [
            (salida["cyano_alta"] == 1)
            & (salida["prediccion"] == 1),
            (salida["cyano_alta"] == 0)
            & (salida["prediccion"] == 1),
            (salida["cyano_alta"] == 1)
            & (salida["prediccion"] == 0)
        ],
        ["Verdadero positivo", "Falso positivo", "Falso negativo"],
        default="Verdadero negativo"
    )

    colores = ListedColormap([
        "#2166ac",
        "#67a9cf",
        "#fdae61",
        "#b2182b"
    ])
    limites = [0, 0.25, 0.50, 0.75, 1.00001]
    norma = BoundaryNorm(limites, colores.N)

    for lago in sorted(salida["lago"].unique()):
        parte = salida[salida["lago"] == lago]
        mapa = parte.sample(
            min(120000, len(parte)),
            random_state=random_state
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        puntos = axes[0].scatter(
            mapa["longitud"],
            mapa["latitud"],
            c=mapa["probabilidad"],
            s=2,
            cmap=colores,
            norm=norma
        )
        barra = fig.colorbar(
            puntos,
            ax=axes[0],
            ticks=[0.125, 0.375, 0.625, 0.875]
        )
        barra.ax.set_yticklabels([
            "Muy baja",
            "Baja",
            "Alta",
            "Muy alta"
        ])
        axes[0].set_title(f"Probabilidad de cianobacteria alta - {lago}")
        axes[0].set_xlabel("Longitud")
        axes[0].set_ylabel("Latitud")

        colores_error = {
            "Verdadero negativo": "#d9d9d9",
            "Verdadero positivo": "#1b9e77",
            "Falso positivo": "#d95f02",
            "Falso negativo": "#7570b3"
        }

        for categoria in [
            "Verdadero negativo",
            "Verdadero positivo",
            "Falso positivo",
            "Falso negativo"
        ]:
            puntos_error = mapa[mapa["error"] == categoria]
            axes[1].scatter(
                puntos_error["longitud"],
                puntos_error["latitud"],
                s=2 if categoria == "Verdadero negativo" else 8,
                alpha=0.35 if categoria == "Verdadero negativo" else 0.75,
                color=colores_error[categoria],
                label=categoria
            )

        axes[1].set_title(f"Distribucion espacial de errores - {lago}")
        axes[1].set_xlabel("Longitud")
        axes[1].set_ylabel("Latitud")
        axes[1].legend(markerscale=3, fontsize=8)

        for eje in axes:
            eje.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.savefig(
            GRAFICOS_DIR / f"mapa_predictivo_{nombre_seguro(lago)}.png",
            dpi=180,
            bbox_inches="tight"
        )
        plt.show()
        plt.close()

    resumen = (
        salida
        .groupby(["lago", "error"])
        .size()
        .rename("observaciones")
        .reset_index()
    )

    return salida, resumen
