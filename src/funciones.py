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