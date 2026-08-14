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


# ============================================================
# CONEXIÓN A COPERNICUS / OPENEO
# ============================================================

def conectar_copernicus():
    """
    Se conecta al backend openEO de Copernicus.
    La autenticación se hace desde el navegador.
    """

    conn = openeo.connect(CONEXION_URL)

    conn.authenticate_oidc()

    print("Conexión realizada correctamente.")
    print("Backend:", CONEXION_URL)

    return conn


# ============================================================
# FUNCIONES BÁSICAS
# ============================================================

def nombre_seguro(texto):
    """
    Convierte un nombre como Atitlán en atitlan.
    """

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


def division_segura(numerador, denominador):
    """
    Hace una división evitando dividir entre cero.
    """

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


# ============================================================
# DESCARGA DE SENTINEL-2
# ============================================================

def descargar_bandas_fecha(
    conn,
    lago,
    bbox,
    fecha,
    resolucion=RESOLUCION_METROS
):
    """
    Descarga las bandas necesarias de Sentinel-2 L2A
    para un lago y una fecha.

    Si el archivo ya existe no lo vuelve a descargar.
    """

    slug = nombre_seguro(lago)

    salida = (
        DATOS_DIR
        / f"{slug}_{fecha}.tif"
    )

    if salida.exists():

        print(f"Ya existe: {salida}")

        return salida


    fecha_inicio = pd.Timestamp(fecha)

    fecha_fin = (
        fecha_inicio
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")


    print(
        f"Descargando {lago} - {fecha}..."
    )


    cubo = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=[
            fecha,
            fecha_fin
        ],
        bands=BANDAS
    )


    # Todas las bandas quedan en una misma grilla.
    cubo = cubo.resample_spatial(
        resolution=resolucion,
        method="near"
    )


    # Quitamos la dimensión temporal.
    cubo = cubo.max_time()


    # Ejecutamos como batch job.
    cubo.execute_batch(
        outputfile=str(salida),
        out_format="GTiff",
        title=f"Lab4 {lago} {fecha}"
    )


    print(
        f"Guardado: {salida}"
    )

    return salida


# ============================================================
# MÁSCARA DE AGUA
# ============================================================

def mascara_agua_cyanolakes(
    b02,
    b03,
    b04,
    b08,
    b11,
    b12
):
    """
    Máscara de agua basada en el script
    Maximum Peak Height Bloom Index de CyanoLakes.
    """

    mndwi_threshold = 0.42

    ndwi_threshold = 0.40


    ndvi = division_segura(
        b08 - b04,
        b08 + b04
    )


    mndwi = division_segura(
        b03 - b11,
        b03 + b11
    )


    ndwi = division_segura(
        b03 - b08,
        b03 + b08
    )


    ndwi_leaves = division_segura(
        b08 - b11,
        b08 + b11
    )


    aweish = (
        b02
        + 2.5 * b03
        - 1.5 * (b08 + b11)
        - 0.25 * b12
    )


    aweinsh = (
        4.0 * (b03 - b11)
        - (
            0.25 * b08
            + 2.75 * b11
        )
    )


    dbsi = (
        division_segura(
            b11 - b03,
            b11 + b03
        )
        - ndvi
    )


    condicion_agua = (
        (mndwi > mndwi_threshold)
        | (ndwi > ndwi_threshold)
        | (aweinsh > 0.1879)
        | (aweish > 0.1112)
        | (ndvi < -0.2)
        | (ndwi_leaves > 1)
    )


    # Quitar áreas urbanas y suelo desnudo.
    condicion_agua &= ~(
        (aweinsh <= -0.03)
        | (dbsi > 0)
    )


    return condicion_agua


# ============================================================
# CÁLCULO DE ÍNDICES
# ============================================================

def leer_y_calcular_indices(ruta_tif):
    """
    Lee un GeoTIFF y calcula:

    NDVI
    NDWI
    Maximum Peak Height Bloom Index

    También quita nubes y píxeles fuera del agua.
    """

    ruta_tif = Path(ruta_tif)


    with rasterio.open(ruta_tif) as src:

        data = (
            src.read()
            .astype("float32")
        )

        perfil = src.profile.copy()

        transform = src.transform

        crs = src.crs

        nodata = src.nodata


    if data.shape[0] != len(BANDAS):

        raise ValueError(
            f"Se esperaban {len(BANDAS)} bandas, "
            f"pero el archivo tiene "
            f"{data.shape[0]}."
        )


    valores = {
        banda: data[i]
        for i, banda in enumerate(BANDAS)
    }


    # Revisar si la reflectancia viene escalada.
    p99_rojo = np.nanpercentile(
        valores["B04"],
        99
    )


    if p99_rojo > 2:
        escala = 0.0001
    else:
        escala = 1.0


    bandas_reflectancia = [
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B08",
        "B11",
        "B12"
    ]


    for banda in bandas_reflectancia:

        valores[banda] = (
            valores[banda]
            * escala
        )


    b02 = valores["B02"]
    b03 = valores["B03"]
    b04 = valores["B04"]
    b05 = valores["B05"]
    b06 = valores["B06"]
    b08 = valores["B08"]
    b11 = valores["B11"]
    b12 = valores["B12"]
    scl = valores["SCL"]


    # ========================================================
    # NDVI
    # ========================================================

    ndvi = division_segura(
        b08 - b04,
        b08 + b04
    )


    # ========================================================
    # NDWI
    # ========================================================

    ndwi = division_segura(
        b03 - b08,
        b03 + b08
    )


    # ========================================================
    # MCI
    # Maximum Chlorophyll Index
    # ========================================================

    mci = (
        b05
        - b04
        - (
            (b06 - b04)
            * (
                (705 - 665)
                / (740 - 665)
            )
        )
    )


    # ========================================================
    # FAI
    # Floating Algal Index
    # ========================================================

    fai = (
        b06
        - b04
        - (
            (b08 - b04)
            * (
                (740 - 665)
                / (842 - 665)
            )
        )
    )


    # ========================================================
    # ÍNDICE DE CIANOBACTERIA
    # ========================================================

    cyano = np.maximum(
        mci,
        fai
    )


    # ========================================================
    # MÁSCARA DE AGUA
    # ========================================================

    mascara_agua = (
        mascara_agua_cyanolakes(
            b02=b02,
            b03=b03,
            b04=b04,
            b08=b08,
            b11=b11,
            b12=b12
        )
    )


    # ========================================================
    # MÁSCARA DE NUBES / CALIDAD
    # ========================================================

    clases_scl_no_validas = [
        0,   # No data
        1,   # Saturated / defective
        3,   # Cloud shadow
        8,   # Cloud medium probability
        9,   # Cloud high probability
        10,  # Cirrus
        11   # Snow / ice
    ]


    mascara_calidad = ~np.isin(
        scl.astype("int16"),
        clases_scl_no_validas
    )


    # ========================================================
    # MÁSCARA FINAL
    # ========================================================

    mascara_valida = (
        np.isfinite(b02)
        & np.isfinite(b03)
        & np.isfinite(b04)
        & np.isfinite(b08)
        & np.isfinite(b11)
        & np.isfinite(b12)
        & (b02 > 0)
        & (b03 > 0)
        & (b04 > 0)
        & (b08 > 0)
        & mascara_agua
        & mascara_calidad
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


# ============================================================
# MAPAS
# ============================================================

def mostrar_mapa(
    matriz,
    titulo,
    nombre_archivo,
    etiqueta="Valor del índice",
    percentiles=(2, 98)
):
    """
    Muestra un raster y guarda la imagen.
    """

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


# ============================================================
# GRÁFICA TEMPORAL POR LAGO
# ============================================================

def grafica_temporal_lago(
    df,
    lago
):
    """
    Grafica el promedio de cianobacteria por fecha.
    """

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


    plt.xlabel(
        "Fecha"
    )


    plt.ylabel(
        "Índice promedio de cianobacteria"
    )


    plt.grid(
        alpha=0.25
    )


    plt.xticks(
        rotation=45
    )


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


    print(
        "Gráfica guardada en:",
        salida
    )


# ============================================================
# COMPARACIÓN DE LOS DOS LAGOS
# ============================================================

def grafica_comparacion_lagos(df):

    plt.figure(
        figsize=(10, 5)
    )


    hay_datos = False


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


        hay_datos = True


        plt.plot(
            datos["fecha"],
            datos["cyano_promedio"],
            marker="o",
            label=lago
        )


    if not hay_datos:

        plt.close()

        print(
            "No hay datos para la comparación."
        )

        return


    plt.title(
        "Comparación temporal del "
        "índice de cianobacteria"
    )


    plt.xlabel(
        "Fecha"
    )


    plt.ylabel(
        "Índice promedio de cianobacteria"
    )


    plt.legend()


    plt.grid(
        alpha=0.25
    )


    plt.xticks(
        rotation=45
    )


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


    print(
        "Gráfica guardada en:",
        salida
    )


# ============================================================
# ANÁLISIS TEMPORAL
# ============================================================

def analizar_temporal(df):
    """
    Crea un análisis corto usando los resultados reales.
    """

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
                f"datos para analizar "
                f"la tendencia."
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


        dias = (
            datos["fecha"]
            - datos["fecha"].min()
        ).dt.days.to_numpy(
            dtype=float
        )


        valores = (
            datos[
                "cyano_promedio"
            ]
            .to_numpy(
                dtype=float
            )
        )


        if np.ptp(dias) > 0:

            pendiente = np.polyfit(
                dias,
                valores,
                1
            )[0]

        else:

            pendiente = 0.0


        cambio_estimado = (
            pendiente
            * np.ptp(dias)
        )


        rango = (
            np.nanmax(valores)
            - np.nanmin(valores)
        )


        if (
            rango == 0
            or abs(cambio_estimado)
            < 0.15 * rango
        ):

            tendencia = (
                "no muestra una tendencia clara "
                "y más bien fluctúa entre fechas"
            )


        elif cambio_estimado > 0:

            tendencia = (
                "muestra una tendencia general "
                "hacia valores más altos"
            )


        else:

            tendencia = (
                "muestra una tendencia general "
                "hacia valores más bajos"
            )


        texto = (
            f"{lago}: el valor promedio más alto "
            f"aparece el {pico['fecha'].date()} "
            f"con {pico['cyano_promedio']:.5f}. "
            f"El valor más bajo aparece el "
            f"{minimo['fecha'].date()} "
            f"con {minimo['cyano_promedio']:.5f}. "
            f"En general, la serie {tendencia}. "
            f"En la fecha del pico, "
            f"{pico['porcentaje_cyano_alto']:.2f}% "
            f"de los píxeles válidos de agua "
            f"superó el valor 0.05 del índice."
        )


        textos.append(texto)


    return textos


# ============================================================
# MAPA DE LA FECHA PICO
# ============================================================

def mapa_fecha_pico(
    df,
    lago
):
    """
    Busca la fecha con el promedio más alto
    y genera su mapa.
    """

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