from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
DATOS_ORIGINALES = RAIZ / "data" / "raw"
DATOS_PROCESADOS = RAIZ / "data" / "processed"

STOPWORDS_ES = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "aquel",
    "aquella", "aquellas", "aquellos", "aqui", "aquí", "asi", "así", "aun", "aún", "aunque",
    "bajo", "bien", "cada", "como", "con", "contra", "cual", "cuando",
    "de", "del", "desde", "donde", "dos", "durante", "e", "el", "ella",
    "ellas", "ellos", "en", "entre", "era", "eramos", "eran", "es", "esa",
    "esas", "ese", "eso", "esos", "esta", "estaba", "estamos", "estan",
    "estar", "estas", "estás", "este", "esto", "estos", "fue", "fuera", "fueron",
    "ha", "hace", "hacia", "han", "hasta", "hay", "la", "las", "le",
    "les", "lo", "los", "mas", "más", "me", "mi", "mis", "mucho", "muy", "nada",
    "ni", "no", "nos", "nuestra", "nuestro", "o", "otra", "otro", "para",
    "pero", "poco", "por", "porque", "que", "quien", "se", "ser", "si",
    "sin", "sobre", "son", "su", "sus", "tambien", "también", "te", "tiene", "todo",
    "tu", "un", "una", "uno", "unos", "usted", "ya", "y", "yo",
}

PALABRAS_POSITIVAS = {
    "bien", "bueno", "buena", "buenos", "buenas", "excelente", "gracias",
    "feliz", "felicidades", "apoyo", "apoyar", "mejor", "correcto", "justicia",
    "esperanza", "amor", "bendiciones", "bravo", "alegria", "logro", "exito",
}

PALABRAS_NEGATIVAS = {
    "mal", "malo", "mala", "peor", "corrupcion", "corrupto", "corrupta",
    "ladron", "ladrona", "odio", "triste", "mentira", "mentiroso", "fracaso",
    "delito", "criminal", "violencia", "muerto", "muerte", "robo", "problema",
    "vergüenza", "verguenza", "injusticia", "basura", "terrible", "desastre",
}


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    videos = pd.read_csv(DATOS_ORIGINALES / "youtube_videos.csv")
    comentarios = pd.read_csv(DATOS_ORIGINALES / "youtube_comments.csv")
    return videos, comentarios


def normalizar_columnas_texto(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    resultado = df.copy()
    for columna in columnas:
        resultado[columna] = resultado[columna].astype("string").str.strip()
        resultado.loc[resultado[columna].eq(""), columna] = pd.NA
    return resultado


def convertir_conteo(valor: object) -> float:
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip().lower().replace("\u00a0", " ")
    if not texto:
        return np.nan
    multiplicador = 1
    if re.search(r"\bmil\b|k\b", texto):
        multiplicador = 1_000
    elif re.search(r"mill[oó]n|\bm\b", texto):
        multiplicador = 1_000_000
    if multiplicador > 1:
        coincidencia = re.search(r"\d+(?:[.,]\d+)?", texto)
        if not coincidencia:
            return np.nan
        numero = float(coincidencia.group().replace(",", "."))
        return round(numero * multiplicador)
    digitos = re.sub(r"\D", "", texto)
    return float(digitos) if digitos else np.nan


def limpiar_texto(texto: object) -> str:
    if pd.isna(texto):
        return ""
    limpio = unicodedata.normalize("NFKC", str(texto)).lower()
    limpio = re.sub(r"https?://\S+|www\.\S+", " ", limpio)
    limpio = re.sub(r"@[\w.-]+", " ", limpio)
    limpio = re.sub(r"#(?=\w)", "", limpio)
    limpio = re.sub(r"[_\d]+", " ", limpio)
    limpio = re.sub(r"[^a-záéíóúüñ\s]", " ", limpio)
    tokens = [
        token for token in limpio.split()
        if len(token) > 1 and token not in STOPWORDS_ES
    ]
    return " ".join(tokens)


def preparar_datos(
    videos: pd.DataFrame, comentarios: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos_limpios = normalizar_columnas_texto(
        videos,
        [
            "video_id", "title", "channel_name", "channel_id", "channel_handle",
            "owner_handle", "source_query", "source_group", "category",
        ],
    )
    comentarios_limpios = normalizar_columnas_texto(
        comentarios,
        [
            "video_id", "comment_id", "channel_name", "channel_id", "author_name",
            "author_channel_id", "author_handle", "source_query", "source_group",
        ],
    )

    videos_limpios["view_count_desde_texto"] = (
        videos_limpios["view_count_text"].map(convertir_conteo).astype("Int64")
    )
    videos_limpios["publish_date"] = pd.to_datetime(
        videos_limpios["publish_date"], errors="coerce", utc=True
    )
    videos_limpios["upload_date"] = pd.to_datetime(
        videos_limpios["upload_date"], errors="coerce", utc=True
    )
    videos_limpios["texto_original"] = (
        videos_limpios["title"].fillna("") + " "
        + videos_limpios["description"].fillna("")
    ).str.strip()
    videos_limpios["texto_limpio"] = videos_limpios["texto_original"].map(limpiar_texto)

    comentarios_limpios["like_count"] = (
        comentarios_limpios["like_count_text"].map(convertir_conteo).fillna(0).astype("Int64")
    )
    comentarios_limpios["reply_count"] = (
        pd.to_numeric(comentarios_limpios["reply_count"], errors="coerce")
        .fillna(0)
        .astype("Int64")
    )
    comentarios_limpios["texto_original"] = comentarios_limpios["text"].fillna("")
    comentarios_limpios["texto_limpio"] = comentarios_limpios["texto_original"].map(limpiar_texto)
    return videos_limpios, comentarios_limpios


def diagnostico_variables(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for columna in df.columns:
        serie = df[columna]
        filas.append(
            {
                "variable": columna,
                "tipo": str(serie.dtype),
                "faltantes": int(serie.isna().sum()),
                "porcentaje_faltante": round(100 * serie.isna().mean(), 2),
                "valores_unicos": int(serie.nunique(dropna=True)),
                "constante": bool(serie.nunique(dropna=True) <= 1),
            }
        )
    return pd.DataFrame(filas)


def diagnostico_atipicos(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for columna in df.select_dtypes(include="number").columns:
        serie = pd.to_numeric(df[columna], errors="coerce").dropna()
        if serie.empty:
            continue
        q1, q3 = serie.quantile([0.25, 0.75])
        rango = q3 - q1
        inferior = q1 - 1.5 * rango
        superior = q3 + 1.5 * rango
        atipicos = serie[(serie < inferior) | (serie > superior)]
        filas.append(
            {
                "variable": columna,
                "minimo": serie.min(),
                "mediana": serie.median(),
                "maximo": serie.max(),
                "limite_inferior": inferior,
                "limite_superior": superior,
                "cantidad_atipicos": int(atipicos.size),
            }
        )
    return pd.DataFrame(filas)


def revisar_consistencia(
    videos: pd.DataFrame, comentarios: pd.DataFrame
) -> pd.DataFrame:
    union = comentarios.merge(
        videos[["video_id", "channel_id", "channel_name", "channel_handle"]],
        on="video_id",
        how="left",
        suffixes=("_comentario", "_video"),
        indicator=True,
    )
    revision = [
        {
            "revision": "video_id duplicado en videos",
            "cantidad": int(videos["video_id"].duplicated().sum()),
        },
        {
            "revision": "comment_id duplicado en comentarios",
            "cantidad": int(comentarios["comment_id"].duplicated().sum()),
        },
        {
            "revision": "comentarios sin video asociado",
            "cantidad": int(union["_merge"].ne("both").sum()),
        },
        {
            "revision": "channel_id distinto entre comentario y video",
            "cantidad": int(
                union["channel_id_comentario"].ne(union["channel_id_video"]).sum()
            ),
        },
        {
            "revision": "channel_id con varios nombres en videos",
            "cantidad": int((videos.groupby("channel_id")["channel_name"].nunique() > 1).sum()),
        },
        {
            "revision": "channel_id con varios handles en videos",
            "cantidad": int((videos.groupby("channel_id")["channel_handle"].nunique() > 1).sum()),
        },
        {
            "revision": "nombre de canal ligado a varios channel_id",
            "cantidad": int((videos.groupby("channel_name")["channel_id"].nunique() > 1).sum()),
        },
        {
            "revision": "author_channel_id con varios nombres",
            "cantidad": int(
                (comentarios.groupby("author_channel_id")["author_name"].nunique() > 1).sum()
            ),
        },
        {
            "revision": "registros con caracter de reemplazo en videos",
            "cantidad": int(videos.astype("string").apply(lambda s: s.str.contains("�", na=False)).any(axis=1).sum()),
        },
        {
            "revision": "registros con caracter de reemplazo en comentarios",
            "cantidad": int(comentarios.astype("string").apply(lambda s: s.str.contains("�", na=False)).any(axis=1).sum()),
        },
    ]
    return pd.DataFrame(revision)


def integrar_datos(
    videos: pd.DataFrame, comentarios: pd.DataFrame
) -> pd.DataFrame:
    columnas_video = [
        "video_id", "title", "channel_id", "channel_name", "channel_handle",
        "view_count", "category", "publish_date", "texto_limpio",
    ]
    return comentarios.merge(
        videos[columnas_video],
        on="video_id",
        how="left",
        suffixes=("_comentario", "_video"),
        validate="many_to_one",
        indicator="estado_union",
    )


def efecto_limpieza(comentarios: pd.DataFrame) -> pd.DataFrame:
    original = comentarios["texto_original"].fillna("").str.strip()
    limpio = comentarios["texto_limpio"].fillna("").str.strip()
    return pd.DataFrame(
        {
            "medida": [
                "registros originales", "registros eliminados", "textos modificados",
                "textos vacios antes", "textos vacios despues",
                "duplicados de texto antes", "duplicados de texto despues",
            ],
            "cantidad": [
                len(comentarios), 0, int(original.ne(limpio).sum()),
                int(original.eq("").sum()), int(limpio.eq("").sum()),
                int(original.duplicated().sum()), int(limpio.duplicated().sum()),
            ],
        }
    )


def extraer_hashtags(serie: pd.Series) -> pd.DataFrame:
    contador = Counter()
    for texto in serie.fillna(""):
        contador.update(etiqueta.lower() for etiqueta in re.findall(r"#([\wáéíóúüñ]+)", texto))
    return pd.DataFrame(contador.most_common(20), columns=["hashtag", "frecuencia"])


def frecuencias_texto(serie: pd.Series, ngrama: int = 1, limite: int = 20) -> pd.DataFrame:
    contador = Counter()
    for texto in serie.fillna(""):
        tokens = texto.split()
        if ngrama == 1:
            contador.update(tokens)
        else:
            contador.update(" ".join(grupo) for grupo in zip(*(tokens[i:] for i in range(ngrama))))
    nombre = "palabra" if ngrama == 1 else "bigrama"
    return pd.DataFrame(contador.most_common(limite), columns=[nombre, "frecuencia"])


def resumen_numerico(serie: pd.Series, nombre: str) -> pd.DataFrame:
    valores = pd.to_numeric(serie, errors="coerce")
    resumen = valores.describe(percentiles=[0.25, 0.5, 0.75]).rename(
        {"count": "cantidad", "mean": "media", "std": "desviacion", "min": "minimo", "25%": "q1", "50%": "mediana", "75%": "q3", "max": "maximo"}
    )
    return resumen.rename(nombre).to_frame().T.round(2)


def tablas_exploratorias(
    videos: pd.DataFrame, comentarios: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    comentarios_video = (
        comentarios.groupby("video_id")
        .agg(comentarios=("comment_id", "count"), autores_unicos=("author_channel_id", "nunique"))
        .reset_index()
        .merge(videos[["video_id", "title", "channel_id", "channel_name", "view_count"]], on="video_id", how="right")
        .fillna({"comentarios": 0, "autores_unicos": 0})
    )
    comentarios_video[["comentarios", "autores_unicos"]] = comentarios_video[
        ["comentarios", "autores_unicos"]
    ].astype(int)
    videos_canal = (
        videos.groupby(["channel_id", "channel_name"], dropna=False)
        .agg(videos=("video_id", "nunique"), visualizaciones=("view_count", "sum"))
        .reset_index()
        .sort_values("videos", ascending=False)
    )
    comentarios_canal = (
        comentarios.groupby(["channel_id", "channel_name"], dropna=False)
        .agg(comentarios=("comment_id", "count"), autores_unicos=("author_channel_id", "nunique"))
        .reset_index()
        .sort_values("comentarios", ascending=False)
    )
    categorias = videos["category"].value_counts(dropna=False).rename_axis("categoria").reset_index(name="videos")
    consultas = videos["source_query"].value_counts(dropna=False).rename_axis("consulta").reset_index(name="videos")
    return {
        "comentarios_video": comentarios_video.sort_values("comentarios", ascending=False),
        "videos_canal": videos_canal,
        "comentarios_canal": comentarios_canal,
        "categorias": categorias,
        "consultas": consultas,
        "hashtags": extraer_hashtags(comentarios["texto_original"]),
        "palabras": frecuencias_texto(comentarios["texto_limpio"], 1),
        "bigramas": frecuencias_texto(comentarios["texto_limpio"], 2),
    }


def calcular_concentracion(tabla: pd.DataFrame, columna: str, etiqueta: str) -> pd.DataFrame:
    valores = tabla[columna].sort_values(ascending=False).reset_index(drop=True)
    total = valores.sum()
    cortes = {
        "principal": 1,
        "cinco principales": min(5, len(valores)),
        "diez por ciento principal": max(1, math.ceil(0.10 * len(valores))),
    }
    filas = []
    for grupo, cantidad in cortes.items():
        filas.append(
            {
                "unidad": etiqueta,
                "grupo": grupo,
                "cantidad_unidades": cantidad,
                "comentarios": int(valores.head(cantidad).sum()),
                "porcentaje_comentarios": round(100 * valores.head(cantidad).sum() / total, 2) if total else 0,
            }
        )
    return pd.DataFrame(filas)


def popularidad_participacion(tabla_video: pd.DataFrame) -> pd.DataFrame:
    observados = tabla_video[tabla_video["comentarios"] > 0].copy()
    return pd.DataFrame(
        {
            "medida": ["correlacion de Pearson", "correlacion de Spearman", "videos con comentarios"],
            "valor": [
                observados["view_count"].corr(observados["comentarios"], method="pearson"),
                observados["view_count"].corr(observados["comentarios"], method="spearman"),
                len(observados),
            ],
        }
    ).round(3)


def clasificar_sentimiento(texto_limpio: str) -> tuple[int, str]:
    tokens = set(texto_limpio.split())
    puntuacion = len(tokens & PALABRAS_POSITIVAS) - len(tokens & PALABRAS_NEGATIVAS)
    etiqueta = "positivo" if puntuacion > 0 else "negativo" if puntuacion < 0 else "neutral"
    return puntuacion, etiqueta


def agregar_sentimiento_exploratorio(comentarios: pd.DataFrame) -> pd.DataFrame:
    resultado = comentarios.copy()
    pares = resultado["texto_limpio"].map(clasificar_sentimiento)
    resultado["sentimiento_puntaje"] = pares.map(lambda valor: valor[0])
    resultado["sentimiento"] = pares.map(lambda valor: valor[1])
    return resultado


def crear_tablas_red(
    videos: pd.DataFrame, comentarios: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aristas = (
        comentarios.groupby(["author_channel_id", "video_id"], dropna=False)
        .agg(peso=("comment_id", "count"))
        .reset_index()
    )
    aristas["origen"] = "autor:" + aristas["author_channel_id"].astype(str)
    aristas["destino"] = "video:" + aristas["video_id"].astype(str)
    aristas = aristas[["origen", "destino", "author_channel_id", "video_id", "peso"]]

    autores = (
        comentarios.groupby("author_channel_id", dropna=False)
        .agg(
            etiqueta=("author_name", "first"),
            handle=("author_handle", "first"),
            comentarios=("comment_id", "count"),
            videos_comentados=("video_id", "nunique"),
        )
        .reset_index()
    )
    autores["nodo_id"] = "autor:" + autores["author_channel_id"].astype(str)
    autores["tipo"] = "autor"
    autores["channel_id"] = pd.NA
    autores["visualizaciones"] = pd.NA
    autores["categoria"] = pd.NA
    autores["video_id"] = pd.NA
    autores["canal"] = pd.NA

    videos_nodos = (
        videos[videos["video_id"].isin(comentarios["video_id"])]
        .copy()
        .rename(columns={"title": "etiqueta", "view_count": "visualizaciones"})
    )
    conteos_video = comentarios.groupby("video_id").agg(
        comentarios=("comment_id", "count"),
        autores_unicos=("author_channel_id", "nunique"),
    )
    videos_nodos = videos_nodos.join(conteos_video, on="video_id")
    videos_nodos["nodo_id"] = "video:" + videos_nodos["video_id"].astype(str)
    videos_nodos["tipo"] = "video"
    videos_nodos["handle"] = videos_nodos["channel_handle"]
    videos_nodos["videos_comentados"] = pd.NA
    videos_nodos["author_channel_id"] = pd.NA
    videos_nodos["canal"] = videos_nodos["channel_name"]

    columnas = [
        "nodo_id", "tipo", "etiqueta", "handle", "author_channel_id", "video_id",
        "channel_id", "canal", "comentarios", "videos_comentados", "autores_unicos",
        "visualizaciones", "categoria",
    ]
    autores["autores_unicos"] = pd.NA
    registros = autores.reindex(columns=columnas).to_dict("records")
    registros.extend(videos_nodos.reindex(columns=columnas).to_dict("records"))
    nodos = pd.DataFrame.from_records(registros, columns=columnas)
    return nodos, aristas


def crear_red(nodos: pd.DataFrame, aristas: pd.DataFrame) -> nx.Graph:
    red = nx.Graph()
    for fila in nodos.to_dict("records"):
        nodo_id = fila.pop("nodo_id")
        red.add_node(nodo_id, **fila)
    for fila in aristas.to_dict("records"):
        red.add_edge(fila["origen"], fila["destino"], peso=int(fila["peso"]))
    return red


def resumen_red(red: nx.Graph) -> pd.DataFrame:
    componentes = list(nx.connected_components(red))
    grados = dict(red.degree())
    return pd.DataFrame(
        {
            "medida": [
                "nodos", "autores", "videos", "aristas", "componentes conexas",
                "nodos en componente mayor", "grado medio", "densidad bipartita",
            ],
            "valor": [
                red.number_of_nodes(),
                sum(d["tipo"] == "autor" for _, d in red.nodes(data=True)),
                sum(d["tipo"] == "video" for _, d in red.nodes(data=True)),
                red.number_of_edges(),
                len(componentes),
                max(map(len, componentes), default=0),
                np.mean(list(grados.values())) if grados else 0,
                nx.algorithms.bipartite.density(
                    red, {n for n, d in red.nodes(data=True) if d["tipo"] == "autor"}
                ),
            ],
        }
    ).round(4)


def audiencias_compartidas(
    comentarios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    autores = (
        comentarios.groupby("author_channel_id")
        .agg(
            autor=("author_name", "first"),
            comentarios=("comment_id", "count"),
            videos=("video_id", "nunique"),
            canales=("channel_id", "nunique"),
            consultas=("source_query", "nunique"),
        )
        .reset_index()
        .sort_values(["videos", "canales", "comentarios"], ascending=False)
    )
    pares = Counter()
    for _, grupo in comentarios.groupby("author_channel_id"):
        videos = sorted(grupo["video_id"].unique())
        pares.update(combinations(videos, 2))
    videos_compartidos = pd.DataFrame(
        [(a, b, peso) for (a, b), peso in pares.most_common()],
        columns=["video_id_1", "video_id_2", "autores_compartidos"],
    )
    return autores, videos_compartidos


def puentes_observados(red: nx.Graph, limite: int = 10) -> pd.DataFrame:
    puntos = set(nx.articulation_points(red))
    autores = []
    for nodo, datos in red.nodes(data=True):
        if datos["tipo"] != "autor":
            continue
        autores.append(
            {
                "nodo_id": nodo,
                "autor": datos["etiqueta"],
                "videos_comentados": red.degree(nodo),
                "punto_de_articulacion": nodo in puntos,
            }
        )
    return (
        pd.DataFrame(autores)
        .sort_values(["punto_de_articulacion", "videos_comentados"], ascending=False)
        .head(limite)
    )


def resumen_componentes(
    red: nx.Graph, comentarios: pd.DataFrame, limite: int = 3
) -> pd.DataFrame:
    componentes = sorted(nx.connected_components(red), key=len, reverse=True)
    filas = []
    for numero, componente in enumerate(componentes[:limite], start=1):
        videos = {
            red.nodes[nodo]["video_id"]
            for nodo in componente
            if red.nodes[nodo]["tipo"] == "video"
        }
        autores = {
            red.nodes[nodo]["author_channel_id"]
            for nodo in componente
            if red.nodes[nodo]["tipo"] == "autor"
        }
        seleccion = comentarios[comentarios["video_id"].isin(videos)]
        palabras = frecuencias_texto(seleccion["texto_limpio"], limite=5)
        sentimiento = seleccion["sentimiento"].value_counts(normalize=True)
        filas.append(
            {
                "componente": numero,
                "nodos": len(componente),
                "autores": len(autores),
                "videos": len(videos),
                "comentarios": len(seleccion),
                "palabras_frecuentes": ", ".join(palabras["palabra"]),
                "sentimiento_principal": sentimiento.index[0] if not sentimiento.empty else pd.NA,
                "porcentaje_sentimiento_principal": round(100 * sentimiento.iloc[0], 2) if not sentimiento.empty else pd.NA,
            }
        )
    return pd.DataFrame(filas)


def graficar_barras(
    tabla: pd.DataFrame,
    etiqueta: str,
    valor: str,
    titulo: str,
    cantidad: int = 10,
    color: str = "#167D9A",
) -> plt.Figure:
    datos = tabla.head(cantidad).sort_values(valor)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(datos[etiqueta].astype(str), datos[valor], color=color)
    ax.set_title(titulo)
    ax.set_xlabel(valor.replace("_", " ").capitalize())
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def graficar_popularidad(tabla_video: pd.DataFrame) -> plt.Figure:
    datos = tabla_video[tabla_video["comentarios"] > 0]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(datos["view_count"], datos["comentarios"], alpha=0.65, color="#C6536B")
    ax.set_xscale("symlog")
    ax.set_title("Visualizaciones y comentarios observados")
    ax.set_xlabel("Visualizaciones en escala logaritmica")
    ax.set_ylabel("Comentarios recolectados")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def graficar_red_completa(red: nx.Graph) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 10))
    posicion = nx.spring_layout(red, seed=42, k=1 / max(math.sqrt(red.number_of_nodes()), 1), iterations=120)
    autores = [n for n, d in red.nodes(data=True) if d["tipo"] == "autor"]
    videos = [n for n, d in red.nodes(data=True) if d["tipo"] == "video"]
    pesos = [red[u][v]["peso"] for u, v in red.edges()]
    nx.draw_networkx_edges(
        red, posicion, ax=ax, alpha=0.20,
        width=[0.35 + 0.45 * min(peso, 5) for peso in pesos], edge_color="#6B7280",
    )
    nx.draw_networkx_nodes(
        red, posicion, nodelist=autores, node_size=20, node_color="#4DA3D9",
        alpha=0.75, linewidths=0, label="Autores", ax=ax,
    )
    nx.draw_networkx_nodes(
        red, posicion, nodelist=videos,
        node_size=[45 + 8 * red.degree(nodo) for nodo in videos],
        node_color="#E07A5F", alpha=0.90, linewidths=0.3, edgecolors="white",
        label="Videos", ax=ax,
    )
    ax.set_title("Red bipartita completa de autores y videos")
    ax.legend(frameon=False)
    ax.axis("off")
    fig.tight_layout()
    return fig


def guardar_resultados(
    videos: pd.DataFrame,
    comentarios: pd.DataFrame,
    nodos: pd.DataFrame,
    aristas: pd.DataFrame,
) -> None:
    DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)
    videos.to_csv(DATOS_PROCESADOS / "youtube_videos_limpio.csv", index=False)
    comentarios.to_csv(DATOS_PROCESADOS / "youtube_comments_limpio.csv", index=False)
    nodos.to_csv(DATOS_PROCESADOS / "nodos_red_bipartita.csv", index=False)
    aristas.to_csv(DATOS_PROCESADOS / "aristas_red_bipartita.csv", index=False)


def ejecutar_avance() -> dict[str, object]:
    videos_originales, comentarios_originales = cargar_datos()
    videos, comentarios = preparar_datos(videos_originales, comentarios_originales)
    comentarios = agregar_sentimiento_exploratorio(comentarios)
    integrados = integrar_datos(videos, comentarios)
    tablas = tablas_exploratorias(videos, comentarios)
    nodos, aristas = crear_tablas_red(videos, comentarios)
    red = crear_red(nodos, aristas)
    autores, videos_compartidos = audiencias_compartidas(comentarios)
    guardar_resultados(videos, comentarios, nodos, aristas)
    return {
        "videos_originales": videos_originales,
        "comentarios_originales": comentarios_originales,
        "videos": videos,
        "comentarios": comentarios,
        "integrados": integrados,
        "tablas": tablas,
        "nodos": nodos,
        "aristas": aristas,
        "red": red,
        "autores": autores,
        "videos_compartidos": videos_compartidos,
    }


if __name__ == "__main__":
    resultado = ejecutar_avance()
    print(f"Videos procesados: {len(resultado['videos'])}")
    print(f"Comentarios procesados: {len(resultado['comentarios'])}")
    print(f"Nodos de la red: {resultado['red'].number_of_nodes()}")
    print(f"Aristas de la red: {resultado['red'].number_of_edges()}")
