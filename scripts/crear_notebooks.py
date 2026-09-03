from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import networkx as nx

from scripts.analisis_avance import (
    calcular_concentracion,
    ejecutar_avance,
    popularidad_participacion,
    puentes_observados,
    resumen_componentes,
)


RAIZ = Path(__file__).resolve().parents[1]
CARPETA_NOTEBOOKS = RAIZ / "notebooks"
REPOSITORIO = "https://github.com/Ninaswiftie09/Data_Science/tree/Laboratorio6"


def markdown(texto: str):
    return nbf.v4.new_markdown_cell(texto.strip())


def codigo(texto: str):
    return nbf.v4.new_code_cell(texto.strip())


def crear_notebook(celdas: list) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook(cells=celdas)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3"}
    return notebook


def preparar_valores() -> dict[str, object]:
    resultado = ejecutar_avance()
    videos = resultado["videos"]
    comentarios = resultado["comentarios"]
    tablas = resultado["tablas"]
    red = resultado["red"]
    autores = resultado["autores"]
    pares = resultado["videos_compartidos"]

    concentracion_videos = calcular_concentracion(
        tablas["comentarios_video"], "comentarios", "videos"
    )
    concentracion_canales = calcular_concentracion(
        tablas["comentarios_canal"], "comentarios", "canales"
    )
    correlaciones = popularidad_participacion(tablas["comentarios_video"])
    top_video = tablas["comentarios_video"].iloc[0]
    top_canal = tablas["comentarios_canal"].iloc[0]
    top_vistas = tablas["comentarios_video"].sort_values("view_count", ascending=False).iloc[0]
    puentes = puentes_observados(red)
    componentes = resumen_componentes(red, comentarios)
    sentimientos = comentarios["sentimiento"].value_counts()
    autores_recurrentes = int(autores["videos"].gt(1).sum())
    autores_multicanal = int(autores["canales"].gt(1).sum())
    cobertura = 100 * comentarios["video_id"].nunique() / videos["video_id"].nunique()

    if pares.empty:
        par_texto = "No se encontraron pares de videos con autores compartidos."
    else:
        primero = pares.iloc[0]
        titulos = videos.set_index("video_id")["title"]
        par_texto = (
            f"El par con mayor cruce fue {titulos[primero['video_id_1']]} y "
            f"{titulos[primero['video_id_2']]}, con {primero['autores_compartidos']} autores compartidos."
        )

    nombres_puentes = ", ".join(
        puentes.loc[puentes["punto_de_articulacion"], "autor"].head(5).astype(str)
    )
    resumen_comunidades = " ".join(
        f"La componente {fila.componente} tiene {fila.autores} autores, {fila.videos} videos y {fila.comentarios} comentarios. Sus palabras frecuentes son {fila.palabras_frecuentes}. El sentimiento principal es {fila.sentimiento_principal} con {fila.porcentaje_sentimiento_principal} por ciento."
        for fila in componentes.itertuples()
    )

    return {
        "resultado": resultado,
        "concentracion_videos": concentracion_videos,
        "concentracion_canales": concentracion_canales,
        "correlaciones": correlaciones,
        "top_video": top_video,
        "top_canal": top_canal,
        "top_vistas": top_vistas,
        "puentes": puentes,
        "componentes": componentes,
        "sentimientos": sentimientos,
        "autores_recurrentes": autores_recurrentes,
        "autores_multicanal": autores_multicanal,
        "cobertura": cobertura,
        "par_texto": par_texto,
        "nombres_puentes": nombres_puentes,
        "resumen_comunidades": resumen_comunidades,
    }


def notebook_exploratorio(valores: dict[str, object]) -> nbf.NotebookNode:
    resultado = valores["resultado"]
    videos = resultado["videos"]
    comentarios = resultado["comentarios"]
    top_video = valores["top_video"]
    top_canal = valores["top_canal"]
    top_vistas = valores["top_vistas"]
    correlaciones = valores["correlaciones"].set_index("medida")["valor"]
    porcentaje_video = valores["concentracion_videos"].iloc[0]["porcentaje_comentarios"]
    porcentaje_canal = valores["concentracion_canales"].iloc[0]["porcentaje_comentarios"]
    sentimientos = valores["sentimientos"]
    total_respuestas = int(comentarios["reply_count"].sum())
    con_respuestas = int(comentarios["reply_count"].gt(0).sum())
    total_me_gusta = int(comentarios["like_count"].sum())
    autores_un_video = int(
        resultado["autores"]["videos"].eq(1).sum()
    )

    celdas = [
        markdown(
            f"""
# Laboratorio 6

## Avance de análisis exploratorio

Repositorio

{REPOSITORIO}

Este avance cubre la carga, la integración, la limpieza y el análisis exploratorio de los datos. También deja preparada la información que se usa para construir la red bipartita.
"""
        ),
        codigo(
            """
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

RAIZ = Path.cwd()
if not (RAIZ / "scripts").exists():
    RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ))

from scripts.analisis_avance import (
    calcular_concentracion,
    diagnostico_atipicos,
    diagnostico_variables,
    efecto_limpieza,
    ejecutar_avance,
    graficar_barras,
    graficar_popularidad,
    popularidad_participacion,
    revisar_consistencia,
    resumen_componentes,
    resumen_numerico,
)

resultado = ejecutar_avance()
videos_originales = resultado["videos_originales"]
comentarios_originales = resultado["comentarios_originales"]
videos = resultado["videos"]
comentarios = resultado["comentarios"]
integrados = resultado["integrados"]
tablas = resultado["tablas"]
pd.set_option("display.max_colwidth", 80)
"""
        ),
        markdown(
            """
## 1. Carga, comprensión e integración

La unidad de observación de youtube_videos.csv es un video. Su llave primaria es video_id.

La unidad de observación de youtube_comments.csv es un comentario principal. Su llave primaria es comment_id y video_id funciona como llave foránea.

Un canal publica videos. Cada video tiene una categoría y fue recuperado por una consulta de búsqueda. Un autor publica un comentario en un video. El autor se identifica con author_channel_id. El canal dueño del video se identifica con channel_id. Los nombres y handles solo se usan como etiquetas visibles.
"""
        ),
        codigo(
            """
pd.DataFrame({
    "archivo": ["youtube_videos.csv", "youtube_comments.csv"],
    "filas": [videos_originales.shape[0], comentarios_originales.shape[0]],
    "columnas": [videos_originales.shape[1], comentarios_originales.shape[1]],
    "unidad": ["video", "comentario principal"],
    "llave_primaria": ["video_id", "comment_id"],
})
"""
        ),
        codigo(
            """
integrados["estado_union"].value_counts().rename_axis("estado").reset_index(name="comentarios")
"""
        ),
        markdown(
            f"Los {len(comentarios)} comentarios se asociaron con un video. No quedó ningún comentario sin correspondencia."
        ),
        markdown(
            """
## 2. Calidad, limpieza y preprocesamiento

El diagnóstico se hizo sobre los archivos originales. Los valores atípicos se identificaron con el rango intercuartílico. En variables de conteo, un valor atípico no se elimina de forma automática porque puede representar un video o comentario realmente popular.
"""
        ),
        codigo(
            """
display(diagnostico_variables(videos_originales))
display(diagnostico_variables(comentarios_originales))
"""
        ),
        codigo(
            """
display(diagnostico_atipicos(videos))
display(diagnostico_atipicos(comentarios[["like_count", "reply_count"]]))
display(revisar_consistencia(videos, comentarios))
"""
        ),
        markdown(
            """
### Variables que requieren cuidado

- viewer_rating está vacía en todos los comentarios y no se usa.
- is_pinned es constante y no ayuda a comparar comentarios.
- owner_handle repite channel_handle y upload_date coincide con publish_date.
- video_title, channel_name y channel_id aparecen en comentarios, pero se verifican contra la tabla de videos después de la unión.
- published_time y published_text son fechas relativas y no se convierten en fechas exactas.
- source_query explica cómo se encontró el contenido y no se interpreta como el tema definitivo.
- reply_count indica cuántas respuestas recibió un comentario, pero no dice quién respondió. No se usa para crear aristas entre autores.
- Los nombres y handles pueden cambiar. Los identificadores se mantienen como llaves.
- Un registro de videos contiene un carácter de reemplazo. Se conserva porque no se puede reconstruir con seguridad el carácter original.
"""
        ),
        markdown(
            """
### Conversión de conteos

Los espacios y separadores se eliminan antes de convertir los conteos. Las abreviaturas mil, k, millón y m se convierten con su multiplicador. Los valores vacíos o sin números se dejan como faltantes. Para los comentarios, los valores vacíos de me gusta se interpretan como cero porque el archivo usa el vacío para ese caso.
"""
        ),
        codigo(
            """
comparacion_vistas = videos[["view_count_text", "view_count", "view_count_desde_texto"]].copy()
pd.DataFrame({
    "medida": [
        "conteos de vistas convertidos",
        "conteos de vistas sin texto",
        "diferencias entre conteo numérico y texto",
        "conteos de me gusta convertidos",
    ],
    "cantidad": [
        comparacion_vistas["view_count_desde_texto"].notna().sum(),
        comparacion_vistas["view_count_desde_texto"].isna().sum(),
        comparacion_vistas.dropna()["view_count"].ne(comparacion_vistas.dropna()["view_count_desde_texto"]).sum(),
        comentarios["like_count"].notna().sum(),
    ],
})
"""
        ),
        markdown(
            """
### Limpieza de texto

Se conserva texto_original para auditoría y para análisis posteriores. texto_limpio se pasa a minúsculas. Se eliminan URL, menciones, puntuación, números, emojis y palabras vacías en español. En los hashtags se elimina el símbolo y se conserva la palabra. No se aplicó lematización porque el proyecto todavía no incluye un modelo de español validado y una regla simple podría cambiar nombres o palabras de manera incorrecta.

No se eliminaron registros por el resultado de la limpieza. Los textos vacíos se conservan para que el número de comentarios siga siendo auditable.
"""
        ),
        codigo("efecto_limpieza(comentarios)"),
        markdown(
            """
## 3. Análisis exploratorio

Los conteos siguientes describen la muestra disponible. No representan a toda la población de usuarios de YouTube ni a toda Guatemala.
"""
        ),
        codigo(
            """
pd.DataFrame({
    "medida": ["videos", "canales", "comentarios", "autores", "videos con comentarios"],
    "cantidad": [
        videos["video_id"].nunique(),
        videos["channel_id"].nunique(),
        comentarios["comment_id"].nunique(),
        comentarios["author_channel_id"].nunique(),
        comentarios["video_id"].nunique(),
    ],
})
"""
        ),
        codigo(
            """
display(tablas["videos_canal"].head(10))
display(tablas["comentarios_video"].head(10))
display(tablas["comentarios_canal"].head(10))
"""
        ),
        codigo(
            """
display(resumen_numerico(videos["view_count"], "visualizaciones"))
display(resumen_numerico(comentarios["reply_count"], "respuestas"))
display(resumen_numerico(comentarios["like_count"], "me gusta"))
"""
        ),
        codigo(
            """
display(tablas["categorias"])
display(tablas["consultas"].head(15))
display(tablas["hashtags"])
display(tablas["palabras"])
display(tablas["bigramas"])
"""
        ),
        codigo(
            """
graficar_barras(tablas["comentarios_video"], "title", "comentarios", "Videos con más comentarios observados")
plt.show()
graficar_barras(tablas["comentarios_canal"], "channel_name", "comentarios", "Canales con más comentarios observados", cantidad=8, color="#E07A5F")
plt.show()
graficar_barras(tablas["palabras"], "palabra", "frecuencia", "Palabras más frecuentes", cantidad=15, color="#4D908E")
plt.show()
graficar_barras(tablas["bigramas"], "bigrama", "frecuencia", "Bigramas más frecuentes", cantidad=15, color="#8E6C8A")
plt.show()
"""
        ),
        markdown("### Concentración de la participación"),
        codigo(
            """
concentracion_videos = calcular_concentracion(tablas["comentarios_video"], "comentarios", "videos")
concentracion_canales = calcular_concentracion(tablas["comentarios_canal"], "comentarios", "canales")
display(concentracion_videos)
display(concentracion_canales)
"""
        ),
        markdown(
            f"El video con más actividad reúne {porcentaje_video} por ciento de los comentarios. El canal con más actividad reúne {porcentaje_canal} por ciento. La participación está muy concentrada."
        ),
        markdown("### Popularidad y participación"),
        codigo(
            """
display(popularidad_participacion(tablas["comentarios_video"]))
graficar_popularidad(tablas["comentarios_video"])
plt.show()
"""
        ),
        markdown(
            f"La correlación de Pearson entre visualizaciones y comentarios es {correlaciones['correlacion de Pearson']}. La correlación de Spearman es {correlaciones['correlacion de Spearman']}. Esto muestra que el orden general puede ser parecido, pero la relación lineal es débil por la presencia de diferencias grandes entre videos. Solo se comparan los {int(correlaciones['videos con comentarios'])} videos con comentarios recolectados. Las visualizaciones y los comentarios son conteos observados en momentos y coberturas distintas."
        ),
        markdown(
            f"""
### Respuestas a las preguntas obligatorias

#### Qué videos y canales concentran la mayor participación observada

El video con más comentarios es {top_video['title']}, del canal {top_video['channel_name']}. Tiene {int(top_video['comentarios'])} comentarios y {int(top_video['autores_unicos'])} autores únicos. El canal con más comentarios es {top_canal['channel_name']}, con {int(top_canal['comentarios'])} comentarios y {int(top_canal['autores_unicos'])} autores únicos.

#### Existen audiencias compartidas

Hay {valores['autores_recurrentes']} autores que aparecen en más de un video y {valores['autores_multicanal']} autores que aparecen en más de un canal. {valores['par_texto']} Estos cruces muestran coparticipación observada, no amistad ni conversación directa.

#### Qué autores funcionan como puentes

Los primeros autores que aparecen como puntos de articulación son {valores['nombres_puentes']}. Si se quita uno de estos nodos, aumenta la cantidad de partes desconectadas en la red observada. Esto no prueba influencia fuera de la muestra.

#### Qué temas y sentimientos caracterizan las principales comunidades

En este avance se usan componentes conectadas como una descripción preliminar. Todavía no se aplica un algoritmo de comunidades. {valores['resumen_comunidades']}

El análisis de sentimiento de este avance usa un léxico pequeño y transparente en español. Clasifica {int(sentimientos.get('positivo', 0))} comentarios como positivos, {int(sentimientos.get('negativo', 0))} como negativos y {int(sentimientos.get('neutral', 0))} como neutrales. Es un resultado preliminar y no detecta bien ironía, contexto ni negaciones.

#### La visibilidad coincide con la participación observada

No coincide de forma directa. El video con más visualizaciones dentro de los videos comentados es {top_vistas['title']}, con {int(top_vistas['view_count'])} visualizaciones y {int(top_vistas['comentarios'])} comentarios. El video con más comentarios tiene {int(top_video['view_count'])} visualizaciones y {int(top_video['comentarios'])} comentarios.

#### Qué conclusiones están limitadas por la recolección

Los comentarios solo cubren una parte de los videos. La selección depende de consultas y canales usados durante la recolección. Las fechas relativas no permiten reconstruir el momento exacto. Los conteos cambian con el tiempo. No se conoce quién respondió a quién. La concentración en pocos videos puede reflejar tanto participación real como diferencias en la cobertura.
"""
        ),
        markdown(
            f"""
### Preguntas adicionales

#### Qué parte de los videos tiene comentarios recolectados

Los comentarios cubren {comentarios['video_id'].nunique()} de {videos['video_id'].nunique()} videos. Esto equivale a {valores['cobertura']:.2f} por ciento. El análisis de participación se limita a esa parte de la muestra.

#### La mayoría de autores participa en varios videos

No. {autores_un_video} de {comentarios['author_channel_id'].nunique()} autores aparecen en un solo video. Solo {valores['autores_recurrentes']} aparecen en más de uno. La audiencia observada es principalmente local a cada video.

#### Cuánta interacción adicional muestran las respuestas y los me gusta

Se observaron {total_respuestas} respuestas distribuidas en {con_respuestas} comentarios principales. Los comentarios recibieron {total_me_gusta} me gusta en total. reply_count no permite saber quién respondió y por eso no agrega relaciones entre autores.
"""
        ),
        markdown(
            """
## Cierre del avance

La muestra tiene una participación muy concentrada en pocos videos y canales. La mayoría de autores aparece en un solo video. Existen algunos cruces de audiencia y puntos de articulación que se revisan con más detalle en el notebook de la red bipartita. Todas las conclusiones describen únicamente los datos recolectados.
"""
        ),
    ]
    return crear_notebook(celdas)


def notebook_red(valores: dict[str, object]) -> nbf.NotebookNode:
    resultado = valores["resultado"]
    red = resultado["red"]
    nodos = resultado["nodos"]
    aristas = resultado["aristas"]
    autores = resultado["autores"]
    puentes = valores["puentes"]
    componentes = valores["componentes"]
    puntos = int(puentes["punto_de_articulacion"].sum())

    celdas = [
        markdown(
            f"""
# Laboratorio 6

## Red bipartita autor video

Repositorio

{REPOSITORIO}

Este notebook presenta la actividad 4 del avance.
"""
        ),
        codigo(
            """
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

RAIZ = Path.cwd()
if not (RAIZ / "scripts").exists():
    RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ))

from scripts.analisis_avance import (
    ejecutar_avance,
    graficar_red_completa,
    puentes_observados,
    resumen_componentes,
    resumen_red,
)

resultado = ejecutar_avance()
nodos = resultado["nodos"]
aristas = resultado["aristas"]
red = resultado["red"]
autores = resultado["autores"]
videos_compartidos = resultado["videos_compartidos"]
comentarios = resultado["comentarios"]
pd.set_option("display.max_colwidth", 80)
"""
        ),
        markdown(
            """
## 4.1 Construcción de la red

La red es no dirigida y tiene dos tipos de nodos. Un tipo representa autores y el otro representa videos. Los autores se identifican con author_channel_id y los videos con video_id.

Una arista indica que un autor publicó al menos un comentario principal en un video. Su peso es la cantidad de comentarios de ese autor en ese video.

La arista no demuestra amistad, conversación directa, aprobación ni respuesta entre usuarios.
"""
        ),
        codigo(
            """
pd.DataFrame({
    "comprobacion": [
        "la red es bipartita",
        "suma de pesos",
        "comentarios originales",
        "aristas únicas autor video",
    ],
    "valor": [
        nx.algorithms.bipartite.is_bipartite(red),
        aristas["peso"].sum(),
        len(comentarios),
        len(aristas),
    ],
})
"""
        ),
        markdown(
            """
## 4.2 Tabla de nodos

La tabla completa se guarda en data/processed/nodos_red_bipartita.csv. Incluye el tipo de nodo, su etiqueta visible, identificadores, actividad y atributos del video cuando corresponden.
"""
        ),
        codigo(
            """
display(nodos.groupby("tipo").size().rename("cantidad").reset_index())
display(nodos.head(15))
"""
        ),
        markdown(
            """
## 4.3 Tabla de aristas

La tabla completa se guarda en data/processed/aristas_red_bipartita.csv. Cada fila es una combinación única de autor y video.
"""
        ),
        codigo("aristas.head(20)"),
        markdown("## 4.4 Visualización de la red completa"),
        codigo(
            """
graficar_red_completa(red)
plt.show()
"""
        ),
        markdown(
            """
Todos los nodos y todas las aristas se mantienen en la figura. Los autores se muestran en azul y los videos en naranja. Los nodos de video crecen según su grado. La posición solo ayuda a ver la estructura y no representa distancia geográfica ni cercanía social.
"""
        ),
        markdown("## Resumen de la estructura observada"),
        codigo("resumen_red(red)"),
        markdown(
            f"La red tiene {red.number_of_nodes()} nodos y {red.number_of_edges()} aristas. Incluye {int((nodos['tipo'] == 'autor').sum())} autores y {int((nodos['tipo'] == 'video').sum())} videos. Hay {len(list(nx.connected_components(red)))} componentes conectadas. La componente mayor contiene {len(max(nx.connected_components(red), key=len))} nodos."
        ),
        markdown("### Participantes recurrentes y cruces de audiencia"),
        codigo(
            """
display(autores.head(15))
display(videos_compartidos.head(15))
"""
        ),
        markdown(
            f"Hay {int(autores['videos'].gt(1).sum())} autores que comentaron en más de un video y {int(autores['canales'].gt(1).sum())} que aparecen en más de un canal. Estos autores conectan contenidos dentro de la muestra, pero eso no demuestra una relación personal con otros autores."
        ),
        markdown("### Puntos de articulación"),
        codigo("puentes_observados(red)"),
        markdown(
            f"En la tabla mostrada hay {puntos} autores que son puntos de articulación. Quitarlos aumentaría la fragmentación de la red observada. Esta propiedad depende de la cobertura de comentarios y no debe interpretarse como influencia general en YouTube."
        ),
        markdown("### Componentes principales"),
        codigo("resumen_componentes(red, comentarios)"),
        markdown(
            """
Las componentes muestran grupos separados por falta de autores compartidos en los datos. Un grupo separado puede reflejar una audiencia distinta, pero también puede aparecer porque la recolección no incluyó suficientes comentarios. La detección formal de comunidades se realizará en la siguiente parte del laboratorio.
"""
        ),
        markdown(
            """
## Conclusión del avance de red

La red confirma que la mayor parte de la participación observada ocurre alrededor de pocos videos y que casi todos los autores aparecen en un solo contenido. Los pocos autores recurrentes ayudan a unir videos o canales. La estructura solo representa coparticipación en los comentarios recolectados.
"""
        ),
    ]
    return crear_notebook(celdas)


def guardar_notebooks() -> None:
    CARPETA_NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    valores = preparar_valores()
    nbf.write(
        notebook_exploratorio(valores),
        CARPETA_NOTEBOOKS / "01.Analisis_Exploratorio.ipynb",
    )
    nbf.write(
        notebook_red(valores),
        CARPETA_NOTEBOOKS / "02.Red_Bipartita.ipynb",
    )


if __name__ == "__main__":
    guardar_notebooks()
    print("Notebooks creados")
