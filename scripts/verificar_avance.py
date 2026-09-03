from pathlib import Path

import networkx as nx

from scripts.analisis_avance import ejecutar_avance


def verificar() -> None:
    resultado = ejecutar_avance()
    videos_originales = resultado["videos_originales"]
    comentarios_originales = resultado["comentarios_originales"]
    videos = resultado["videos"]
    comentarios = resultado["comentarios"]
    integrados = resultado["integrados"]
    nodos = resultado["nodos"]
    aristas = resultado["aristas"]
    red = resultado["red"]

    assert videos_originales.shape == (293, 20)
    assert comentarios_originales.shape == (406, 17)
    assert videos["video_id"].is_unique
    assert comentarios["comment_id"].is_unique
    assert integrados["estado_union"].eq("both").all()
    assert comentarios["texto_original"].equals(comentarios_originales["text"])
    assert len(comentarios) == len(comentarios_originales)
    assert aristas["peso"].sum() == len(comentarios)
    assert red.number_of_nodes() == len(nodos)
    assert red.number_of_edges() == len(aristas)
    assert nx.algorithms.bipartite.is_bipartite(red)

    esperados = [
        "youtube_videos_limpio.csv",
        "youtube_comments_limpio.csv",
        "nodos_red_bipartita.csv",
        "aristas_red_bipartita.csv",
    ]
    carpeta = Path(__file__).resolve().parents[1] / "data" / "processed"
    assert all((carpeta / nombre).exists() for nombre in esperados)

    print("Todas las verificaciones terminaron correctamente")


if __name__ == "__main__":
    verificar()
