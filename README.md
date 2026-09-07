# Laboratorio 6

## Integrante

Nina Nájera Marakovits, 231088.

## Repositorio y espacio colaborativo

El repositorio de GitHub se utiliza como espacio colaborativo de código y registro de versiones:

https://github.com/Ninaswiftie09/Data_Science/tree/Laboratorio6

## Entrega

El informe con resultados, visualizaciones, interpretación y conclusiones está en output/pdf/Laboratorio_6_YouTube.pdf.

Los notebooks ejecutados contienen las actividades 1 a 10:

- notebooks/01.Analisis_Exploratorio.ipynb: integración, calidad, limpieza, exploración y preguntas.
- notebooks/02.Red_Bipartita.ipynb: construcción, tablas y visualización completa.
- notebooks/03.Resultados_Finales.ipynb: proyecciones, topología, comunidades, centralidades, sentimiento y conclusiones.

Los archivos originales se conservan en data/raw. Las tablas reproducibles están en data/processed y las figuras finales en output/figures. El código de limpieza y análisis está en scripts.

## Instalación

Se requiere Python 3.11 o posterior. La entrega fue ejecutada con Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Ejecución

Desde la raíz del repositorio se procesan los datos y se calculan las tablas, redes y figuras:

```bash
python -m scripts.analisis_final
```

Para consultar los resultados y ejecutar nuevamente las celdas de los notebooks:

```bash
jupyter notebook
```

Los notebooks se ejecutan en orden: 01, 02 y 03. Las tablas y figuras del tercero se actualizan con el comando de análisis anterior. El informe PDF se entrega como documento final.

scripts/analisis_avance.py contiene la carga, limpieza, exploración y construcción de la red bipartita. scripts/analisis_final.py contiene las proyecciones, métricas, comunidades, centralidades y tablas de sentimiento.

## Decisiones metodológicas

Los ID se mantienen como llaves; los nombres y handles son etiquetas. La bipartita conserva los 293 videos, incluidos 274 sin comentarios recolectados. La suma de pesos es 406 comentarios; las proyecciones cuentan vecinos distintos y no multiplican relaciones por comentarios repetidos.

Louvain se aplica a la proyección ponderada de autores con resolución 1 y semilla 42. Se comparan tres semillas. La intermediación y cercanía se calculan sobre caminos sin pesos en la bipartita, porque los pesos representan frecuencia y no distancia.

El sentimiento utiliza el léxico español del proyecto sobre el texto original con una regla local de negación. Es una clasificación descriptiva sin precisión validada; neutral también puede significar ausencia de vocabulario reconocido. Los grupos con menos de 10 comentarios se muestran pero no se comparan como evidencia estable. La consulta de búsqueda describe la selección y no constituye una etiqueta temática validada.

reply_count no identifica relaciones entre autores. Los videos sin cobertura no se interpretan como videos sin audiencia. Las conclusiones se limitan a la muestra y no establecen causalidad.

## Fuentes metodológicas

- https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html
- https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html
- https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
