from __future__ import annotations

import base64
from pathlib import Path
from xml.sax.saxutils import escape
import nbformat as nbf
from nbclient import NotebookClient
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from scripts.analisis_final import ejecutar_final, FIGURAS
from scripts.crear_notebooks import guardar_notebooks, crear_notebook, markdown, codigo, RAIZ


def crear_final(r):
    t=r['final']; com=t['comunidades_resumen']; top=t['topologia']; cen=t['centralidad']
    c=r['comentarios']; videos=r['videos']; q=t['calidad_comunidades'].iloc[0]
    celdas=[markdown('# Laboratorio 6\n\n## Proyecciones, comunidades y conclusiones'),codigo('from pathlib import Path\nimport sys\nRAIZ = Path.cwd() if (Path.cwd() / "scripts").exists() else Path.cwd().parent\nsys.path.insert(0, str(RAIZ))\nfrom scripts.presentar_resultados import tabla, figura')]
    def texto(s): celdas.append(markdown(s))
    def tabla(nombre,limite=20): celdas.append(codigo(f'tabla("{nombre}", {limite})'))
    def figura(nombre): celdas.append(codigo(f'figura("{nombre}")'))
    texto('## 5. Proyecciones\n\nLa proyección autor-autor conecta autores que comentaron el mismo video y pesa cada relación por el número de videos compartidos. La proyección video-video conecta contenidos con autores en común y el peso cuenta autores únicos. Se usan vecinos distintos de la red bipartita, por lo que comentar varias veces no multiplica estos pesos. Se conservan todos los nodos, incluso los aislados. La primera representa coincidencia de participación y la segunda cruce de audiencias; ninguna representa amistad ni aprobación.')
    tabla('aristas_autores',10); tabla('aristas_videos',20)
    figura('red_autores'); figura('red_videos')
    texto(f'La proyección de autores tiene {int(top.iloc[1].aristas)} aristas frente a {int(top.iloc[2].aristas)} en videos. Cada video genera un grupo de autores completamente conectado, aunque estos no conversen. Esta expansión explica gran parte de la densidad y de los triángulos de la primera proyección.')
    texto('## 6. Topología y fragmentación\n\nLa densidad general divide las aristas entre todos los pares posibles. En la bipartita también se informa la densidad respecto a pares autor-video permitidos. La cohesión se define como la fracción de pares de nodos que pueden alcanzarse por algún camino. La transitividad es la proporción de ternas conectadas que cierran un triángulo; el clustering medio promedia el cierre local incluyendo los ceros. Las dos últimas medidas son estructuralmente cero en la bipartita, porque no admite triángulos; no significan ausencia total de cohesión.')
    tabla('topologia'); figura('distribucion_grados')
    texto(f'La bipartita contiene 625 nodos, 343 aristas y 284 componentes, con 286 nodos en la mayor. Sus 327 hojas muestran participación poco diversa. Hay 274 videos sin comentarios recolectados. La proyección de videos tiene 283 aislados: 274 sin cobertura y 9 con comentarios cuyos autores no aparecen en otros videos. Los 4 autores aislados en su proyección sí comentaron, pero no coinciden con otro autor en esos contenidos. La componente mayor de autores contiene 276 de 332 nodos; aun así, la transitividad de {top.iloc[1].transitividad:.3f} refleja especialmente los grupos completos creados por la proyección. Los grados altos de autores pueden deberse a comentar en un video concurrido y no a recurrencia.')
    tabla('periferia_videos',12); tabla('periferia_autores',12)
    texto('## 7. Comunidades\n\nSe elige la proyección de autores para agrupar patrones de participación compartida. Louvain optimiza modularidad mediante movimientos locales y agrupación de comunidades. Se usa el peso como cantidad de videos compartidos, resolución 1 y semilla 42. El método busca más conexiones internas que las esperadas bajo un modelo que conserva fuerzas; no garantiza el óptimo global ni grupos sociales reales. La proyección favorece los videos con muchos comentaristas. Las comunidades pequeñas pueden fusionarse por el límite de resolución. Los autores aislados se conservan como comunidades unitarias.')
    texto(f'Se detectaron {int(q.comunidades)} comunidades y modularidad {q.modularidad:.4f}. La tabla de semillas comprueba cuánto cambia este indicador con diferentes inicializaciones; una modularidad parecida no garantiza miembros idénticos.')
    tabla('calidad_comunidades'); tabla('comunidades_resumen',40); figura('comunidades_autores')
    texto('Cada color representa una comunidad en la figura y todos los nodos están incluidos. Los identificadores exactos se entregan en comunidades_autores.csv. Los colores ayudan a ubicar grupos; la tabla permite comparar incluso las comunidades unitarias. Los comentarios se asignan por la comunidad de su autor; un video puede aparecer en varias comunidades.')
    for f in com.head(3).itertuples():
        cv=t['comunidades_videos'].query('comunidad == @f.comunidad').sort_values('comentarios',ascending=False)
        nombres='; '.join(cv.title.astype(str))
        canales='; '.join(cv.channel_name.unique())
        autores=c[c.author_channel_id.isin(t['comunidades_autores'].query('comunidad == @f.comunidad').nodo_id.str.removeprefix('autor:'))].groupby('author_name').size().sort_values(ascending=False).head(3)
        texto(f'### Comunidad {f.comunidad}\n\nReúne {f.autores} autores, {f.videos} videos de {f.canales} canales y {f.comentarios} comentarios ({f.comentarios/f.autores:.2f} por autor). Los contenidos son {nombres}. Los canales son {canales}. Los autores con más comentarios son '+ '; '.join(f'{a}: {n}' for a,n in autores.items())+f'. Las palabras más frecuentes son {f.palabras}. El léxico asigna {f.positivo} positivos, {f.negativo} negativos y {f.neutral} neutrales. Estas palabras describen el contenido y no prueban que los miembros compartan una postura.')
    texto('Las tres comunidades principales reúnen conversaciones sobre diputados y recursos públicos, gobierno e infraestructura, y la USAC. Predomina la etiqueta neutral en las tres, pero la cobertura limitada del léxico impide afirmar que predomine una actitud neutral. El grupo gubernamental presenta más positivos que negativos reconocidos; las críticas políticas pueden quedar sin identificar por vocabulario no incluido, como plurales o expresiones locales.')
    texto('## 8. Nodos centrales y participantes puente\n\nSe calculan grado, fuerza, intermediación y cercanía en la red bipartita. El grado de un autor cuenta videos distintos y su fuerza cuenta comentarios; para un video cuentan autores distintos y comentarios, respectivamente. La intermediación normalizada mide la fracción de caminos mínimos que atraviesan el nodo. Se usan caminos sin pesos porque frecuencia no es distancia. La cercanía incorpora la corrección de Wasserman-Faust para componentes desconectadas, de modo que se reduce cuando pocos nodos son alcanzables. No se comparan autores y videos como si cumplieran la misma función.')
    tabla('centralidad',15); tabla('articulaciones',30)
    for tipo in ['autor','video']:
        f=cen[cen.tipo==tipo].iloc[0]
        texto(f'Entre los nodos de tipo {tipo}, {f.etiqueta} tiene la mayor intermediación ({f.intermediacion:.4f}), grado {f.grado} y fuerza {f.fuerza}. Al eliminarlo, el número de componentes aumenta en {f.aumento_componentes}. '+('Su papel de puente proviene de conectar contenidos y debe distinguirse del volumen de comentarios.' if tipo=='autor' else 'Su capacidad articuladora incluye desconectar autores que solo aparecen en ese video; no equivale necesariamente a conectar varios temas.'))
    texto(f'Hay {int(r["autores"].videos.gt(1).sum())} autores recurrentes en varios videos y {int(r["autores"].canales.gt(1).sum())} que cruzan canales. Se identifican {len(cen.query("tipo == \'autor\' and articulacion"))} autores y {len(cen.query("tipo == \'video\' and articulacion"))} videos de articulación. La columna aumento_componentes verifica cada eliminación por separado, manteniendo los demás nodos. La recurrencia por sí sola no basta para ser punto de articulación.')
    texto('## 9. Contenido y sentimiento\n\nSe usa el léxico español definido en analisis_avance.py, con términos positivos como excelente, gracias y apoyo y negativos como corrupción, mentira y violencia. Es una herramienta de reglas propia, transparente y reproducible, adecuada al idioma pero de cobertura limitada y sin validación externa de precisión. Se aplica a texto_original, no al texto sin stopwords. Se normalizan mayúsculas y tildes, se suman ocurrencias positivas y se restan negativas. No, nunca, jamás y sin invierten el signo dentro de las tres palabras previas sin cruzar la puntuación reconocida. Un puntaje mayor que cero es positivo, menor que cero negativo y cero neutral. No se usa VADER en inglés.\n\nLa regla no resuelve ironía, citas, sujeto del sentimiento, todos los límites de negación, emojis ni variantes morfológicas. Neutral incluye ausencia de coincidencias y cancelación de signos. Hablar de corrupción o muerte puede describir una noticia sin expresar una valoración. Estos conteos son etiquetas del método y no una medición validada de opinión política.')
    tabla('sentimiento_global'); figura('sentimiento'); tabla('sentimiento_video_id',30); tabla('sentimiento_channel_name',30); tabla('sentimiento_comunidad',30); tabla('sentimiento_source_query',30)
    texto('En el conjunto completo, 79 comentarios son positivos, 38 negativos y 289 neutrales según el léxico. Quorum reúne 256 comentarios, con 19.14 por ciento positivos y 10.16 por ciento negativos; el canal del Gobierno reúne 70, con 25.71 por ciento positivos y 12.86 por ciento negativos. La diferencia de porcentajes no demuestra una mejor recepción del Gobierno: cambian los videos, las personas y la cobertura. Los dos comentarios de Noticiero Guatevisión no permiten una comparación estable aunque uno tenga etiqueta negativa.\n\nSe muestran los tamaños y porcentajes por grupo. Solo se comparan descriptivamente grupos con al menos 10 comentarios; ese umbral práctico no garantiza representatividad. source_query se usa como tema de recolección, no como clasificación temática validada. No se realizan pruebas de diferencias poblacionales, ya que los comentarios no forman una muestra aleatoria y los autores pueden repetir participación.')
    texto('## 10. Interpretación y conclusiones\n\nLa participación recolectada se concentra en pocos contenidos. La red de autores parece densa porque cada video conecta a sus comentaristas entre sí, mientras la proyección de videos muestra pocos cruces. Los autores puente son escasos y sostienen conexiones que pueden desaparecer al retirar un solo participante. Las comunidades reflejan principalmente los contenidos comentados, con asuntos de gasto público, gobierno e infraestructura y universidad entre los grupos mayores. El sentimiento aporta una descripción léxica limitada, que no permite atribuir posturas a todas las audiencias.\n\nLa cobertura alcanza solo 19 de 293 videos y no equivale al total de comentarios publicados. La selección por consultas y canales limita qué temas aparecen. Las fechas relativas no permiten una cronología exacta. Las visualizaciones, respuestas y me gusta son conteos del momento de recolección, sujetos a cambio. No hay relaciones explícitas entre autores ni identificación de quienes respondieron. La concentración en pocos videos puede mezclar actividad y diferencias de recolección. Un video sin comentarios en estos archivos no necesariamente carece de ellos en YouTube.\n\nLos conteos de nodos y palabras son descripciones. Las correlaciones entre visualizaciones y comentarios y los cruces de audiencia son asociaciones observadas. No se establece causalidad ni se hace inferencia hacia todos los usuarios de YouTube o la población de Guatemala. Redes, contenido y sentimiento deben leerse juntos con estas restricciones.')
    texto('## Referencias\n\nNetworkX. Louvain communities. https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html\n\nNetworkX. Betweenness centrality. https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html\n\nNetworkX. Closeness centrality. https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html\n\nRepositorio y espacio colaborativo de código: https://github.com/Ninaswiftie09/Data_Science/tree/Laboratorio6')
    nbf.write(crear_notebook(celdas), RAIZ/'notebooks/03.Resultados_Finales.ipynb')


def informe():
    estilos=getSampleStyleSheet()
    estilos['BodyText'].fontSize=9; estilos['BodyText'].leading=13
    destino=RAIZ/'output/pdf'; destino.mkdir(parents=True,exist_ok=True)
    story=[Paragraph('Laboratorio 6. Análisis de redes sociales en YouTube',estilos['Title']),Paragraph('Nina Nájera Marakovits, 231088',estilos['BodyText']),Spacer(1,15)]
    temporal=RAIZ/'tmp/pdf_images'; temporal.mkdir(parents=True,exist_ok=True)
    numero=0
    for archivo in sorted((RAIZ/'notebooks').glob('*.ipynb')):
        nb=nbf.read(archivo,as_version=4)
        if numero: story.append(PageBreak())
        for cell in nb.cells:
            if cell.cell_type=='markdown':
                for bloque in cell.source.split('\n\n'):
                    if not bloque.strip(): continue
                    titulo=bloque.startswith('#')
                    texto=bloque.lstrip('# ').replace('\n',' ')
                    story.append(Paragraph(escape(texto),estilos['Heading2'] if titulo else estilos['BodyText']))
                    if not titulo:
                        story.append(Spacer(1,5))
            else:
                for out in cell.get('outputs',[]):
                    data=out.get('data',{})
                    if 'image/png' in data:
                        numero+=1; p=temporal/f'{numero}.png'; p.write_bytes(base64.b64decode(data['image/png']))
                        from PIL import Image as PILImage
                        w,h=PILImage.open(p).size; scale=min(470/w,490/h)
                        story.append(Image(str(p),width=w*scale,height=h*scale)); story.append(Spacer(1,8))
                    elif 'text/html' in data:
                        from bs4 import BeautifulSoup
                        from reportlab.platypus import LongTable, TableStyle
                        soup=BeautifulSoup(data['text/html'],'html.parser')
                        rows=[[x.get_text(' ',strip=True) for x in row.find_all(['td','th'])] for row in soup.find_all('tr')]
                        if rows:
                            count=max(map(len,rows)); width=470/count
                            small=estilos['BodyText'].clone('small'); small.fontSize=6; small.leading=8
                            cells=[[Paragraph(escape(v),small) for v in row]+['']*(count-len(row)) for row in rows]
                            table=LongTable(cells,colWidths=[width]*count,repeatRows=1,hAlign='LEFT')
                            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),'#e8eef3'),('GRID',(0,0),(-1,-1),0.2,'#cccccc')]))
                            story.append(table); story.append(Spacer(1,8))
                    elif 'text/plain' in data:
                        for line in data['text/plain'].splitlines():
                            story.append(Paragraph(escape(line),estilos['BodyText']))
    def pagina(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.drawRightString(550,25,str(doc.page))
    SimpleDocTemplate(str(destino/'Laboratorio_6_YouTube.pdf'),rightMargin=60,leftMargin=60,topMargin=45,bottomMargin=45).build(story,onFirstPage=pagina,onLaterPages=pagina)


def main():
    guardar_notebooks(); crear_final(ejecutar_final())
    for archivo in sorted((RAIZ/'notebooks').glob('*.ipynb')):
        nb=nbf.read(archivo,as_version=4)
        NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(RAIZ)}}).execute()
        nbf.write(nb,archivo)
        print(f'Ejecutado: {archivo.name}',flush=True)
    informe()


if __name__=='__main__':
    main()

