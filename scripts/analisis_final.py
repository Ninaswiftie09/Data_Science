from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scripts.analisis_avance import ejecutar_avance, DATOS_PROCESADOS, RAIZ, frecuencias_texto

FIGURAS = RAIZ / 'output' / 'figures'


def topologia(g, nombre):
    n, m = len(g), g.number_of_edges()
    componentes = list(nx.connected_components(g))
    pares = sum(len(c)*(len(c)-1) for c in componentes)
    return dict(red=nombre, nodos=n, aristas=m, densidad=nx.density(g),
                grado_medio=2*m/n, componentes=len(componentes),
                componente_mayor=max(map(len, componentes)), aislados=nx.number_of_isolates(g),
                hojas=sum(d == 1 for _, d in g.degree()),
                cohesion_pares=pares/(n*(n-1)), transitividad=nx.transitivity(g),
                clustering_medio=nx.average_clustering(g))


def dibujar(g, nombre, comunidades=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    componentes = sorted(nx.connected_components(g), key=len, reverse=True)
    activos = [c for c in componentes if len(c)>1]
    aislados = [next(iter(c)) for c in componentes if len(c)==1]
    pos = {}
    for i, componente in enumerate(activos):
        local = nx.spring_layout(g.subgraph(sorted(componente)), seed=42, iterations=100, weight='weight')
        for n, xy in local.items():
            pos[n] = (xy[0] + (i % 3)*3, xy[1] - (i // 3)*3)
    for i,n in enumerate(aislados):
        pos[n] = ((i % 40)*0.19-1, -((len(activos)+2)//3)*3-0.2*(i//40))
    colores = [comunidades[n] if comunidades else (0 if str(n).startswith('autor:') else 1) for n in g]
    nx.draw_networkx(g, pos, ax=ax, with_labels=False, node_color=colores,
                     cmap=plt.get_cmap('turbo'), node_size=[12+3*g.degree(n)**0.5 for n in g],
                     width=0.3, edge_color='#999999', alpha=0.7)
    ax.set_title(nombre.replace('_', ' '))
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(FIGURAS / f'{nombre}.png', dpi=160)
    plt.close(fig)


def ejecutar_final():
    r = ejecutar_avance()
    g = r['red']
    autores = {n for n, d in g.nodes(data=True) if d['tipo'] == 'autor'}
    videos = set(g)-autores
    proyecciones = { 'autores': nx.bipartite.weighted_projected_graph(g, sorted(autores)),
                     'videos': nx.bipartite.weighted_projected_graph(g, sorted(videos)) }
    redes = {'bipartita': g, **proyecciones}
    tablas = {}
    tablas['topologia'] = pd.DataFrame([topologia(red, nombre) for nombre, red in redes.items()])
    tablas['topologia']['densidad_bipartita'] = [g.number_of_edges()/(len(autores)*len(videos)), float('nan'), float('nan')]
    for nombre, red in redes.items():
        tablas[f'grados_{nombre}'] = pd.DataFrame(sorted(Counter(dict(red.degree()).values()).items()), columns=['grado','nodos'])
        tablas[f'periferia_{nombre}'] = pd.DataFrame([dict(nodo_id=n, etiqueta=g.nodes[n]['etiqueta'], grado=d, estado='aislado observado' if d==0 else 'hoja') for n,d in red.degree() if d<=1])
        if nombre != 'bipartita':
            tablas[f'aristas_{nombre}'] = nx.to_pandas_edgelist(red).rename(columns={'source':'origen','target':'destino','weight':'peso'})
    particion = sorted(nx.community.louvain_communities(proyecciones['autores'], weight='weight', resolution=1, seed=42), key=lambda c: (-len(c), min(c)))
    asignacion = {n:i for i,c in enumerate(particion,1) for n in c}
    tablas['comunidades_autores'] = pd.DataFrame([dict(nodo_id=n, comunidad=i) for n,i in asignacion.items()])
    c = r['comentarios'].copy()
    c['comunidad'] = ('autor:'+c.author_channel_id).map(asignacion)
    tablas['comentarios_sentimiento'] = c
    filas=[]
    for i, grupo in c.groupby('comunidad'):
        palabras=frecuencias_texto(grupo.texto_limpio, limite=8)
        filas.append(dict(comunidad=i, autores=grupo.author_channel_id.nunique(), videos=grupo.video_id.nunique(), canales=grupo.channel_id.nunique(), comentarios=len(grupo), palabras=', '.join(palabras.palabra), positivo=int((grupo.sentimiento=='positivo').sum()), negativo=int((grupo.sentimiento=='negativo').sum()), neutral=int((grupo.sentimiento=='neutral').sum())))
    tablas['comunidades_resumen']=pd.DataFrame(filas).sort_values('autores',ascending=False)
    tablas['comunidades_videos']=c.groupby(['comunidad','video_id']).size().reset_index(name='comentarios').merge(r['videos'][['video_id','title','channel_name']],on='video_id')
    tablas['comunidades_canales']=c.groupby(['comunidad','channel_id','channel_name']).size().reset_index(name='comentarios')
    calidad=[]
    for semilla in [42,7,21]:
        p=nx.community.louvain_communities(proyecciones['autores'],weight='weight',seed=semilla)
        calidad.append(dict(semilla=semilla, comunidades=len(p), modularidad=nx.community.modularity(proyecciones['autores'],p,weight='weight')))
    tablas['calidad_comunidades']=pd.DataFrame(calidad)
    bc=nx.betweenness_centrality(g,normalized=True,weight=None)
    cc=nx.closeness_centrality(g,wf_improved=True)
    articulaciones=set(nx.articulation_points(g))
    base=nx.number_connected_components(g)
    central=[]
    diversidad=r['autores'].set_index('author_channel_id')
    for n,d in g.nodes(data=True):
        aumento=0
        if n in articulaciones:
            copia=g.copy(); copia.remove_node(n)
            aumento=nx.number_connected_components(copia)-base
        central.append(dict(nodo_id=n, tipo=d['tipo'], etiqueta=d['etiqueta'], grado=g.degree(n), fuerza=g.degree(n,weight='peso'), intermediacion=bc[n], cercania=cc[n], articulacion=n in articulaciones, aumento_componentes=aumento, canales=int(diversidad.loc[d['author_channel_id'],'canales']) if d['tipo']=='autor' else None))
    tablas['centralidad']=pd.DataFrame(central).sort_values(['intermediacion','grado'],ascending=False)
    tablas['articulaciones']=tablas['centralidad'].query('articulacion')
    for columna in ['video_id','channel_name','comunidad','source_query']:
        t=c.groupby([columna,'sentimiento']).size().unstack(fill_value=0).reindex(columns=['positivo','negativo','neutral'],fill_value=0)
        t['n']=t.sum(axis=1)
        for etiqueta in ['positivo','negativo','neutral']:
            t[f'porcentaje_{etiqueta}']=100*t[etiqueta]/t['n']
        t['comparacion_descriptiva']=t.n>=10
        tablas[f'sentimiento_{columna}']=t.reset_index()
    tablas['sentimiento_global']=c.sentimiento.value_counts().rename_axis('sentimiento').reset_index(name='comentarios')
    for nombre,t in tablas.items():
        t.to_csv(DATOS_PROCESADOS/f'{nombre}.csv',index=False)
    FIGURAS.mkdir(parents=True,exist_ok=True)
    for nombre,red in redes.items():
        dibujar(red,f'red_{nombre}')
    dibujar(proyecciones['autores'],'comunidades_autores',asignacion)
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,(nombre,red) in zip(axs,redes.items()):
        hist=Counter(dict(red.degree()).values())
        ax.bar(list(hist),list(hist.values())); ax.set_title(nombre)
        ax.set_xlabel('Grado'); ax.set_ylabel('Nodos'); ax.set_yscale('log')
    fig.tight_layout(); fig.savefig(FIGURAS/'distribucion_grados.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    c.sentimiento.value_counts().plot.bar(ax=ax,rot=0,color=['#8a9dad','#cc7755','#559988'])
    ax.set_ylabel('Comentarios'); fig.tight_layout(); fig.savefig(FIGURAS/'sentimiento.png',dpi=160); plt.close(fig)
    r.update(final=tablas, proyecciones=proyecciones, particion=particion)
    return r


if __name__ == '__main__':
    r=ejecutar_final()
    print(r['final']['topologia'].to_string(index=False))
    print(r['final']['comunidades_resumen'].head(3).to_string(index=False))
    print(r['final']['centralidad'].head(8).to_string(index=False))
