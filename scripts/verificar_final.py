from itertools import combinations
from pathlib import Path
import networkx as nx
import nbformat
import pandas as pd
from scripts.analisis_final import ejecutar_final
from scripts.analisis_avance import clasificar_sentimiento
from scripts.verificar_avance import verificar


def verificar_final():
    verificar()
    r=ejecutar_final(); g=r['red']; c=r['comentarios']
    assert len(g)==c.author_channel_id.nunique()+r['videos'].video_id.nunique()
    assert g.number_of_edges()==c.groupby(['author_channel_id','video_id']).ngroups
    for tipo,red in r['proyecciones'].items():
        campo,otro=('author_channel_id','video_id') if tipo=='autores' else ('video_id','author_channel_id')
        prefijo='autor:' if tipo=='autores' else 'video:'
        grupos={prefijo+str(k):set(v[otro]) for k,v in c.groupby(campo)}
        esperadas={}
        for a,b in combinations(red.nodes,2):
            peso=len(grupos.get(a,set()) & grupos.get(b,set()))
            if peso: esperadas[frozenset([a,b])]=peso
        observadas={frozenset([a,b]):d['weight'] for a,b,d in red.edges(data=True)}
        assert esperadas==observadas
    miembros=[n for grupo in r['particion'] for n in grupo]
    assert len(miembros)==len(set(miembros))==c.author_channel_id.nunique()
    assert r['final']['comunidades_resumen'].comentarios.sum()==len(c)
    assert r['final']['sentimiento_global'].comentarios.sum()==len(c)
    assert clasificar_sentimiento('No es bueno')[1]=='negativo'
    assert clasificar_sentimiento('No es malo')[1]=='positivo'
    assert clasificar_sentimiento('Gracias, excelente trabajo')[1]=='positivo'
    assert clasificar_sentimiento('Es terrible')[1]=='negativo'
    for p in Path('notebooks').glob('*.ipynb'):
        nb=nbformat.read(p,as_version=4); nbformat.validate(nb)
        assert all(cell.execution_count is not None for cell in nb.cells if cell.cell_type=='code')
        assert not any(o.output_type=='error' for cell in nb.cells if cell.cell_type=='code' for o in cell.outputs)
        assert not any('**' in cell.source or '`' in cell.source for cell in nb.cells if cell.cell_type=='markdown')
    print('Integración, pesos independientes, partición, sentimiento y notebooks verificados.')


if __name__=='__main__':
    verificar_final()
