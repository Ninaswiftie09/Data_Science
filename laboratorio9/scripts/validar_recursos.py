"""Verifica consistencia de los CSV y calidad de exportación de los PNG."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image

out = Path(__file__).resolve().parents[1]
data = out / 'datos'
def read(name):
    return pd.read_csv(data / (name+'.csv'))
s = read('evolucion_mensual')
total = s.Viajeros.sum()
for name in ['totales_anuales', 'vias_ingreso', 'mercados_origen']:
    assert np.isclose(read(name).Viajeros.sum(),total,rtol=0,atol=1e-6),name
assert len(s)==210
assert len(read('estacionalidad_mensual'))==12
p = read('predicciones_prophet')
assert len(p)==63 and p.Fecha.iloc[0]=='2021-04-01' and p.Fecha.iloc[-1]=='2026-06-01'
e = read('evaluacion_prophet').iloc[0]
assert np.isclose(np.sqrt(np.mean((p.Viajeros_reales-p.Prediccion_Prophet)**2)),e.RMSE)
assert np.isclose(np.mean(np.abs(p.Viajeros_reales-p.Prediccion_Prophet)),e.MAE)
coverage = ((p.Viajeros_reales>=p.Limite_inferior)&(p.Viajeros_reales<=p.Limite_superior)).mean()
assert np.isclose(coverage,e.Cobertura_observada)
assert (p.Limite_inferior<=p.Limite_superior).all()
original = read('metricas_originales_notebook').set_index('Modelo')
for r in read('metricas_modelos').itertuples(index=False):
    assert r.RMSE==original.loc[r.Modelo,'RMSE'] and r.MAE==original.loc[r.Modelo,'MAE']
repro = read('predicciones_modelos_reproducidos')
for _,r in read('metricas_modelos_reproducidas').iterrows():
    error = repro.Viajeros_reales-repro[r.Modelo]
    assert np.isclose(np.sqrt(np.mean(error**2)),r.RMSE_reproducido)
    assert np.isclose(np.mean(np.abs(error)),r.MAE_reproducido)
images=[]
for path in sorted((out/'figuras').glob('*.png')):
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        assert min(img.info['dpi'])>=299
        assert img.width>=3000 and img.height>=1200
        images.append({'Archivo':path.name,'Ancho_px':img.width,'Alto_px':img.height,'DPI':img.info['dpi'][0]})
assert len(images)==7 and len(list((out/'figuras').glob('*.svg')))==7
report={'Totales_coinciden':True,'Metricas_y_cobertura_coinciden':True,'PNG_verificados':images}
(data/'validacion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print('Validación correcta: totales, métricas, cobertura, 7 PNG a 300 DPI y 7 SVG.')
