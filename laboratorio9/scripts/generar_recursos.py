"""Análisis de series, evaluación de modelos y exportación de gráficos."""
from pathlib import Path
import json
import importlib.metadata as md
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from prophet import Prophet

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'laboratorio9'
FIG = OUT / 'figuras'
DATA = OUT / 'datos'
for p in [FIG, DATA]:
    p.mkdir(parents=True, exist_ok=True)
BLUE, TEAL, AMBER = '#1A334B', '#148383', '#C48014'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': BLUE, 'text.color': BLUE,
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
def csv(frame, name, index=False):
    frame.to_csv(DATA / (name + '.csv'), index=index, encoding='utf-8-sig')
def save(fig, name, note='Turistas y excursionistas | INGUAT/IGM | enero 2009–junio 2026'):
    fig.text(.02, .015, note, fontsize=9, color=BLUE)
    fig.tight_layout(rect=[0, .055, 1, 1])
    fig.savefig(FIG / (name + '.png'), dpi=300, facecolor='white')
    fig.savefig(FIG / (name + '.svg'), facecolor='white')
    plt.close(fig)
def chart(title, size=(12, 5.5)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, loc='left', pad=22, fontsize=16)
    return fig, ax
def decompose(s):
    return seasonal_decompose(np.log1p(s), model='additive', period=12, extrapolate_trend='freq')
def metrics(y, p):
    return {'RMSE': float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2))),
            'MAE': float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))}

df = pd.read_excel(ROOT / 'data/raw/Base_Migracion_2009-2026jun.xlsx', sheet_name='Datos')
df.columns = df.columns.str.strip()
for c in ['Año', 'Mes cod', 'Viajero']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['Fecha'] = pd.to_datetime(dict(year=df['Año'], month=df['Mes cod'], day=1), errors='coerce')
df = df.dropna(subset=['Fecha', 'Viajero'])
df = df[df['Tipo de Viajero'].isin(['Turista', 'Excursionista'])].copy()
s = df.groupby('Fecha').Viajero.sum().sort_index().asfreq('MS').astype(float)
assert not s.isna().any(), 'Hay meses ausentes: no se interpolan ni inventan ceros.'
train, test = s.iloc[:int(len(s)*.7)], s.iloc[int(len(s)*.7):]
annual = s.resample('YS').sum()
csv(s.rename('Viajeros').rename_axis('Fecha').reset_index(), 'evolucion_mensual')
af = annual.rename('Viajeros').rename_axis('Fecha').reset_index()
af['Meses_observados'] = [int((s.index.year == d.year).sum()) for d in af.Fecha]
csv(af, 'totales_anuales')
vias = df.groupby('Vía', dropna=False).Viajero.sum().sort_values(ascending=False)
regions = df.groupby('Región dos', dropna=False).Viajero.sum().sort_values(ascending=False)
assert np.allclose([vias.sum(), regions.sum(), annual.sum()], s.sum(), rtol=0, atol=1e-6)
for values, name in [(vias, 'vias_ingreso'), (regions, 'mercados_origen')]:
    table = values.rename('Viajeros').to_frame()
    table['Participacion_porcentaje'] = values / s.sum()*100
    csv(table, name, True)
csv(df.groupby(['Año', 'Región dos'], dropna=False).Viajero.sum().rename('Viajeros').reset_index(), 'auditoria_regiones_por_anio')
csv(df.groupby(['Año', 'País'], dropna=False).Viajero.sum().rename('Viajeros').reset_index(), 'auditoria_paises_por_anio')
fig, ax = chart('Evolución mensual de viajeros internacionales a Guatemala')
ax.plot(s.index, s, color=BLUE, lw=1.8)
ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-01'), color=AMBER, alpha=.16)
ax.text(pd.Timestamp('2012-01-01'), s.max()*.9, 'Antes de la pandemia', fontsize=11)
ax.annotate('Caída de 2020', xy=(pd.Timestamp('2020-07-01'), s.loc['2020-07-01']), xytext=(pd.Timestamp('2016-01-01'), s.max()*.4), arrowprops={'arrowstyle':'->', 'color':AMBER}, color=AMBER)
ax.text(pd.Timestamp('2022-01-01'), s.max()*.8, 'Recuperación\nposterior', color=TEAL)
ax.set(ylabel='Viajeros mensuales', xlabel='Año', ylim=(0, s.max()*1.15))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p: f'{x/1000:,.0f} mil'))
save(fig, '01_evolucion_historica', '2026 incluye solo enero–junio. Serie mensual real | Turistas y excursionistas | INGUAT/IGM')
season = decompose(train).seasonal.groupby(train.index.month).mean()
months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
csv(pd.DataFrame({'Mes':range(1,13), 'Nombre_mes':['enero','febrero','marzo','abril','mayo','junio','julio','agosto','septiembre','octubre','noviembre','diciembre'], 'Componente_estacional_log1p':season.values}), 'estacionalidad_mensual')
fig, ax = chart('¿En qué meses aumenta el movimiento turístico?')
ax.bar(months, season, color=[TEAL if i==season.idxmax() else BLUE for i in range(1,13)])
ax.axhline(0, color=BLUE, lw=.7)
for i,v in enumerate(season):
    ax.annotate(f'{v:.3f}', (i,v), xytext=(0,5 if v>=0 else -16), textcoords='offset points', ha='center', fontsize=10)
ax.margins(y=.25)
ax.set_ylabel('Componente estacional en log(1 + viajeros)')
save(fig, '02_estacionalidad_mensual', f'Descomposición aditiva mensual, período 12 | Entrenamiento: {train.index.min():%Y-%m}–{train.index.max():%Y-%m}')
for values, title, name in [(vias,'¿Por dónde ingresan los viajeros a Guatemala?','03_vias_ingreso'),(regions.head(5),'Principales mercados de origen','04_mercados_origen')]:
    fig, ax = chart(title)
    ax.barh(values.index.astype(str), values/1e6, color=[TEAL]+[BLUE]*(len(values)-1))
    ax.invert_yaxis()
    for i,v in enumerate(values):
        ax.text(v/1e6+.15, i, f'{v/1e6:.2f} millones · {v/s.sum()*100:.2f} %', va='center', fontsize=11)
    ax.set_xlim(0, values.max()/1e6*1.65)
    ax.set_xlabel('Viajeros acumulados (millones)')
    save(fig, name)

series = {'Total': s}
for c, names in [('Vía', vias.index), ('Región dos', regions.head(3).index)]:
    pivot = df.pivot_table(index='Fecha', columns=c, values='Viajero', aggfunc='sum', fill_value=0)
    for name in names:
        series[('Vía ' if c=='Vía' else 'Región ')+name] = pivot[name].reindex(s.index, fill_value=0).astype(float)
rows = []
for name, ss in series.items():
    desc = decompose(ss)
    a = ss.resample('YS').sum()
    rows.append({'Serie':name,'Fuerza estacional':max(0,1-np.var(desc.resid)/np.var(desc.seasonal+desc.resid)),
                 'Crecimiento anual aproximado':100*np.expm1(np.polyfit(np.arange(len(ss)),np.log1p(ss),1)[0]*12),
                 'Coeficiente de variación':ss.std()/ss.mean(),
                 'Cambio 2020 frente a 2019':100*(a.loc['2020-01-01']/a.loc['2019-01-01']-1) if a.loc['2019-01-01'] else np.nan})
comparison = pd.DataFrame(rows).set_index('Serie')
csv(comparison, 'resumen_siete_series', True)
csv(pd.DataFrame(series).rename_axis('Fecha').reset_index(), 'series_mensuales')
definitions = [('Mayor estacionalidad','Fuerza estacional','1 − Var(residuo)/Var(estacional + residuo), mínimo 0; log1p, serie completa','Patrón mensual más marcado',False),('Mayor crecimiento','Crecimiento anual aproximado','100 × (exp(12 × pendiente OLS de log1p mensual) − 1)','Tendencia anual aproximada; no es variación interanual',False),('Mayor volatilidad','Coeficiente de variación','Desviación estándar muestral / media mensual','Mayor dispersión relativa',False),('Mayor caída en 2020','Cambio 2020 frente a 2019','100 × (total 2020 / total 2019 − 1); excluye base 2019 cero','Mayor caída porcentual entre series con base comparable',True)]
quad=[]
for label,col,method,interpret,minimum in definitions:
    winner = comparison[col].idxmin() if minimum else comparison[col].idxmax()
    quad.append({'Indicador':label,'Serie ganadora':winner,'Valor':comparison.loc[winner,col], 'Método de cálculo':method,'Interpretación':interpret})
q = pd.DataFrame(quad)
csv(q,'comparacion_series')
fig, ax = chart('Siete series: cuatro resultados destacados', (12,4.7))
ax.axis('off')
for i,row in q.iterrows():
    y=.87-i*.23
    ax.text(0,y,row['Indicador'],weight='bold',fontsize=13)
    ax.text(.36,y,row['Serie ganadora'],fontsize=12)
    ax.text(.96,y,f"{row['Valor']:.3f}"+(' %' if i in [1,3] else ''),ha='right',color=TEAL,weight='bold',fontsize=14)
save(fig,'05_comparacion_series','Fuerza estacional y CV: sin unidades. Crecimiento y caída: %. Serie marítima sin base 2019 comparable.')

# Recuperar cifras originales sin atribuir precisión inexistente al notebook.
nb = json.loads((ROOT/'notebooks/02_resultados.ipynb').read_text(encoding='utf8'))
original = None
for cell in nb['cells']:
    for output in cell.get('outputs',[]):
        html = ''.join(output.get('data',{}).get('text/html',[]))
        if '<table' in html and 'Prophet' in html and '139,548.877' in html:
            table_xml = ET.fromstring(html[html.index('<table'):html.index('</table>')+8])
            headers = [x.text for x in table_xml.find('thead').find('tr').findall('th')][1:]
            original = pd.DataFrame([[x.text for x in row.findall('td')] for row in table_xml.find('tbody').findall('tr')], columns=headers)
            for column in ['MAE','RMSE','AIC','BIC']:
                original[column] = pd.to_numeric(original[column].str.replace(',',''),errors='coerce')
            break
    if original is not None:
        break
assert original is not None, 'No se encontró la tabla original de la serie total'
original = original.loc[:,~original.columns.astype(str).str.startswith('Unnamed')]
csv(original, 'metricas_originales_notebook')
sarima = original[original.Modelo.str.startswith('SARIMA')].sort_values('RMSE').iloc[0]
selected = original[(~original.Modelo.str.startswith('SARIMA')) | (original.Modelo==sarima.Modelo)].sort_values('RMSE').copy()
selected['Precision_fuente'] = '3 decimales guardados en notebook; no disponibles más decimales originales'
selected['Prueba_inicio'] = str(test.index.min().date())
selected['Prueba_fin'] = str(test.index.max().date())
csv(selected[['Modelo','RMSE','MAE','Precision_fuente','Prueba_inicio','Prueba_fin']], 'metricas_modelos')
fig, ax = chart('Comparación de modelos: ¿cuál predice con menor error?', (13,6))
labels = ['Prophet','SARIMA con menor RMSE','Suavizamiento exponencial simple','Seasonal Naive','Holt-Winters']
ax.barh(labels, selected.RMSE, color=[TEAL]+[BLUE]*4)
ax.invert_yaxis()
for i,(_,r) in enumerate(selected.iterrows()):
    ax.text(r.RMSE+3500,i,f'RMSE {r.RMSE:,.2f}\nMAE {r.MAE:,.3f}',va='center',fontsize=10)
ax.set_xlim(0, selected.RMSE.max()*1.43)
ax.set_xlabel('Error en viajeros mensuales · menor RMSE = mejor desempeño')
save(fig,'06_comparacion_modelos','Mismo conjunto de prueba: abril 2021–junio 2026 | Métricas originales guardadas a 3 decimales')

np.random.seed(2026)
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                seasonality_mode='additive', interval_width=.80)
model.fit(pd.DataFrame({'ds':train.index,'y':train.values}))
np.random.seed(2026)
forecast = model.predict(pd.DataFrame({'ds':test.index}))
pred = np.maximum(forecast.yhat.values,0)  # misma restricción del notebook
pred_df = pd.DataFrame({'Fecha':test.index,'Viajeros_reales':test.values,'Prediccion_Prophet':pred,
                        'yhat_original':forecast.yhat.values,'Limite_inferior':forecast.yhat_lower.values,
                        'Limite_superior':forecast.yhat_upper.values})
csv(pred_df,'predicciones_prophet')
pm = metrics(test,pred)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    sarima_fit = SARIMAX(np.log1p(train), order=(1,1,1), seasonal_order=(0,0,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False,maxiter=250)
    ses_fit = SimpleExpSmoothing(train, initialization_method='estimated').fit(optimized=True)
    hw_fit = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12, initialization_method='estimated').fit(optimized=True,use_brute=True)
reproduced = {'Prophet':pred, str(sarima.Modelo):np.maximum(np.expm1(sarima_fit.forecast(len(test))),0),
              'Suavizamiento exponencial simple':np.maximum(ses_fit.forecast(len(test)),0),
              'Seasonal naive':np.resize(train.iloc[-12:].values,len(test)),
              'Holt Winters':np.maximum(hw_fit.forecast(len(test)),0)}
repro_table = pd.DataFrame([{'Modelo':name,**metrics(test,values)} for name,values in reproduced.items()])
repro_table = repro_table.merge(selected[['Modelo','RMSE','MAE']],on='Modelo',suffixes=('_reproducido','_original_guardado'))
for metric in ['MAE','RMSE']:
    repro_table['Diferencia_'+metric] = repro_table[metric+'_reproducido']-repro_table[metric+'_original_guardado']
csv(repro_table,'metricas_modelos_reproducidas')
csv(pd.DataFrame({'Fecha':test.index,'Viajeros_reales':test.values,**reproduced}), 'predicciones_modelos_reproducidos')
coverage = float(((test.values>=forecast.yhat_lower.values)&(test.values<=forecast.yhat_upper.values)).mean())
pm.update({'Cobertura_observada':coverage,'Nivel_nominal':.8,'Meses_prueba':len(test),
           'Diferencia_RMSE_original_redondeado':pm['RMSE']-float(selected.iloc[0].RMSE)})
csv(pd.DataFrame([pm]),'evaluacion_prophet')
fig, ax = chart('Predicción de viajeros con Prophet', (12,6))
ax.text(0,1.025,'Valores observados y estimaciones sobre el conjunto de prueba',transform=ax.transAxes,fontsize=11)
ax.fill_between(test.index, forecast.yhat_lower.values, forecast.yhat_upper.values, color=AMBER,alpha=.22,label='Estimación — intervalo del 80 %')
ax.plot(test.index,test.values,color=BLUE,lw=2,label='Valores reales observados')
ax.plot(test.index,pred,color=TEAL,lw=2,label='Predicción de Prophet')
ax.set(ylabel='Viajeros mensuales',xlabel='Fecha')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x/1000:,.0f} mil'))
ax.legend(loc='upper left',fontsize=10,frameon=False)
save(fig,'07_prediccion_prophet',f'Evaluación retrospectiva: abril 2021–junio 2026 | Cobertura observada: {coverage:.1%} | Límites originales de Prophet')

kpis = pd.DataFrame([
    ['Total acumulado',s.sum(),f'{s.sum()/1e6:.1f} millones','Viajeros, enero 2009–junio 2026'],
    ['Variación 2020 vs 2019',100*(annual.loc['2020-01-01']/annual.loc['2019-01-01']-1),f'{100*(annual.loc["2020-01-01"]/annual.loc["2019-01-01"]-1):.2f} %','Cambio porcentual entre años completos'],
    ['Participación terrestre',vias['Terrestre']/s.sum()*100,f'{vias["Terrestre"]/s.sum()*100:.1f} %','Porcentaje del acumulado comparable'],
    ['Variación 2025 vs 2019',100*(annual.loc['2025-01-01']/annual.loc['2019-01-01']-1),f'{100*(annual.loc["2025-01-01"]/annual.loc["2019-01-01"]-1):.2f} %','Cambio porcentual entre años completos']],
    columns=['Indicador','Valor_exacto','Valor_presentacion','Descripción'])
csv(kpis,'indicadores_infografia')
versions={p:md.version(p) for p in ['numpy','pandas','matplotlib','statsmodels','prophet','cmdstanpy','openpyxl','pillow']}
(OUT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n',encoding='utf8')
(DATA/'configuracion_ejecucion.json').write_text(json.dumps({'versiones':versions,'semilla':2026,'train_inicio':str(train.index.min().date()),'train_fin':str(train.index.max().date()),'test_inicio':str(test.index.min().date()),'test_fin':str(test.index.max().date()),'prophet':{'yearly_seasonality':True,'weekly_seasonality':False,'daily_seasonality':False,'seasonality_mode':'additive','interval_width':.8,'uncertainty_samples':1000}},ensure_ascii=False,indent=2),encoding='utf8')
assert len(list(FIG.glob('*.png')))==7
assert np.all(pred_df.Limite_inferior<=pred_df.Limite_superior)
assert train.index.max()<test.index.min()
print(kpis.to_string(index=False))
print('Prophet:',pm)
print('Recursos generados en',OUT)
