from IPython.display import display, Image
import pandas as pd
from scripts.analisis_avance import RAIZ


def tabla(nombre, limite=20):
    datos=pd.read_csv(RAIZ/'data/processed'/f'{nombre}.csv')
    with pd.option_context('display.max_columns',None,'display.max_colwidth',70):
        display(datos.head(limite).round(4))


def figura(nombre):
    display(Image(filename=str(RAIZ/'output/figures'/f'{nombre}.png')))
