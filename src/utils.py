# Librerías
from pathlib import Path
import pandas as pd


# Ruta principal del proyecto
BASE_DIR = Path(__file__).resolve().parents[1]


# Rutas de los datos
RAW_FILE = BASE_DIR / "data" / "raw" / "guatemala_temperatura.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# Columnas de temperatura
TEMPERATURE_COLUMNS = [
    "temperature_2m_c",
    "skin_temperature_c",
    "soil_temperature_layer_1_c",
    "soil_temperature_layer_2_c",
    "soil_temperature_layer_3_c",
    "soil_temperature_layer_4_c"
]


def cargar_datos():
    # Cargar el archivo CSV
    df = pd.read_csv(RAW_FILE)

    # Convertir la fecha
    df["month"] = pd.to_datetime(df["month"])

    # Ordenar los datos
    df = df.sort_values("month")

    # Reiniciar los índices
    df = df.reset_index(drop=True)

    return df


def crear_carpeta_processed():
    # Crear la carpeta si no existe
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)