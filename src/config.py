from pathlib import Path

# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATOS_DIR = RAW_DATA_DIR / "sentinel2"
RESULTADOS_DIR = PROCESSED_DATA_DIR
GRAFICOS_DIR = PROCESSED_DATA_DIR / "figures"

for carpeta in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATOS_DIR,
    GRAFICOS_DIR
]:
    carpeta.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

CONEXION_URL = "https://openeo.dataspace.copernicus.eu"

RESOLUCION_METROS = 20

MAX_FECHAS = None

# Bandas necesarias para:
# NDVI -> B04 y B08
# NDWI -> B03 y B08
# Cianobacteria -> B04, B05, B06 y B08
# SCL -> máscara de nubes
BANDAS = [
    "B03",
    "B04",
    "B05",
    "B06",
    "B08",
    "SCL"
]


# ============================================================
# LAGOS Y FECHAS OFICIALES
# ============================================================

LAGOS = {
    "Atitlán": {
        "bbox": {
            "west": -91.326256,
            "east": -91.07151,
            "south": 14.5948,
            "north": 14.750979
        },
        "fechas": [
            "2025-01-18",
            "2025-04-13",
            "2025-05-13",
            "2025-07-17",
            "2025-11-21",
            "2025-12-29",
            "2026-02-12",
            "2026-03-24",
            "2026-04-13",
            "2026-04-28",
            "2026-07-22"
        ]
    },

    "Amatitlán": {
        "bbox": {
            "west": -90.638065,
            "east": -90.512924,
            "south": 14.412347,
            "north": 14.493799
        },
        "fechas": [
            "2025-01-28",
            "2025-04-15",
            "2025-04-28",
            "2025-11-24",
            "2026-01-08",
            "2026-02-02",
            "2026-02-07",
            "2026-03-29",
            "2026-04-13",
            "2026-04-28",
            "2026-06-19"
        ]
    }
}


# ============================================================
# NUBOSIDAD REPORTADA EN LA GUÍA
# ============================================================

NUBOSIDAD_OFICIAL = [
    # Amatitlán
    ("Amatitlán", "2025-01-28", 0.06, "Sentinel-2B"),
    ("Amatitlán", "2025-04-15", 0.09, "Sentinel-2A"),
    ("Amatitlán", "2025-04-28", 1.03, "Sentinel-2B"),
    ("Amatitlán", "2025-11-24", 0.50, "Sentinel-2B"),
    ("Amatitlán", "2026-01-08", 0.77, "Sentinel-2C"),
    ("Amatitlán", "2026-02-02", 0.39, "Sentinel-2B"),
    ("Amatitlán", "2026-02-07", 0.02, "Sentinel-2C"),
    ("Amatitlán", "2026-03-29", 0.01, "Sentinel-2C"),
    ("Amatitlán", "2026-04-13", 0.09, "Sentinel-2B"),
    ("Amatitlán", "2026-04-28", 4.96, "Sentinel-2C"),
    ("Amatitlán", "2026-06-19", 13.00, "Sentinel-2A"),

    # Atitlán
    ("Atitlán", "2025-01-18", 0.02, "Sentinel-2B"),
    ("Atitlán", "2025-04-13", 0.54, "Sentinel-2C"),
    ("Atitlán", "2025-05-13", 4.37, "Sentinel-2C"),
    ("Atitlán", "2025-07-17", 3.57, "Sentinel-2A"),
    ("Atitlán", "2025-11-21", 3.15, "Sentinel-2A"),
    ("Atitlán", "2025-12-29", 3.17, "Sentinel-2C"),
    ("Atitlán", "2026-02-12", 0.04, "Sentinel-2B"),
    ("Atitlán", "2026-03-24", 3.17, "Sentinel-2B"),
    ("Atitlán", "2026-04-13", 0.01, "Sentinel-2B"),
    ("Atitlán", "2026-04-28", 4.96, "Sentinel-2C"),
    ("Atitlán", "2026-07-22", 4.02, "Sentinel-2B")
]