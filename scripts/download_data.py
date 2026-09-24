"""Descarga los archivos oficiales del INE sin reemplazar archivos existentes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.ine.gob.gt/wp-content/uploads/"
FILES = {
    "data/raw/2025T1.xlsx": "2026/01/Personas_ENEIC_T1_2025.xlsx",
    "data/raw/2025T2.xlsx": "2026/01/Personas-ENEIC-T2-2025.xlsx",
    "data/raw/2025T3.xlsx": "2026/05/Base-de-datos-Personas-ENEIC-III-2025.xlsx",
    "data/raw/2025T4.xlsx": "2026/06/Base-de-datos-Personas-ENEIC-IV-2025.xlsx",
    "data/raw/Base-de-datos-Personas-ENEIC-I-2026.xlsx": "2026/09/Base-de-datos-Personas-ENEIC-I-2026.xlsx",
    "data/dictionaries/2025T1.xlsx": "2025/11/Diccionario_Personas_ENEIC_I-2025.xlsx",
    "data/dictionaries/2025T2.xlsx": "2025/11/Diccionario_Personas_ENEIC_II-2025.xlsx",
    "data/dictionaries/2025T3.xlsx": "2026/05/Diccionario-Personas-ENEIC-III-2025.xlsx",
    "data/dictionaries/2025T4.xlsx": "2026/06/Diccionario-Personas-ENEIC-IV-2025.xlsx",
    "data/dictionaries/2026T1.xlsx": "2026/09/Diccionario-Personas-ENEIC-I-2026.xlsx",
}

def download(item):
    name, relative_url = item
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".download")
        urlretrieve(BASE + relative_url, temporary)
        temporary.replace(path)
    print(name, flush=True)
    return {"file": name, "url": BASE + relative_url,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(download, FILES.items()))
    (ROOT / "data/sources.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
