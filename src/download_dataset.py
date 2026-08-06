from __future__ import annotations

from pathlib import Path

import kagglehub

from utils import RAW_DATA_DIR, resolve_train_directory

DATASET_HANDLE = "grassknoted/asl-alphabet"


def dataset_is_available() -> bool:
    """Check whether the training images are already inside data/raw."""
    try:
        resolve_train_directory()
        return True
    except FileNotFoundError:
        return False


def download_dataset(force_download: bool = False) -> Path:
    """Download the latest ASL Alphabet dataset into data/raw."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if dataset_is_available() and not force_download:
        train_directory = resolve_train_directory()
        print("El dataset ya existe.")
        print(f"Carpeta de entrenamiento: {train_directory}")
        return train_directory

    downloaded_path = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(RAW_DATA_DIR),
        force_download=force_download,
    )

    print("La descarga terminó.")
    print(f"Ruta de los archivos: {downloaded_path}")
    return Path(downloaded_path)


if __name__ == "__main__":
    download_dataset()
