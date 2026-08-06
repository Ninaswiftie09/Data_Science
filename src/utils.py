from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
REPORTS_DIR = PROCESSED_DATA_DIR / "reports"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"


def ensure_output_directories() -> None:
    """Create the output folders used by the analysis."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    """Return True when a file has a supported image extension."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(folder: Path) -> list[Path]:
    """Return all supported image files inside one folder."""
    return sorted(path for path in folder.iterdir() if is_image_file(path))


def _contains_class_folders(folder: Path, minimum_classes: int = 20) -> bool:
    """Check whether a folder looks like the ASL training folder."""
    if not folder.exists() or not folder.is_dir():
        return False

    class_count = 0
    for child in folder.iterdir():
        if child.is_dir() and any(is_image_file(path) for path in child.iterdir()):
            class_count += 1

    return class_count >= minimum_classes


def resolve_train_directory() -> Path:
    """Find the ASL training directory even when Kaggle adds an extra folder."""
    candidates = [
        RAW_DATA_DIR / "asl_alphabet_train" / "asl_alphabet_train",
        RAW_DATA_DIR / "asl_alphabet_train",
    ]

    for candidate in candidates:
        if _contains_class_folders(candidate):
            return candidate

    if RAW_DATA_DIR.exists():
        for candidate in RAW_DATA_DIR.rglob("*"):
            if candidate.is_dir() and _contains_class_folders(candidate):
                return candidate

    raise FileNotFoundError(
        "No se encontró la carpeta de entrenamiento. "
        "Verifica que el dataset esté dentro de data/raw."
    )


def resolve_test_directory() -> Path | None:
    """Find the small official Kaggle test directory when it exists."""
    candidates = [
        RAW_DATA_DIR / "asl_alphabet_test" / "asl_alphabet_test",
        RAW_DATA_DIR / "asl_alphabet_test",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            if any(is_image_file(path) for path in candidate.rglob("*")):
                return candidate

    return None


def class_directories(train_directory: Path) -> list[Path]:
    """Return the class folders in a stable ASL order."""
    folders = [
        folder
        for folder in train_directory.iterdir()
        if folder.is_dir() and list_image_files(folder)
    ]

    preferred_order = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    preferred_order += ["del", "delete", "nothing", "space"]

    rank = {name.lower(): index for index, name in enumerate(preferred_order)}

    return sorted(
        folders,
        key=lambda folder: (
            rank.get(folder.name.lower(), len(rank)),
            folder.name.lower(),
        ),
    )


def choose_files(
    files: Iterable[Path],
    amount: int,
    seed: int = 42,
) -> list[Path]:
    """Choose a reproducible random sample of files."""
    file_list = list(files)
    if amount >= len(file_list):
        return file_list

    generator = random.Random(seed)
    return generator.sample(file_list, amount)
