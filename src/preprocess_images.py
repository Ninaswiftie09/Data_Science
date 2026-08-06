from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .utils import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SPLITS_DIR,
    class_directories,
    ensure_output_directories,
    list_image_files,
    resolve_train_directory,
)

RANDOM_SEED = 42
DEFAULT_IMAGES_PER_CLASS = 600
DEFAULT_IMAGE_SIZE = 64
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def create_split_manifest(
    images_per_class: int = DEFAULT_IMAGES_PER_CLASS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create balanced train, validation and test splits."""
    train_directory = resolve_train_directory()
    generator = random.Random(seed)
    records = []

    for class_folder in class_directories(train_directory):
        image_files = list_image_files(class_folder)
        generator.shuffle(image_files)

        selected = image_files[: min(images_per_class, len(image_files))]
        total = len(selected)

        train_end = int(total * TRAIN_RATIO)
        validation_end = train_end + int(total * VALIDATION_RATIO)

        split_groups = {
            "train": selected[:train_end],
            "validation": selected[train_end:validation_end],
            "test": selected[validation_end:],
        }

        for split_name, paths in split_groups.items():
            for image_path in paths:
                records.append(
                    {
                        "split": split_name,
                        "class_name": class_folder.name,
                        "source_path": str(
                            image_path.relative_to(PROJECT_ROOT)
                        ).replace("\\", "/"),
                    }
                )

    return pd.DataFrame(records)


def save_split_outputs(manifest: pd.DataFrame) -> dict:
    """Save split tables, chart and summary."""
    manifest_path = SPLITS_DIR / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    split_distribution = (
        manifest.groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    split_distribution.to_csv(
        SPLITS_DIR / "split_distribution.csv",
        index=False,
    )

    totals = manifest["split"].value_counts().to_dict()

    output_path = FIGURES_DIR / "split_distribution.png"
    ordered_splits = ["train", "validation", "test"]
    available_splits = [
        split_name
        for split_name in ordered_splits
        if split_name in split_distribution.columns
    ]

    split_distribution.set_index("class_name")[available_splits].plot(
        kind="bar",
        figsize=(14, 6),
    )
    plt.title("Distribución de clases por conjunto")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad de imágenes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return {
        "manifest_path": str(manifest_path),
        "total_selected_images": int(len(manifest)),
        "train_images": int(totals.get("train", 0)),
        "validation_images": int(totals.get("validation", 0)),
        "test_images": int(totals.get("test", 0)),
    }


def load_and_prepare_image(
    image_path: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Load one image as RGB, resize it and normalize its pixels."""
    with Image.open(image_path) as image:
        prepared = image.convert("RGB").resize(
            (image_size, image_size),
            Image.Resampling.LANCZOS,
        )

    array = np.asarray(prepared, dtype=np.float32) / 255.0
    return array


def save_preprocessing_preview(
    manifest: pd.DataFrame,
    image_size: int = DEFAULT_IMAGE_SIZE,
    amount: int = 3,
) -> Path:
    """Show original images next to resized images."""
    output_path = FIGURES_DIR / "preprocessing_preview.png"

    train_rows = manifest[manifest["split"] == "train"]
    sample = train_rows.sample(
        n=min(amount, len(train_rows)),
        random_state=RANDOM_SEED,
    )

    figure, axes = plt.subplots(
        2,
        len(sample),
        figsize=(4 * len(sample), 7),
    )

    if len(sample) == 1:
        axes = np.array(axes).reshape(2, 1)

    for column_index, row in enumerate(sample.itertuples()):
        source_path = PROJECT_ROOT / row.source_path

        with Image.open(source_path) as image:
            original = image.convert("RGB").copy()

        processed = load_and_prepare_image(
            source_path,
            image_size=image_size,
        )

        axes[0, column_index].imshow(original)
        axes[0, column_index].set_title(
            f"Original clase {row.class_name}"
        )
        axes[0, column_index].axis("off")

        axes[1, column_index].imshow(processed)
        axes[1, column_index].set_title(
            f"Procesada {image_size} por {image_size}"
        )
        axes[1, column_index].axis("off")

    figure.suptitle("Vista previa del preprocesamiento")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path


def export_processed_images(
    manifest: pd.DataFrame,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Path:
    """Create resized RGB copies grouped by split and class."""
    output_root = PROCESSED_DATA_DIR / "images"

    for row in tqdm(
        manifest.itertuples(),
        total=len(manifest),
        desc="Procesando imágenes",
    ):
        source_path = PROJECT_ROOT / row.source_path
        target_folder = output_root / row.split / row.class_name
        target_folder.mkdir(parents=True, exist_ok=True)
        target_path = target_folder / source_path.name

        with Image.open(source_path) as image:
            processed = image.convert("RGB").resize(
                (image_size, image_size),
                Image.Resampling.LANCZOS,
            )
            processed.save(target_path, format="JPEG", quality=92)

    return output_root


def run_preprocessing(
    images_per_class: int = DEFAULT_IMAGES_PER_CLASS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    export_images: bool = False,
) -> dict:
    """Create the split plan and preprocessing evidence."""
    ensure_output_directories()

    manifest = create_split_manifest(
        images_per_class=images_per_class,
        seed=RANDOM_SEED,
    )

    split_summary = save_split_outputs(manifest)
    preview_path = save_preprocessing_preview(
        manifest,
        image_size=image_size,
    )

    exported_directory = None
    if export_images:
        exported_directory = export_processed_images(
            manifest,
            image_size=image_size,
        )

    summary = {
        **split_summary,
        "images_per_class_requested": images_per_class,
        "image_size": image_size,
        "color_mode": "RGB",
        "normalization": "Valores divididos entre 255 durante la carga",
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": round(
            1.0 - TRAIN_RATIO - VALIDATION_RATIO,
            2,
        ),
        "preview_path": str(preview_path),
        "exported_images_directory": (
            str(exported_directory)
            if exported_directory is not None
            else None
        ),
    }

    summary_path = REPORTS_DIR / "preprocessing_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("El plan de preprocesamiento terminó.")
    print(
        "Imágenes seleccionadas: "
        f"{summary['total_selected_images']}"
    )
    print(f"Resumen guardado en: {summary_path}")

    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocesamiento del dataset ASL Alphabet"
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=DEFAULT_IMAGES_PER_CLASS,
        help="Cantidad máxima de imágenes seleccionadas por clase",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Tamaño final de cada lado de la imagen",
    )
    parser.add_argument(
        "--export-images",
        action="store_true",
        help="Guardar copias redimensionadas dentro de data/processed/images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_preprocessing(
        images_per_class=arguments.images_per_class,
        image_size=arguments.image_size,
        export_images=arguments.export_images,
    )
