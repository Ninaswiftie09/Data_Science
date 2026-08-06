from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .utils import (
    FIGURES_DIR,
    REPORTS_DIR,
    choose_files,
    class_directories,
    ensure_output_directories,
    list_image_files,
    resolve_test_directory,
    resolve_train_directory,
)

RANDOM_SEED = 42
DEFAULT_PROPERTY_SAMPLE_PER_CLASS = 50


def collect_class_distribution(train_directory: Path) -> pd.DataFrame:
    """Count the images available in every class."""
    records = []

    for class_folder in class_directories(train_directory):
        records.append(
            {
                "class_name": class_folder.name,
                "image_count": len(list_image_files(class_folder)),
            }
        )

    return pd.DataFrame(records)


def inspect_image_properties(
    train_directory: Path,
    sample_per_class: int = DEFAULT_PROPERTY_SAMPLE_PER_CLASS,
    full_validation: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Inspect image format, resolution, color mode and damaged files."""
    records = []
    damaged_files = []

    for class_index, class_folder in enumerate(class_directories(train_directory)):
        image_files = list_image_files(class_folder)

        if not full_validation:
            image_files = choose_files(
                image_files,
                min(sample_per_class, len(image_files)),
                seed=RANDOM_SEED + class_index,
            )

        for image_path in image_files:
            try:
                with Image.open(image_path) as image:
                    image.load()
                    records.append(
                        {
                            "class_name": class_folder.name,
                            "file_name": image_path.name,
                            "width": image.width,
                            "height": image.height,
                            "format": image.format or "unknown",
                            "color_mode": image.mode,
                        }
                    )
            except (UnidentifiedImageError, OSError, ValueError):
                damaged_files.append(str(image_path))

    return pd.DataFrame(records), damaged_files


def save_class_distribution_plot(distribution: pd.DataFrame) -> Path:
    """Create the class distribution chart."""
    output_path = FIGURES_DIR / "class_distribution.png"

    plt.figure(figsize=(14, 6))
    plt.bar(distribution["class_name"], distribution["image_count"])
    plt.title("Cantidad de imágenes por clase")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad de imágenes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def _open_rgb(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


def save_sample_classes_plot(
    train_directory: Path,
    samples_per_class: int = 3,
) -> Path:
    """Show at least five classes and several examples of each one."""
    folders = class_directories(train_directory)
    preferred_names = ["A", "B", "C", "D", "E"]

    selected = []
    by_name = {folder.name.upper(): folder for folder in folders}

    for name in preferred_names:
        if name in by_name:
            selected.append(by_name[name])

    if len(selected) < 5:
        for folder in folders:
            if folder not in selected:
                selected.append(folder)
            if len(selected) == 5:
                break

    output_path = FIGURES_DIR / "sample_classes.png"
    figure, axes = plt.subplots(
        len(selected),
        samples_per_class,
        figsize=(3 * samples_per_class, 2.8 * len(selected)),
    )

    if len(selected) == 1:
        axes = [axes]

    for row_index, class_folder in enumerate(selected):
        chosen = choose_files(
            list_image_files(class_folder),
            samples_per_class,
            seed=RANDOM_SEED + row_index,
        )

        for column_index in range(samples_per_class):
            axis = axes[row_index][column_index]
            axis.axis("off")

            if column_index < len(chosen):
                axis.imshow(_open_rgb(chosen[column_index]))
                axis.set_title(f"Clase {class_folder.name}")

    figure.suptitle("Ejemplos de cinco clases")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path


def save_same_class_variability_plot(
    train_directory: Path,
    class_name: str = "A",
    amount: int = 8,
) -> Path:
    """Show several samples from the same class."""
    folders = class_directories(train_directory)
    by_name = {folder.name.upper(): folder for folder in folders}
    selected_folder = by_name.get(class_name.upper(), folders[0])

    chosen = choose_files(
        list_image_files(selected_folder),
        amount,
        seed=RANDOM_SEED,
    )

    columns = 4
    rows = max(1, (len(chosen) + columns - 1) // columns)
    output_path = FIGURES_DIR / "same_class_variability.png"

    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for axis in axes_list:
        axis.axis("off")

    for axis, image_path in zip(axes_list, chosen):
        axis.imshow(_open_rgb(image_path))
        axis.set_title(f"Clase {selected_folder.name}")
        axis.axis("off")

    figure.suptitle("Variabilidad dentro de una misma clase")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path


def save_similar_signs_plot(train_directory: Path) -> Path:
    """Compare signs mentioned as visually similar in the instructions."""
    target_names = ["M", "N", "S", "U", "V", "R"]
    folders = class_directories(train_directory)
    by_name = {folder.name.upper(): folder for folder in folders}
    selected = [by_name[name] for name in target_names if name in by_name]

    output_path = FIGURES_DIR / "similar_signs_comparison.png"

    if not selected:
        return output_path

    figure, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes_list = list(axes.flat)

    for axis in axes_list:
        axis.axis("off")

    for index, class_folder in enumerate(selected[:6]):
        image_path = choose_files(
            list_image_files(class_folder),
            1,
            seed=RANDOM_SEED + index,
        )[0]

        axes_list[index].imshow(_open_rgb(image_path))
        axes_list[index].set_title(f"Clase {class_folder.name}")
        axes_list[index].axis("off")

    figure.suptitle("Clases con señas visualmente similares")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return output_path


def save_image_properties_plot(properties: pd.DataFrame) -> Path:
    """Create a chart with the most common image resolutions."""
    output_path = FIGURES_DIR / "image_properties.png"

    if properties.empty:
        return output_path

    resolution_counts = (
        properties.assign(
            resolution=properties["width"].astype(str)
            + " por "
            + properties["height"].astype(str)
        )["resolution"]
        .value_counts()
        .head(10)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(resolution_counts.index, resolution_counts.values)
    plt.title("Resoluciones encontradas en la muestra")
    plt.xlabel("Resolución")
    plt.ylabel("Cantidad de imágenes revisadas")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def build_summary(
    train_directory: Path,
    distribution: pd.DataFrame,
    properties: pd.DataFrame,
    damaged_files: list[str],
    full_validation: bool,
) -> dict:
    """Build a JSON summary used by the notebook."""
    total_images = int(distribution["image_count"].sum())
    minimum_count = int(distribution["image_count"].min())
    maximum_count = int(distribution["image_count"].max())
    balance_ratio = minimum_count / maximum_count if maximum_count else 0.0

    resolutions = Counter(
        f"{row.width}x{row.height}" for row in properties.itertuples()
    )
    formats = Counter(str(value) for value in properties["format"].tolist())
    color_modes = Counter(str(value) for value in properties["color_mode"].tolist())

    test_directory = resolve_test_directory()
    official_test_images = 0
    if test_directory is not None:
        official_test_images = sum(
            1 for path in test_directory.rglob("*") if path.is_file()
        )

    return {
        "train_directory": str(train_directory),
        "number_of_classes": int(len(distribution)),
        "class_names": distribution["class_name"].tolist(),
        "total_images": total_images,
        "minimum_images_per_class": minimum_count,
        "maximum_images_per_class": maximum_count,
        "balance_ratio": round(balance_ratio, 4),
        "most_common_classes": distribution.nlargest(
            3, "image_count"
        ).to_dict(orient="records"),
        "least_common_classes": distribution.nsmallest(
            3, "image_count"
        ).to_dict(orient="records"),
        "inspected_images": int(len(properties)),
        "full_validation": full_validation,
        "damaged_images_found": int(len(damaged_files)),
        "damaged_files": damaged_files,
        "resolutions": dict(resolutions),
        "formats": dict(formats),
        "color_modes": dict(color_modes),
        "official_test_directory": str(test_directory) if test_directory else None,
        "official_test_images": official_test_images,
    }


def run_exploratory_analysis(
    sample_per_class: int = DEFAULT_PROPERTY_SAMPLE_PER_CLASS,
    full_validation: bool = False,
) -> dict:
    """Run the complete exploratory analysis and save its outputs."""
    ensure_output_directories()
    train_directory = resolve_train_directory()

    distribution = collect_class_distribution(train_directory)
    properties, damaged_files = inspect_image_properties(
        train_directory,
        sample_per_class=sample_per_class,
        full_validation=full_validation,
    )

    distribution.to_csv(
        REPORTS_DIR / "class_distribution.csv",
        index=False,
    )
    properties.to_csv(
        REPORTS_DIR / "image_properties_sample.csv",
        index=False,
    )

    save_class_distribution_plot(distribution)
    save_sample_classes_plot(train_directory)
    save_same_class_variability_plot(train_directory)
    save_similar_signs_plot(train_directory)
    save_image_properties_plot(properties)

    summary = build_summary(
        train_directory,
        distribution,
        properties,
        damaged_files,
        full_validation,
    )

    summary_path = REPORTS_DIR / "exploratory_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("El análisis exploratorio terminó.")
    print(f"Clases encontradas: {summary['number_of_classes']}")
    print(f"Imágenes encontradas: {summary['total_images']}")
    print(f"Resumen guardado en: {summary_path}")

    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis exploratorio del dataset ASL Alphabet"
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=DEFAULT_PROPERTY_SAMPLE_PER_CLASS,
        help="Cantidad de imágenes por clase para revisar propiedades",
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Revisar todas las imágenes para detectar archivos dañados",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_exploratory_analysis(
        sample_per_class=arguments.sample_per_class,
        full_validation=arguments.full_validation,
    )
