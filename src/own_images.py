from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score

from .model_training import IMAGE_SIZE, hog_descriptor
from .utils import IMAGE_EXTENSIONS, PROCESSED_DATA_DIR, PROJECT_ROOT, REPORTS_DIR

OWN_IMAGES_DIR = PROJECT_ROOT / "data" / "own_images"


def normalize_label(label: str) -> str:
    value = label.strip()
    lowered = value.lower()

    if lowered == "delete":
        return "del"
    if lowered in {"del", "nothing", "space"}:
        return lowered
    return value.upper()


def expected_label_from_filename(path: Path) -> str:
    first_part = path.stem.split("_")[0]
    return normalize_label(first_part)


def collect_own_images() -> pd.DataFrame:
    if not OWN_IMAGES_DIR.exists():
        return pd.DataFrame(columns=["member", "expected_label", "image_path"])

    records = []

    for member_folder in sorted(OWN_IMAGES_DIR.iterdir()):
        if not member_folder.is_dir():
            continue

        for image_path in sorted(member_folder.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            records.append(
                {
                    "member": member_folder.name,
                    "expected_label": expected_label_from_filename(image_path),
                    "image_path": str(image_path.relative_to(PROJECT_ROOT)).replace(
                        "\\", "/"
                    ),
                }
            )

    return pd.DataFrame(records)


def prepare_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        prepared = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(prepared, dtype=np.float32) / 255.0


def load_best_model_metadata() -> dict:
    metadata_path = REPORTS_DIR / "best_model.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "No existe best_model.json. Ejecuta primero el entrenamiento de modelos."
        )

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def predict_with_keras(
    model_path: Path,
    class_names: list[str],
    image_paths: list[Path],
) -> tuple[list[str], list[float]]:
    model = tf.keras.models.load_model(model_path)
    predictions = []
    confidence = []

    for image_path in image_paths:
        image = prepare_image(image_path)
        probabilities = model.predict(image[None, ...], verbose=0)[0]
        index = int(np.argmax(probabilities))
        predictions.append(class_names[index])
        confidence.append(float(probabilities[index]))

    return predictions, confidence


def predict_with_svm(
    model_path: Path,
    image_paths: list[Path],
) -> tuple[list[str], list[float]]:
    saved = joblib.load(model_path)
    model = saved["model"]
    class_names = saved["class_names"]

    features = np.asarray(
        [hog_descriptor(prepare_image(path)) for path in image_paths],
        dtype=np.float32,
    )

    predicted_indices = model.predict(features)
    decision_scores = model.decision_function(features)
    max_scores = np.max(decision_scores, axis=1)

    predictions = [class_names[int(index)] for index in predicted_indices]
    confidence = [float(score) for score in max_scores]
    return predictions, confidence


def build_member_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for member, group in predictions.groupby("member"):
        distinct_letters = int(group["expected_label"].nunique())
        correct = group["expected_label"] == group["predicted_label"]

        rows.append(
            {
                "member": member,
                "images": int(len(group)),
                "distinct_letters": distinct_letters,
                "minimum_five_letters_met": distinct_letters >= 5,
                "accuracy": float(correct.mean()) if len(group) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def evaluate_own_images() -> dict | None:
    images = collect_own_images()

    if images.empty:
        print("No se encontraron fotos propias.")
        print("Crea carpetas dentro de data/own_images con el nombre de cada integrante.")
        print("Usa nombres como A_1.jpg, B_1.jpg, C_1.jpg, D_1.jpg y E_1.jpg.")
        return None

    metadata = load_best_model_metadata()
    model_path = PROJECT_ROOT / metadata["model_path"]
    class_names = metadata["class_names"]
    image_paths = [PROJECT_ROOT / value for value in images["image_path"]]

    if metadata["model_type"] == "keras":
        predicted_labels, confidence = predict_with_keras(
            model_path,
            class_names,
            image_paths,
        )
    elif metadata["model_type"] == "svm":
        predicted_labels, confidence = predict_with_svm(
            model_path,
            image_paths,
        )
    else:
        raise ValueError(f"Tipo de modelo desconocido: {metadata['model_type']}")

    predictions = images.copy()
    predictions["predicted_label"] = predicted_labels
    predictions["confidence"] = confidence
    predictions["correct"] = (
        predictions["expected_label"] == predictions["predicted_label"]
    )

    predictions_path = REPORTS_DIR / "own_image_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    member_summary = build_member_summary(predictions)
    member_summary_path = REPORTS_DIR / "own_image_member_summary.csv"
    member_summary.to_csv(member_summary_path, index=False)

    overall_accuracy = float(
        accuracy_score(
            predictions["expected_label"],
            predictions["predicted_label"],
        )
    )

    print(f"Fotos propias evaluadas: {len(predictions)}")
    print(f"Accuracy general: {overall_accuracy:.4f}")

    for row in member_summary.itertuples():
        if not row.minimum_five_letters_met:
            print(
                f"Aviso: {row.member} tiene {row.distinct_letters} letras distintas. "
                "Debe tener al menos 5."
            )

    return {
        "predictions": predictions,
        "member_summary": member_summary,
        "overall_accuracy": overall_accuracy,
        "predictions_path": str(predictions_path),
        "member_summary_path": str(member_summary_path),
    }


if __name__ == "__main__":
    evaluate_own_images()
