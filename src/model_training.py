from __future__ import annotations

import json
import random
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageEnhance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC
from skimage.color import rgb2gray
from skimage.feature import hog

from .preprocess_images import create_split_manifest, save_split_outputs
from .utils import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SPLITS_DIR,
    ensure_output_directories,
)

RANDOM_SEED = 42
IMAGE_SIZE = 64
DEFAULT_BATCH_SIZE = 64
DEFAULT_TUNING_EPOCHS = 3
DEFAULT_AUGMENTED_EPOCHS = 5

MODELS_DIR = PROCESSED_DATA_DIR / "models"
HOG_DIR = PROCESSED_DATA_DIR / "hog"

CNN_BASE_CONFIGS = [
    {
        "name": "cnn_base_a",
        "filters": [24, 48],
        "dense_units": 128,
        "dropout": 0.30,
        "learning_rate": 0.001,
    },
    {
        "name": "cnn_base_b",
        "filters": [32, 64],
        "dense_units": 192,
        "dropout": 0.40,
        "learning_rate": 0.0005,
    },
]

CNN_IMPROVED_CONFIGS = [
    {
        "name": "cnn_improved_a",
        "filters": [24, 48, 96],
        "dropout": 0.35,
        "learning_rate": 0.001,
    },
    {
        "name": "cnn_improved_b",
        "filters": [32, 64, 128],
        "dropout": 0.45,
        "learning_rate": 0.0005,
    },
]

FULLY_CONNECTED_CONFIGS = [
    {
        "name": "fully_connected_a",
        "hidden_units": [256, 128],
        "dropout": 0.40,
        "learning_rate": 0.001,
    },
    {
        "name": "fully_connected_b",
        "hidden_units": [512, 256],
        "dropout": 0.50,
        "learning_rate": 0.0005,
    },
]

SVM_C_VALUES = [0.1, 1.0, 10.0]


def ensure_model_directories() -> None:
    ensure_output_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HOG_DIR.mkdir(parents=True, exist_ok=True)


def stable_class_names(values: list[str]) -> list[str]:
    preferred = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    preferred += ["del", "delete", "nothing", "space"]
    rank = {name.lower(): index for index, name in enumerate(preferred)}

    return sorted(
        sorted(set(values)),
        key=lambda value: (rank.get(value.lower(), len(rank)), value.lower()),
    )


def load_manifest() -> pd.DataFrame:
    manifest_path = SPLITS_DIR / "split_manifest.csv"

    if not manifest_path.exists():
        manifest = create_split_manifest(images_per_class=600, seed=RANDOM_SEED)
        save_split_outputs(manifest)
    else:
        manifest = pd.read_csv(manifest_path)

    required_columns = {"split", "class_name", "source_path"}
    missing_columns = required_columns.difference(manifest.columns)

    if missing_columns:
        raise ValueError(
            "El archivo split_manifest.csv no tiene las columnas necesarias: "
            + ", ".join(sorted(missing_columns))
        )

    return manifest


def get_class_names(manifest: pd.DataFrame) -> list[str]:
    class_names = stable_class_names(manifest["class_name"].astype(str).tolist())
    class_path = REPORTS_DIR / "class_names.json"
    class_path.write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return class_names


def absolute_path(relative_path: str) -> str:
    return str((PROJECT_ROOT / relative_path).resolve())


def load_tf_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.04, seed=RANDOM_SEED),
            tf.keras.layers.RandomTranslation(0.05, 0.05, seed=RANDOM_SEED + 1),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.08, 0.08),
                width_factor=(-0.08, 0.08),
                seed=RANDOM_SEED + 2,
            ),
            tf.keras.layers.RandomBrightness(
                factor=0.10,
                value_range=(0.0, 1.0),
                seed=RANDOM_SEED + 3,
            ),
            tf.keras.layers.RandomContrast(0.10, seed=RANDOM_SEED + 4),
        ],
        name="image_augmentation",
    )


def make_dataset(
    manifest: pd.DataFrame,
    split_name: str,
    class_names: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    augment: bool = False,
    shuffle: bool = False,
) -> tf.data.Dataset:
    rows = manifest[manifest["split"] == split_name].copy()

    if rows.empty:
        raise ValueError(f"No hay imágenes para el conjunto {split_name}.")

    class_to_index = {name: index for index, name in enumerate(class_names)}
    paths = rows["source_path"].map(absolute_path).to_numpy()
    labels = rows["class_name"].map(class_to_index).astype("int32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(rows),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(load_tf_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmentation = build_augmentation_layer()
        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_base(num_classes: int, config: dict) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            tf.keras.layers.Conv2D(
                config["filters"][0], 3, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(
                config["filters"][1], 3, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(config["dense_units"], activation="relu"),
            tf.keras.layers.Dropout(config["dropout"]),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name=config["name"],
    )
    return compile_model(model, config["learning_rate"])


def convolution_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return tf.keras.layers.MaxPooling2D()(x)


def build_cnn_improved(num_classes: int, config: dict) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = inputs

    for filters in config["filters"]:
        x = convolution_block(x, filters)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(config["dropout"])(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=config["name"])
    return compile_model(model, config["learning_rate"])


def build_fully_connected(num_classes: int, config: dict) -> tf.keras.Model:
    layers = [
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.Flatten(),
    ]

    for units in config["hidden_units"]:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
        layers.append(tf.keras.layers.Dropout(config["dropout"]))

    layers.append(tf.keras.layers.Dense(num_classes, activation="softmax"))
    model = tf.keras.Sequential(layers, name=config["name"])
    return compile_model(model, config["learning_rate"])


def training_callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        ),
    ]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def predict_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    true_labels = []
    predicted_labels = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        true_labels.extend(labels.numpy().tolist())
        predicted_labels.extend(predictions.tolist())

    return np.asarray(true_labels), np.asarray(predicted_labels)


def save_history_plots(history: tf.keras.callbacks.History, model_name: str) -> None:
    history_data = history.history

    plt.figure(figsize=(8, 5))
    plt.plot(history_data.get("accuracy", []), label="Entrenamiento")
    plt.plot(history_data.get("val_accuracy", []), label="Validación")
    plt.title(f"Accuracy de {model_name}")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"history_accuracy_{model_name}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_data.get("loss", []), label="Entrenamiento")
    plt.plot(history_data.get("val_loss", []), label="Validación")
    plt.title(f"Pérdida de {model_name}")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"history_loss_{model_name}.png", dpi=160)
    plt.close()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
) -> Path:
    output_path = FIGURES_DIR / f"confusion_{model_name}.png"

    figure, axis = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"Matriz de confusión de {model_name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def model_builder_for_family(family: str):
    if family == "cnn_base":
        return build_cnn_base
    if family == "cnn_improved":
        return build_cnn_improved
    if family == "fully_connected":
        return build_fully_connected
    raise ValueError(f"Familia desconocida: {family}")


def tune_keras_family(
    family: str,
    configs: list[dict],
    manifest: pd.DataFrame,
    class_names: list[str],
    epochs: int,
    batch_size: int,
) -> tuple[pd.DataFrame, dict, Path]:
    builder = model_builder_for_family(family)
    train_dataset = make_dataset(
        manifest,
        "train",
        class_names,
        batch_size=batch_size,
        augment=False,
        shuffle=True,
    )
    validation_dataset = make_dataset(
        manifest,
        "validation",
        class_names,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
    )

    rows = []
    best_config = None
    best_accuracy = -1.0
    best_model_path = MODELS_DIR / f"best_{family}.keras"

    for config in configs:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_SEED)
        model = builder(len(class_names), config)

        start_time = time.perf_counter()
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=training_callbacks(),
            verbose=1,
        )
        elapsed = time.perf_counter() - start_time

        y_true, y_pred = predict_dataset(model, validation_dataset)
        metrics = classification_metrics(y_true, y_pred)

        row = {
            "family": family,
            "model_name": config["name"],
            "augmentation": False,
            "training_time_seconds": round(elapsed, 2),
            "epochs_completed": len(history.history.get("loss", [])),
            "parameters": json.dumps(config, ensure_ascii=False),
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        rows.append(row)

        save_history_plots(history, config["name"])

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_config = config.copy()
            model.save(best_model_path)

    if best_config is None:
        raise RuntimeError(f"No se pudo entrenar la familia {family}.")

    return pd.DataFrame(rows), best_config, best_model_path


def evaluate_saved_keras_model(
    model_path: Path,
    model_name: str,
    family: str,
    manifest: pd.DataFrame,
    class_names: list[str],
    batch_size: int,
    augmentation: bool,
    training_time_seconds: float,
) -> dict:
    model = tf.keras.models.load_model(model_path)
    test_dataset = make_dataset(
        manifest,
        "test",
        class_names,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
    )
    y_true, y_pred = predict_dataset(model, test_dataset)
    metrics = classification_metrics(y_true, y_pred)
    confusion_path = save_confusion_matrix(y_true, y_pred, class_names, model_name)

    return {
        "family": family,
        "model_name": model_name,
        "model_type": "keras",
        "augmentation": augmentation,
        "training_time_seconds": round(training_time_seconds, 2),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "confusion_matrix_path": str(confusion_path.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        **metrics,
    }


def train_augmented_keras_model(
    family: str,
    config: dict,
    manifest: pd.DataFrame,
    class_names: list[str],
    epochs: int,
    batch_size: int,
) -> dict:
    builder = model_builder_for_family(family)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    model_name = f"{config['name']}_augmented"
    augmented_config = config.copy()
    augmented_config["name"] = model_name
    model = builder(len(class_names), augmented_config)

    train_dataset = make_dataset(
        manifest,
        "train",
        class_names,
        batch_size=batch_size,
        augment=True,
        shuffle=True,
    )
    validation_dataset = make_dataset(
        manifest,
        "validation",
        class_names,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
    )

    start_time = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=training_callbacks(),
        verbose=1,
    )
    elapsed = time.perf_counter() - start_time

    model_path = MODELS_DIR / f"{model_name}.keras"
    model.save(model_path)
    save_history_plots(history, model_name)

    return evaluate_saved_keras_model(
        model_path=model_path,
        model_name=model_name,
        family=family,
        manifest=manifest,
        class_names=class_names,
        batch_size=batch_size,
        augmentation=True,
        training_time_seconds=elapsed,
    )


def load_image_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        prepared = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(prepared, dtype=np.float32) / 255.0


def hog_descriptor(image_array: np.ndarray) -> np.ndarray:
    gray = rgb2gray(image_array)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)


def extract_hog_split(
    manifest: pd.DataFrame,
    split_name: str,
    class_names: list[str],
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = HOG_DIR / f"hog_{split_name}.npz"

    if use_cache and cache_path.exists():
        cached = np.load(cache_path)
        return cached["X"], cached["y"]

    rows = manifest[manifest["split"] == split_name].copy()
    class_to_index = {name: index for index, name in enumerate(class_names)}

    features = []
    labels = []
    total = len(rows)

    for index, row in enumerate(rows.itertuples(), start=1):
        image_path = PROJECT_ROOT / row.source_path
        features.append(hog_descriptor(load_image_array(image_path)))
        labels.append(class_to_index[row.class_name])

        if index % 1000 == 0 or index == total:
            print(f"HOG {split_name}: {index} de {total}")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def augment_pil_image(image_path: Path, seed: int) -> np.ndarray:
    generator = random.Random(seed)

    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )

        angle = generator.uniform(-10.0, 10.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR)

        width, height = image.size
        dx = int(generator.uniform(-0.05, 0.05) * width)
        dy = int(generator.uniform(-0.05, 0.05) * height)
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, dx, 0, 1, dy),
            resample=Image.Resampling.BILINEAR,
        )

        image = ImageEnhance.Brightness(image).enhance(
            generator.uniform(0.90, 1.10)
        )
        image = ImageEnhance.Contrast(image).enhance(
            generator.uniform(0.90, 1.10)
        )

    return np.asarray(image, dtype=np.float32) / 255.0


def augmented_hog_samples(
    manifest: pd.DataFrame,
    class_names: list[str],
    max_per_class: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    class_to_index = {name: index for index, name in enumerate(class_names)}
    train_rows = manifest[manifest["split"] == "train"].copy()

    sampled_groups = []
    for class_name, group in train_rows.groupby("class_name"):
        amount = min(max_per_class, len(group))
        sampled_groups.append(
            group.sample(n=amount, random_state=RANDOM_SEED)
        )

    sampled = pd.concat(sampled_groups, ignore_index=True)
    features = []
    labels = []
    total = len(sampled)

    for index, row in enumerate(sampled.itertuples(), start=1):
        image_path = PROJECT_ROOT / row.source_path
        augmented = augment_pil_image(image_path, RANDOM_SEED + index)
        features.append(hog_descriptor(augmented))
        labels.append(class_to_index[row.class_name])

        if index % 1000 == 0 or index == total:
            print(f"HOG aumentado: {index} de {total}")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )


def tune_and_train_svm(
    manifest: pd.DataFrame,
    class_names: list[str],
) -> tuple[pd.DataFrame, dict, float]:
    X_train, y_train = extract_hog_split(manifest, "train", class_names)
    X_validation, y_validation = extract_hog_split(
        manifest, "validation", class_names
    )

    rows = []
    best_c = None
    best_accuracy = -1.0
    best_model = None
    best_training_time = 0.0

    for c_value in SVM_C_VALUES:
        model = LinearSVC(
            C=c_value,
            dual="auto",
            max_iter=5000,
            random_state=RANDOM_SEED,
        )

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start_time

        predictions = model.predict(X_validation)
        metrics = classification_metrics(y_validation, predictions)

        rows.append(
            {
                "family": "hog_svm",
                "model_name": f"hog_svm_c_{c_value}",
                "augmentation": False,
                "training_time_seconds": round(elapsed, 2),
                "parameters": json.dumps({"C": c_value}),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_c = c_value
            best_model = model
            best_training_time = elapsed

    if best_model is None or best_c is None:
        raise RuntimeError("No se pudo entrenar el modelo HOG con SVM.")

    model_path = MODELS_DIR / "best_hog_svm.joblib"
    joblib.dump(
        {
            "model": best_model,
            "class_names": class_names,
            "C": best_c,
        },
        model_path,
    )

    return pd.DataFrame(rows), {"C": best_c, "model_path": model_path}, best_training_time


def evaluate_svm(
    model_path: Path,
    model_name: str,
    manifest: pd.DataFrame,
    class_names: list[str],
    augmentation: bool,
    training_time_seconds: float,
) -> dict:
    saved = joblib.load(model_path)
    model = saved["model"]
    X_test, y_test = extract_hog_split(manifest, "test", class_names)
    predictions = model.predict(X_test)
    metrics = classification_metrics(y_test, predictions)
    confusion_path = save_confusion_matrix(
        y_test, predictions, class_names, model_name
    )

    return {
        "family": "hog_svm",
        "model_name": model_name,
        "model_type": "svm",
        "augmentation": augmentation,
        "training_time_seconds": round(training_time_seconds, 2),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "confusion_matrix_path": str(confusion_path.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        **metrics,
    }


def train_augmented_svm(
    manifest: pd.DataFrame,
    class_names: list[str],
    best_c: float,
    max_augmented_per_class: int = 120,
) -> dict:
    X_train, y_train = extract_hog_split(manifest, "train", class_names)
    X_augmented, y_augmented = augmented_hog_samples(
        manifest,
        class_names,
        max_per_class=max_augmented_per_class,
    )

    X_full = np.concatenate([X_train, X_augmented], axis=0)
    y_full = np.concatenate([y_train, y_augmented], axis=0)

    model = LinearSVC(
        C=best_c,
        dual="auto",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )

    start_time = time.perf_counter()
    model.fit(X_full, y_full)
    elapsed = time.perf_counter() - start_time

    model_path = MODELS_DIR / "hog_svm_augmented.joblib"
    joblib.dump(
        {
            "model": model,
            "class_names": class_names,
            "C": best_c,
        },
        model_path,
    )

    return evaluate_svm(
        model_path=model_path,
        model_name="hog_svm_augmented",
        manifest=manifest,
        class_names=class_names,
        augmentation=True,
        training_time_seconds=elapsed,
    )


def select_best_tuning_row(tuning: pd.DataFrame, family: str) -> pd.Series:
    family_rows = tuning[tuning["family"] == family]
    return family_rows.sort_values(
        ["validation_accuracy", "validation_f1_macro"],
        ascending=False,
    ).iloc[0]


def config_by_name(configs: list[dict], model_name: str) -> dict:
    for config in configs:
        if config["name"] == model_name:
            return config.copy()
    raise KeyError(f"No se encontró la configuración {model_name}.")


def save_best_model_metadata(results: pd.DataFrame, class_names: list[str]) -> dict:
    best_row = results.sort_values(
        ["accuracy", "f1_macro"],
        ascending=False,
    ).iloc[0]

    metadata = {
        "model_name": best_row["model_name"],
        "model_type": best_row["model_type"],
        "family": best_row["family"],
        "augmentation": bool(best_row["augmentation"]),
        "accuracy": float(best_row["accuracy"]),
        "precision_macro": float(best_row["precision_macro"]),
        "recall_macro": float(best_row["recall_macro"]),
        "f1_macro": float(best_row["f1_macro"]),
        "model_path": best_row["model_path"],
        "confusion_matrix_path": best_row["confusion_matrix_path"],
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
    }

    output_path = REPORTS_DIR / "best_model.json"
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def run_all_training(
    tuning_epochs: int = DEFAULT_TUNING_EPOCHS,
    augmented_epochs: int = DEFAULT_AUGMENTED_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    svm_augmented_per_class: int = 120,
) -> dict:
    ensure_model_directories()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    manifest = load_manifest()
    class_names = get_class_names(manifest)

    print(f"Clases: {len(class_names)}")
    print(f"Imágenes en el manifiesto: {len(manifest)}")

    tuning_frames = []
    best_configs = {}
    best_model_paths = {}

    family_specs = [
        ("cnn_base", CNN_BASE_CONFIGS),
        ("cnn_improved", CNN_IMPROVED_CONFIGS),
        ("fully_connected", FULLY_CONNECTED_CONFIGS),
    ]

    for family, configs in family_specs:
        print(f"\nAjustando {family}")
        tuning_frame, best_config, best_path = tune_keras_family(
            family=family,
            configs=configs,
            manifest=manifest,
            class_names=class_names,
            epochs=tuning_epochs,
            batch_size=batch_size,
        )
        tuning_frames.append(tuning_frame)
        best_configs[family] = best_config
        best_model_paths[family] = best_path

    print("\nAjustando HOG con SVM")
    svm_tuning, best_svm, svm_time = tune_and_train_svm(manifest, class_names)
    tuning_frames.append(svm_tuning)

    tuning_results = pd.concat(tuning_frames, ignore_index=True)
    tuning_path = REPORTS_DIR / "model_tuning_results.csv"
    tuning_results.to_csv(tuning_path, index=False)

    final_rows = []

    for family in ["cnn_base", "cnn_improved", "fully_connected"]:
        best_tuning = select_best_tuning_row(tuning_results, family)
        best_name = str(best_tuning["model_name"])
        baseline_result = evaluate_saved_keras_model(
            model_path=best_model_paths[family],
            model_name=best_name,
            family=family,
            manifest=manifest,
            class_names=class_names,
            batch_size=batch_size,
            augmentation=False,
            training_time_seconds=float(best_tuning["training_time_seconds"]),
        )
        final_rows.append(baseline_result)

    svm_baseline = evaluate_svm(
        model_path=best_svm["model_path"],
        model_name=f"hog_svm_c_{best_svm['C']}",
        manifest=manifest,
        class_names=class_names,
        augmentation=False,
        training_time_seconds=svm_time,
    )
    final_rows.append(svm_baseline)

    print("\nEntrenando versiones con aumento de datos")
    for family in ["cnn_base", "cnn_improved", "fully_connected"]:
        augmented_result = train_augmented_keras_model(
            family=family,
            config=best_configs[family],
            manifest=manifest,
            class_names=class_names,
            epochs=augmented_epochs,
            batch_size=batch_size,
        )
        final_rows.append(augmented_result)

    svm_augmented = train_augmented_svm(
        manifest=manifest,
        class_names=class_names,
        best_c=float(best_svm["C"]),
        max_augmented_per_class=svm_augmented_per_class,
    )
    final_rows.append(svm_augmented)

    results = pd.DataFrame(final_rows)
    results = results.sort_values(
        ["accuracy", "f1_macro"],
        ascending=False,
    ).reset_index(drop=True)

    results_path = REPORTS_DIR / "all_model_results.csv"
    results.to_csv(results_path, index=False)

    best_model = save_best_model_metadata(results, class_names)

    print("\nEntrenamiento completo")
    print(f"Resultados: {results_path}")
    print(f"Mejor modelo: {best_model['model_name']}")
    print(f"Accuracy de prueba: {best_model['accuracy']:.4f}")

    return {
        "tuning_results": tuning_results,
        "results": results,
        "best_model": best_model,
        "tuning_results_path": str(tuning_path),
        "results_path": str(results_path),
    }


if __name__ == "__main__":
    run_all_training()
