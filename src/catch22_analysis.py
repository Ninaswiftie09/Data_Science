from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from catch22_core import FEATURE_NAMES, SHORT_NAMES, catch22_all

SEED = 231088
TRAIN_RATIO = 0.70
LOOKBACK = 24
VALIDATION_MONTHS = 36
EPOCHS = 100


EDA_RESULTS = pd.DataFrame(
    [
        ["Total", 0.256, 1.283, 0.381, -74.354],
        ["Vía Aérea", 0.163, -0.465, 0.308, -72.797],
        ["Vía Terrestre", 0.230, 2.130, 0.463, -75.225],
        ["Vía Marítima", 0.051, -44.653, 1.134, np.nan],
        ["Región América Del Centro", 0.263, 1.135, 0.419, -74.431],
        ["Región América Del Norte", 0.083, -0.135, 0.410, -74.144],
        ["Región Europa", 0.211, -1.957, 0.430, -71.830],
    ],
    columns=[
        "Serie",
        "Fuerza estacional",
        "Crecimiento anual aproximado (%)",
        "Coeficiente de variación",
        "Cambio 2020 frente a 2019 (%)",
    ],
)

LAB1_RESULTS = pd.DataFrame(
    [
        {
            "Serie": "Vía Aérea",
            "Modelo Laboratorio 1": "SARIMA (1,1,1) x (0,0,1,12)",
            "MAE Laboratorio 1": 31483.389,
            "RMSE Laboratorio 1": 36548.987,
        },
        {
            "Serie": "Vía Terrestre",
            "Modelo Laboratorio 1": "Prophet",
            "MAE Laboratorio 1": 47712.983,
            "RMSE Laboratorio 1": 57657.637,
        },
    ]
)


@dataclass(frozen=True)
class HybridConfig:
    feature_hidden: int
    dropout: float
    learning_rate: float


class Catch22LSTM(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=24,
            num_layers=1,
            batch_first=True,
        )
        self.feature_branch = nn.Sequential(
            nn.Linear(22, config.feature_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        combined = 24 + config.feature_hidden
        self.head = nn.Sequential(
            nn.Linear(combined, max(12, combined // 2)),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(max(12, combined // 2), 1),
        )

    def forward(self, sequence, features):
        output, _ = self.lstm(sequence)
        lstm_state = output[:, -1, :]
        feature_state = self.feature_branch(features)
        return self.head(torch.cat([lstm_state, feature_state], dim=1))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parents[1]]
    for candidate in candidates:
        file_path = candidate / "data" / "raw" / "Base_Migracion_2009-2026jun.xlsx"
        if file_path.exists():
            return candidate.resolve()
    raise FileNotFoundError("No se encontró data/raw/Base_Migracion_2009-2026jun.xlsx")


def slug(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_all_series(root: Path):
    file_path = root / "data" / "raw" / "Base_Migracion_2009-2026jun.xlsx"
    df = pd.read_excel(file_path, sheet_name="Datos")
    df.columns = df.columns.str.strip()

    for column in ["Año", "Mes cod", "Viajero"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Fecha"] = pd.to_datetime(
        {"year": df["Año"], "month": df["Mes cod"], "day": 1},
        errors="coerce",
    )
    df = df.dropna(subset=["Fecha", "Viajero"])
    df = df[df["Tipo de Viajero"].isin(["Turista", "Excursionista"])].copy()

    total = (
        df.groupby("Fecha")["Viajero"]
        .sum()
        .sort_index()
        .asfreq("MS", fill_value=0)
        .astype(float)
    )

    routes = (
        df.pivot_table(
            index="Fecha",
            columns="Vía",
            values="Viajero",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
        .reindex(total.index, fill_value=0)
        .astype(float)
    )

    regions = (
        df.pivot_table(
            index="Fecha",
            columns="Región dos",
            values="Viajero",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
        .reindex(total.index, fill_value=0)
        .astype(float)
    )

    series = {
        "Total": total,
        "Vía Aérea": routes["Aérea"],
        "Vía Terrestre": routes["Terrestre"],
        "Vía Marítima": routes["Marítima"],
        "Región América Del Centro": regions["América Del Centro"],
        "Región América Del Norte": regions["América Del Norte"],
        "Región Europa": regions["Europa"],
    }

    categories = {
        "Total": "Serie total",
        "Vía Aérea": "Vía",
        "Vía Terrestre": "Vía",
        "Vía Marítima": "Vía",
        "Región América Del Centro": "Región",
        "Región América Del Norte": "Región",
        "Región Europa": "Región",
    }

    n_train = int(len(total) * TRAIN_RATIO)
    train_end = total.index[n_train - 1]
    test_start = total.index[n_train]
    return series, categories, train_end, test_start


def extract_feature_matrix(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, values in series.items():
        result = catch22_all(values.to_numpy(dtype=float))
        rows.append(pd.Series(result["values"], index=result["names"], name=name))
    return pd.DataFrame(rows)


def choose_clusters(values: np.ndarray):
    rows = []
    best_score = -np.inf
    best_labels = None
    best_k = None
    for k in range(2, min(5, len(values))):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(values)
        score = silhouette_score(values, labels)
        rows.append({"Número de grupos": k, "Silhouette": score})
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k
    return best_k, best_labels, pd.DataFrame(rows)


def metrics(real, prediction):
    real = np.asarray(real, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    denominator = np.abs(real) + np.abs(prediction)
    smape = 100 * np.mean(
        np.where(denominator == 0, 0, 2 * np.abs(prediction - real) / denominator)
    )
    return {
        "MAE": mean_absolute_error(real, prediction),
        "RMSE": np.sqrt(mean_squared_error(real, prediction)),
        "sMAPE (%)": smape,
    }


def prepare_hybrid_training(values: np.ndarray, target_scaler: StandardScaler):
    scaled = target_scaler.transform(np.log1p(values).reshape(-1, 1)).ravel()
    sequences, features, targets = [], [], []
    for index in range(LOOKBACK, len(values)):
        sequences.append(scaled[index - LOOKBACK : index])
        features.append(catch22_all(values[index - LOOKBACK : index])["values"])
        targets.append(scaled[index])
    return (
        np.asarray(sequences, dtype=np.float32).reshape(-1, LOOKBACK, 1),
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32).reshape(-1, 1),
    )


def train_hybrid(
    train_values: np.ndarray,
    config: HybridConfig,
    seed: int,
):
    set_seed(seed)
    target_scaler = StandardScaler()
    target_scaler.fit(np.log1p(train_values).reshape(-1, 1))
    x_sequence, x_features, y = prepare_hybrid_training(train_values, target_scaler)

    feature_scaler = StandardScaler()
    x_features = feature_scaler.fit_transform(x_features).astype(np.float32)

    sequence_tensor = torch.from_numpy(x_sequence)
    feature_tensor = torch.from_numpy(x_features)
    target_tensor = torch.from_numpy(y)

    model = Catch22LSTM(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(sequence_tensor, feature_tensor)
        loss = loss_function(prediction, target_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"Época": epoch, "Pérdida": float(loss.item())})

    return model, target_scaler, feature_scaler, pd.DataFrame(history)


def walk_forward_hybrid(
    model,
    train_values: np.ndarray,
    future_values: np.ndarray,
    target_scaler: StandardScaler,
    feature_scaler: StandardScaler,
):
    history = list(np.asarray(train_values, dtype=float))
    predictions = []
    model.eval()

    with torch.no_grad():
        for actual in np.asarray(future_values, dtype=float):
            raw_window = np.asarray(history[-LOOKBACK:], dtype=float)
            scaled_window = target_scaler.transform(
                np.log1p(raw_window).reshape(-1, 1)
            ).astype(np.float32)
            features = np.asarray(catch22_all(raw_window)["values"], dtype=np.float32).reshape(1, -1)
            features = feature_scaler.transform(features).astype(np.float32)

            sequence_tensor = torch.from_numpy(scaled_window.reshape(1, LOOKBACK, 1))
            feature_tensor = torch.from_numpy(features)
            prediction_scaled = float(model(sequence_tensor, feature_tensor).item())
            prediction_log = target_scaler.inverse_transform([[prediction_scaled]])[0, 0]
            predictions.append(max(0.0, float(np.expm1(prediction_log))))
            history.append(float(actual))

    return np.asarray(predictions)


def tune_hybrid(train: pd.Series):
    tune_train = train.iloc[:-VALIDATION_MONTHS]
    validation = train.iloc[-VALIDATION_MONTHS:]
    configs = [
        HybridConfig(8, 0.0, 0.003),
        HybridConfig(16, 0.10, 0.003),
        HybridConfig(32, 0.10, 0.003),
        HybridConfig(16, 0.20, 0.001),
    ]
    rows = []

    for index, config in enumerate(configs, start=1):
        model, target_scaler, feature_scaler, _ = train_hybrid(
            tune_train.to_numpy(dtype=float), config, SEED + index
        )
        prediction = walk_forward_hybrid(
            model,
            tune_train.to_numpy(dtype=float),
            validation.to_numpy(dtype=float),
            target_scaler,
            feature_scaler,
        )
        score = metrics(validation.values, prediction)
        rows.append(
            {
                "Configuración": index,
                "Unidades rama catch22": config.feature_hidden,
                "Dropout": config.dropout,
                "Learning rate": config.learning_rate,
                **{f"{key} validación": value for key, value in score.items()},
            }
        )

    table = pd.DataFrame(rows).sort_values("RMSE validación").reset_index(drop=True)
    best_index = int(table.iloc[0]["Configuración"])
    return configs[best_index - 1], table


def table_image(dataframe: pd.DataFrame, title: str, path: Path, decimals=None):
    data = dataframe.copy()
    decimals = decimals or []
    for column in decimals:
        if column in data.columns:
            data[column] = data[column].map(lambda value: f"{float(value):,.2f}")

    width = max(10, 1.45 * len(data.columns))
    height = max(2.6, 0.47 * len(data) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.set_title(title, pad=16, fontsize=14)
    artist = ax.table(
        cellText=data.values,
        colLabels=data.columns,
        cellLoc="center",
        loc="center",
    )
    artist.auto_set_font_size(False)
    artist.set_fontsize(8.5)
    artist.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def annotate_points(ax, coordinates, labels):
    for (x_value, y_value), label in zip(coordinates, labels):
        ax.annotate(label, (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=9)


def main():
    set_seed()
    root = project_root()
    output = root / "data" / "processed" / "lab2_final"
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    series, categories, train_end, test_start = load_all_series(root)

    # ------------------------------------------------------------------
    # catch22: matriz, estandarización, PCA, clustering y distancias
    # ------------------------------------------------------------------
    raw_features = extract_feature_matrix(series)
    feature_scaler = StandardScaler()
    standardised_values = feature_scaler.fit_transform(raw_features)
    standardised = pd.DataFrame(
        standardised_values,
        index=raw_features.index,
        columns=SHORT_NAMES,
    )

    pca = PCA(n_components=2)
    pca_values = pca.fit_transform(standardised_values)
    pca_table = pd.DataFrame(pca_values, index=raw_features.index, columns=["PC1", "PC2"])
    pca_table["Categoría"] = [categories[name] for name in pca_table.index]

    best_k, labels, silhouette = choose_clusters(standardised_values)
    cluster_table = pca_table.copy()
    cluster_table["Grupo"] = labels + 1
    cluster_table["Distancia promedio"] = squareform(pdist(standardised_values)).mean(axis=1)
    cluster_table["Categoría"] = [categories[name] for name in cluster_table.index]

    distances = pd.DataFrame(
        squareform(pdist(standardised_values)),
        index=raw_features.index,
        columns=raw_features.index,
    )

    correlation = standardised.corr()
    loadings = pd.DataFrame(
        pca.components_.T,
        index=SHORT_NAMES,
        columns=["PC1", "PC2"],
    )
    loadings["Importancia ponderada"] = np.sqrt(
        (loadings["PC1"] * pca.explained_variance_ratio_[0]) ** 2
        + (loadings["PC2"] * pca.explained_variance_ratio_[1]) ** 2
    )
    loadings = loadings.sort_values("Importancia ponderada", ascending=False)

    raw_features.to_csv(tables / "catch22_feature_matrix.csv", encoding="utf-8-sig")
    standardised.to_csv(tables / "catch22_standardised_matrix.csv", encoding="utf-8-sig")
    pca_table.to_csv(tables / "catch22_pca_coordinates.csv", encoding="utf-8-sig")
    cluster_table.to_csv(tables / "catch22_clusters.csv", encoding="utf-8-sig")
    distances.to_csv(tables / "catch22_series_distances.csv", encoding="utf-8-sig")
    correlation.to_csv(tables / "catch22_feature_correlations.csv", encoding="utf-8-sig")
    loadings.to_csv(tables / "catch22_pca_loadings.csv", encoding="utf-8-sig")
    silhouette.to_csv(tables / "catch22_silhouette.csv", index=False, encoding="utf-8-sig")

    eda_catch22 = EDA_RESULTS.merge(
        cluster_table.reset_index().rename(columns={"index": "Serie"})[
            ["Serie", "Grupo", "Distancia promedio"]
        ],
        on="Serie",
    )
    eda_catch22.to_csv(tables / "catch22_vs_eda.csv", index=False, encoding="utf-8-sig")
    table_image(
        eda_catch22,
        "Comparación del análisis exploratorio y los resultados de catch22",
        figures / "catch22_vs_eda_table.png",
        [
            "Fuerza estacional",
            "Crecimiento anual aproximado (%)",
            "Coeficiente de variación",
            "Cambio 2020 frente a 2019 (%)",
            "Distancia promedio",
        ],
    )

    table_image(
        cluster_table.reset_index().rename(columns={"index": "Serie"})[
            ["Serie", "Categoría", "Grupo", "PC1", "PC2", "Distancia promedio"]
        ],
        "Grupos y coordenadas obtenidas con catch22",
        figures / "catch22_cluster_table.png",
        ["PC1", "PC2", "Distancia promedio"],
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    for category in cluster_table["Categoría"].unique():
        subset = cluster_table[cluster_table["Categoría"] == category]
        ax.scatter(subset["PC1"], subset["PC2"], s=80, label=category)
    annotate_points(ax, cluster_table[["PC1", "PC2"]].values, cluster_table.index)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_title(
        "PCA de las series a partir de las 22 características catch22\n"
        f"PC1: {100*pca.explained_variance_ratio_[0]:.1f}% | "
        f"PC2: {100*pca.explained_variance_ratio_[1]:.1f}%"
    )
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "catch22_pca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    linkage_matrix = linkage(standardised_values, method="ward")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=raw_features.index.tolist(), leaf_rotation=30, ax=ax)
    ax.set_title("Clustering jerárquico de las series con características catch22")
    ax.set_ylabel("Distancia de Ward")
    fig.tight_layout()
    fig.savefig(figures / "catch22_dendrogram.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 6))
    image = ax.imshow(standardised.values, aspect="auto")
    ax.set_xticks(range(len(SHORT_NAMES)), SHORT_NAMES, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(standardised.index)), standardised.index)
    ax.set_title("Mapa de calor de las características catch22 estandarizadas")
    fig.colorbar(image, ax=ax, label="Valor estandarizado")
    fig.tight_layout()
    fig.savefig(figures / "catch22_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(correlation.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(SHORT_NAMES)), SHORT_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(SHORT_NAMES)), SHORT_NAMES, fontsize=7)
    ax.set_title("Matriz de correlaciones entre características catch22")
    fig.colorbar(image, ax=ax, label="Correlación")
    fig.tight_layout()
    fig.savefig(figures / "catch22_correlations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(distances.values)
    ax.set_xticks(range(len(distances.index)), distances.index, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(distances.index)), distances.index, fontsize=8)
    for row in range(len(distances)):
        for column in range(len(distances)):
            ax.text(column, row, f"{distances.iloc[row, column]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Mapa de distancias entre series")
    fig.colorbar(image, ax=ax, label="Distancia euclidiana")
    fig.tight_layout()
    fig.savefig(figures / "catch22_distances.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    top_loadings = loadings.head(10).sort_values("Importancia ponderada")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(top_loadings.index, top_loadings["Importancia ponderada"])
    ax.set_title("Características más importantes para separar las series en el PCA")
    ax.set_xlabel("Importancia ponderada en PC1 y PC2")
    fig.tight_layout()
    fig.savefig(figures / "catch22_pca_loadings.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Comparación LSTM contra los mejores modelos del Laboratorio 1
    # ------------------------------------------------------------------
    lstm_metrics_path = root / "data" / "processed" / "lab2" / "tables" / "lstm_test_metrics.csv"
    predictions_path = root / "data" / "processed" / "lab2" / "tables" / "lstm_predictions.csv"
    if not lstm_metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(
            "Primero ejecute python src/lstm_models.py para generar los resultados del avance."
        )

    lstm_metrics = pd.read_csv(lstm_metrics_path)
    best_lstm = (
        lstm_metrics.sort_values("RMSE")
        .groupby("Serie", as_index=False)
        .first()[["Serie", "Modelo", "MAE", "RMSE", "sMAPE (%)"]]
        .rename(
            columns={
                "Modelo": "Mejor LSTM",
                "MAE": "MAE LSTM",
                "RMSE": "RMSE LSTM",
                "sMAPE (%)": "sMAPE LSTM (%)",
            }
        )
    )
    lab_comparison = LAB1_RESULTS.merge(best_lstm, on="Serie")
    lab_comparison["Mejora MAE (%)"] = 100 * (
        1 - lab_comparison["MAE LSTM"] / lab_comparison["MAE Laboratorio 1"]
    )
    lab_comparison["Mejora RMSE (%)"] = 100 * (
        1 - lab_comparison["RMSE LSTM"] / lab_comparison["RMSE Laboratorio 1"]
    )
    lab_comparison.to_csv(tables / "lstm_vs_lab1.csv", index=False, encoding="utf-8-sig")

    table_image(
        lab_comparison,
        "Comparación del mejor LSTM con el mejor modelo del Laboratorio 1",
        figures / "lstm_vs_lab1_table.png",
        [
            "MAE Laboratorio 1",
            "RMSE Laboratorio 1",
            "MAE LSTM",
            "RMSE LSTM",
            "sMAPE LSTM (%)",
            "Mejora MAE (%)",
            "Mejora RMSE (%)",
        ],
    )

    chart = lab_comparison.set_index("Serie")[["RMSE Laboratorio 1", "RMSE LSTM"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    chart.plot(kind="bar", ax=ax)
    ax.set_title("RMSE del mejor modelo anterior frente al mejor LSTM")
    ax.set_xlabel("Serie")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(figures / "lstm_vs_lab1_rmse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # LSTM híbrida con catch22 para Vía Aérea
    # ------------------------------------------------------------------
    aerial = series["Vía Aérea"]
    train = aerial.loc[:train_end]
    test = aerial.loc[test_start:]
    best_config, tuning_table = tune_hybrid(train)
    hybrid_model, target_scaler, window_feature_scaler, hybrid_history = train_hybrid(
        train.values.astype(float), best_config, SEED + 99
    )
    hybrid_prediction = walk_forward_hybrid(
        hybrid_model,
        train.values.astype(float),
        test.values.astype(float),
        target_scaler,
        window_feature_scaler,
    )
    hybrid_score = metrics(test.values, hybrid_prediction)

    predictions = pd.read_csv(predictions_path, parse_dates=["Fecha"])
    aerial_predictions = predictions[predictions["Serie"] == "Vía Aérea"].copy()
    standard_prediction = aerial_predictions["LSTM simple"].to_numpy(dtype=float)
    standard_score = metrics(test.values, standard_prediction)

    hybrid_comparison = pd.DataFrame(
        [
            {"Modelo": "LSTM simple", **standard_score},
            {"Modelo": "LSTM + catch22", **hybrid_score},
        ]
    ).sort_values("RMSE")
    hybrid_comparison["Cambio RMSE frente a LSTM simple (%)"] = 100 * (
        1 - hybrid_comparison["RMSE"] / standard_score["RMSE"]
    )

    tuning_table.to_csv(tables / "catch22_lstm_tuning.csv", index=False, encoding="utf-8-sig")
    hybrid_comparison.to_csv(tables / "catch22_lstm_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "Fecha": test.index,
            "Real": test.values,
            "LSTM simple": standard_prediction,
            "LSTM + catch22": hybrid_prediction,
        }
    ).to_csv(tables / "catch22_lstm_predictions.csv", index=False, encoding="utf-8-sig")

    table_image(
        tuning_table,
        "Tuneo del modelo LSTM con características catch22",
        figures / "catch22_lstm_tuning_table.png",
        ["Dropout", "Learning rate", "MAE validación", "RMSE validación", "sMAPE (%) validación"],
    )
    table_image(
        hybrid_comparison,
        "Comparación de la LSTM simple y la LSTM con catch22",
        figures / "catch22_lstm_comparison_table.png",
        ["MAE", "RMSE", "sMAPE (%)", "Cambio RMSE frente a LSTM simple (%)"],
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(train.iloc[-48:].index, train.iloc[-48:].values, label="Entrenamiento (últimos 48 meses)")
    ax.plot(test.index, test.values, label="Valor real")
    ax.plot(test.index, standard_prediction, label="LSTM simple")
    ax.plot(test.index, hybrid_prediction, label="LSTM + catch22")
    ax.axvline(test.index.min(), linestyle="--", label="Inicio de prueba")
    ax.set_title("Predicción de Vía Aérea con LSTM simple y LSTM + catch22")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Viajeros")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "catch22_lstm_predictions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hybrid_history["Época"], hybrid_history["Pérdida"])
    ax.set_title("Pérdida de entrenamiento de la LSTM con catch22")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE estandarizado")
    fig.tight_layout()
    fig.savefig(figures / "catch22_lstm_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Resumen automático para construir el informe sin recalcular nada.
    upper = np.triu_indices_from(distances.values, k=1)
    closest_position = int(np.argmin(distances.values[upper]))
    closest_pair = (
        distances.index[upper[0][closest_position]],
        distances.columns[upper[1][closest_position]],
        float(distances.values[upper][closest_position]),
    )
    farthest_position = int(np.argmax(distances.values[upper]))
    farthest_pair = (
        distances.index[upper[0][farthest_position]],
        distances.columns[upper[1][farthest_position]],
        float(distances.values[upper][farthest_position]),
    )
    atypical = cluster_table["Distancia promedio"].idxmax()

    summary = pd.DataFrame(
        {
            "Indicador": [
                "Número de grupos elegido",
                "Silhouette del agrupamiento",
                "Serie más atípica",
                "Par de series más parecido",
                "Distancia del par más parecido",
                "Par de series más diferente",
                "Distancia del par más diferente",
                "Varianza explicada por PC1",
                "Varianza explicada por PC2",
                "Mejor configuración catch22 LSTM",
            ],
            "Resultado": [
                best_k,
                float(silhouette.loc[silhouette["Número de grupos"] == best_k, "Silhouette"].iloc[0]),
                atypical,
                f"{closest_pair[0]} / {closest_pair[1]}",
                closest_pair[2],
                f"{farthest_pair[0]} / {farthest_pair[1]}",
                farthest_pair[2],
                float(pca.explained_variance_ratio_[0]),
                float(pca.explained_variance_ratio_[1]),
                (
                    f"{best_config.feature_hidden} unidades, "
                    f"dropout {best_config.dropout}, lr {best_config.learning_rate}"
                ),
            ],
        }
    )
    summary.to_csv(tables / "final_summary.csv", index=False, encoding="utf-8-sig")

    print("\nResumen catch22")
    print(summary.to_string(index=False))
    print("\nCaracterísticas principales")
    print(loadings.head(10).to_string())
    print("\nGrupos")
    print(cluster_table[["Categoría", "Grupo", "PC1", "PC2", "Distancia promedio"]].to_string())
    print("\nComparación con Laboratorio 1")
    print(lab_comparison.to_string(index=False))
    print("\nComparación de la LSTM híbrida")
    print(hybrid_comparison.to_string(index=False))
    print(f"\nResultados guardados en {output}")


if __name__ == "__main__":
    main()
