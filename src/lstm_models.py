from dataclasses import dataclass
from pathlib import Path
import itertools
import random
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 231088
TRAIN_RATIO = 0.70
VALIDATION_MONTHS = 36
EPOCHS = 100


@dataclass(frozen=True)
class Config:
    model: str
    lookback: int
    hidden: int
    layers: int
    dropout: float
    learning_rate: float = 0.003


class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        dropout = config.dropout if config.layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=config.hidden,
            num_layers=config.layers,
            dropout=dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(config.hidden, config.hidden // 2),
            nn.ReLU(),
            nn.Linear(config.hidden // 2, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def project_root():
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parents[1],
    ]

    for candidate in candidates:
        path = (
            candidate
            / "data"
            / "raw"
            / "Base_Migracion_2009-2026jun.xlsx"
        )

        if path.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "No se encontró "
        "data/raw/Base_Migracion_2009-2026jun.xlsx"
    )


def slug(text):
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode()
    )

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text.lower(),
    ).strip("_")


def load_series(root):
    path = (
        root
        / "data"
        / "raw"
        / "Base_Migracion_2009-2026jun.xlsx"
    )

    df = pd.read_excel(
        path,
        sheet_name="Datos",
    )

    df.columns = df.columns.str.strip()

    for column in ["Año", "Mes cod", "Viajero"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["Fecha"] = pd.to_datetime(
        {
            "year": df["Año"],
            "month": df["Mes cod"],
            "day": 1,
        },
        errors="coerce",
    )

    df = df.dropna(
        subset=["Fecha", "Viajero"]
    )

    df = df[
        df["Tipo de Viajero"].isin(
            ["Turista", "Excursionista"]
        )
    ]

    total = (
        df.groupby("Fecha")["Viajero"]
        .sum()
        .sort_index()
        .asfreq("MS", fill_value=0)
        .astype(float)
    )

    by_route = (
        df.pivot_table(
            index="Fecha",
            columns="Vía",
            values="Viajero",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
        .asfreq("MS", fill_value=0)
        .astype(float)
    )

    n_train = int(
        len(total) * TRAIN_RATIO
    )

    train_end = total.index[n_train - 1]
    test_start = total.index[n_train]

    series = {
        "Vía Aérea": (
            by_route["Aérea"]
            .reindex(
                total.index,
                fill_value=0,
            )
        ),
        "Vía Terrestre": (
            by_route["Terrestre"]
            .reindex(
                total.index,
                fill_value=0,
            )
        ),
    }

    return series, train_end, test_start


def make_sequences(values, lookback):
    x = []
    y = []

    for index in range(
        len(values) - lookback
    ):
        x.append(
            values[
                index:index + lookback
            ]
        )

        y.append(
            values[index + lookback]
        )

    x = np.asarray(
        x,
        dtype=np.float32,
    ).reshape(
        -1,
        lookback,
        1,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    ).reshape(
        -1,
        1,
    )

    return x, y


def train_model(
    values_scaled,
    config,
    seed,
):
    set_seed(seed)

    x, y = make_sequences(
        values_scaled,
        config.lookback,
    )

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)

    model = LSTMModel(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    loss_fn = nn.MSELoss()

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        optimizer.zero_grad()

        prediction = model(x_tensor)

        loss = loss_fn(
            prediction,
            y_tensor,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    loss.item()
                ),
            }
        )

    return (
        model,
        pd.DataFrame(history),
    )


def walk_forward(
    model,
    train_scaled,
    future_actual_scaled,
    lookback,
):
    history = list(
        np.asarray(
            train_scaled,
            dtype=np.float32,
        )
    )

    predictions = []

    model.eval()

    with torch.no_grad():
        for actual in np.asarray(
            future_actual_scaled,
            dtype=np.float32,
        ):
            window = np.asarray(
                history[-lookback:],
                dtype=np.float32,
            )

            x = torch.from_numpy(
                window.reshape(
                    1,
                    lookback,
                    1,
                )
            )

            prediction = float(
                model(x).item()
            )

            predictions.append(
                prediction
            )

            history.append(
                float(actual)
            )

    return np.asarray(predictions)


def invert(
    predictions_scaled,
    scaler,
):
    log_predictions = (
        scaler.inverse_transform(
            predictions_scaled.reshape(
                -1,
                1,
            )
        )
        .ravel()
    )

    return np.maximum(
        np.expm1(log_predictions),
        0,
    )


def metrics(real, prediction):
    real = np.asarray(
        real,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    denominator = (
        np.abs(real)
        + np.abs(prediction)
    )

    smape = 100 * np.mean(
        np.where(
            denominator == 0,
            0,
            (
                2
                * np.abs(
                    prediction - real
                )
                / denominator
            ),
        )
    )

    return {
        "MAE": mean_absolute_error(
            real,
            prediction,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                real,
                prediction,
            )
        ),
        "sMAPE": smape,
    }


def simple_configs():
    return [
        Config(
            "LSTM simple",
            lookback,
            hidden,
            1,
            0.0,
        )
        for lookback, hidden
        in itertools.product(
            [12, 24],
            [24, 48],
        )
    ]


def stacked_configs():
    return [
        Config(
            "LSTM apilada",
            lookback,
            48,
            2,
            dropout,
        )
        for lookback, dropout
        in itertools.product(
            [12, 24],
            [0.10, 0.25],
        )
    ]


def tune(
    series_name,
    train,
    configs,
):
    tune_train = train.iloc[
        :-VALIDATION_MONTHS
    ]

    validation = train.iloc[
        -VALIDATION_MONTHS:
    ]

    rows = []

    for config_id, config in enumerate(
        configs,
        start=1,
    ):
        scaler = StandardScaler()

        train_scaled = (
            scaler.fit_transform(
                np.log1p(
                    tune_train.values
                ).reshape(-1, 1)
            )
            .ravel()
        )

        validation_scaled = (
            scaler.transform(
                np.log1p(
                    validation.values
                ).reshape(-1, 1)
            )
            .ravel()
        )

        model, history = train_model(
            train_scaled,
            config,
            SEED + config_id,
        )

        prediction_scaled = walk_forward(
            model,
            train_scaled,
            validation_scaled,
            config.lookback,
        )

        prediction = invert(
            prediction_scaled,
            scaler,
        )

        score = metrics(
            validation.values,
            prediction,
        )

        rows.append(
            {
                "Serie": series_name,
                "Modelo": config.model,
                "Configuración": config_id,
                "Lookback": config.lookback,
                "Horizonte de salida": 1,
                "Unidades ocultas": (
                    config.hidden
                ),
                "Capas LSTM": (
                    config.layers
                ),
                "Dropout": (
                    config.dropout
                ),
                "Learning rate": (
                    config.learning_rate
                ),
                "Épocas ejecutadas": int(
                    history["epoch"].max()
                ),
                "MAE validación": (
                    score["MAE"]
                ),
                "RMSE validación": (
                    score["RMSE"]
                ),
                "sMAPE validación": (
                    score["sMAPE"]
                ),
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "RMSE validación",
                "MAE validación",
            ]
        )
        .reset_index(drop=True)
    )

    best_id = int(
        results.iloc[0][
            "Configuración"
        ]
    )

    best_config = configs[
        best_id - 1
    ]

    return best_config, results


def fit_and_predict(
    train,
    test,
    config,
    seed,
):
    scaler = StandardScaler()

    train_scaled = (
        scaler.fit_transform(
            np.log1p(
                train.values
            ).reshape(-1, 1)
        )
        .ravel()
    )

    test_scaled = (
        scaler.transform(
            np.log1p(
                test.values
            ).reshape(-1, 1)
        )
        .ravel()
    )

    model, history = train_model(
        train_scaled,
        config,
        seed,
    )

    prediction_scaled = walk_forward(
        model,
        train_scaled,
        test_scaled,
        config.lookback,
    )

    prediction = invert(
        prediction_scaled,
        scaler,
    )

    return prediction, history


def plot_tuning(
    series_name,
    tuning,
    path,
):
    data = tuning.copy()

    data["Etiqueta"] = (
        data["Modelo"]
        + " C"
        + data[
            "Configuración"
        ].astype(str)
    )

    data = data.sort_values(
        "RMSE validación"
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.barh(
        data["Etiqueta"],
        data["RMSE validación"],
    )

    plt.title(
        "Tuneo de hiperparámetros: "
        f"{series_name}"
    )

    plt.xlabel(
        "RMSE de validación"
    )

    plt.ylabel(
        "Configuración"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def plot_predictions(
    series_name,
    train,
    test,
    predictions,
    path,
):
    plt.figure(
        figsize=(13, 6)
    )

    visible_train = train.iloc[-48:]

    plt.plot(
        visible_train.index,
        visible_train.values,
        label=(
            "Entrenamiento "
            "(últimos 48 meses)"
        ),
    )

    plt.plot(
        test.index,
        test.values,
        label="Valor real",
    )

    for model_name, values in (
        predictions.items()
    ):
        plt.plot(
            test.index,
            values,
            label=model_name,
        )

    plt.axvline(
        test.index.min(),
        linestyle="--",
        label="Inicio de prueba",
    )

    plt.title(
        "Predicciones LSTM para "
        f"{series_name}"
    )

    plt.xlabel("Fecha")
    plt.ylabel("Viajeros")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def plot_losses(
    series_name,
    histories,
    path,
):
    plt.figure(
        figsize=(12, 6)
    )

    for model_name, history in (
        histories.items()
    ):
        plt.plot(
            history["epoch"],
            history["train_loss"],
            label=model_name,
        )

    plt.title(
        "Pérdida de entrenamiento "
        "de los modelos finales: "
        f"{series_name}"
    )

    plt.xlabel("Época")
    plt.ylabel("MSE estandarizado")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def table_image(
    dataframe,
    title,
    path,
    decimals,
):
    table = dataframe.copy()

    for column in decimals:
        table[column] = table[
            column
        ].map(
            lambda value: (
                f"{float(value):,.2f}"
            )
        )

    width = max(
        10,
        1.55 * len(table.columns),
    )

    height = max(
        2.5,
        0.48 * len(table) + 1.6,
    )

    fig, ax = plt.subplots(
        figsize=(width, height)
    )

    ax.axis("off")

    ax.set_title(
        title,
        pad=16,
        fontsize=14,
    )

    artist = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="center",
        loc="center",
    )

    artist.auto_set_font_size(False)
    artist.set_fontsize(9)
    artist.scale(1, 1.4)

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    set_seed(SEED)

    root = project_root()

    output = (
        root
        / "data"
        / "processed"
        / "lab2"
    )

    figures = (
        output
        / "figures"
    )

    tables = (
        output
        / "tables"
    )

    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        series_map,
        train_end,
        test_start,
    ) = load_series(root)

    all_tuning = []
    metric_rows = []
    parameter_rows = []
    prediction_rows = []

    for (
        series_index,
        (series_name, series),
    ) in enumerate(
        series_map.items(),
        start=1,
    ):
        train = series.loc[
            :train_end
        ]

        test = series.loc[
            test_start:
        ]

        file_name = slug(
            series_name
        )

        print("=" * 80)
        print(series_name)

        print(
            "Train: "
            f"{train.index.min():%Y-%m} "
            "a "
            f"{train.index.max():%Y-%m}"
        )

        print(
            "Test:  "
            f"{test.index.min():%Y-%m} "
            "a "
            f"{test.index.max():%Y-%m}"
        )

        (
            best_simple,
            tuning_simple,
        ) = tune(
            series_name,
            train,
            simple_configs(),
        )

        (
            best_stacked,
            tuning_stacked,
        ) = tune(
            series_name,
            train,
            stacked_configs(),
        )

        tuning = pd.concat(
            [
                tuning_simple,
                tuning_stacked,
            ],
            ignore_index=True,
        )

        tuning["Seleccionada"] = False

        predictions = {}
        histories = {}

        for (
            model_index,
            config,
        ) in enumerate(
            [
                best_simple,
                best_stacked,
            ],
            start=1,
        ):
            mask = (
                (
                    tuning["Modelo"]
                    == config.model
                )
                & (
                    tuning["Lookback"]
                    == config.lookback
                )
                & (
                    tuning[
                        "Unidades ocultas"
                    ]
                    == config.hidden
                )
                & (
                    tuning["Dropout"]
                    == config.dropout
                )
            )

            tuning.loc[
                mask,
                "Seleccionada",
            ] = True

            (
                prediction,
                history,
            ) = fit_and_predict(
                train,
                test,
                config,
                (
                    SEED
                    + series_index * 100
                    + model_index
                ),
            )

            score = metrics(
                test.values,
                prediction,
            )

            predictions[
                config.model
            ] = prediction

            histories[
                config.model
            ] = history

            selected = tuning[
                (
                    tuning["Modelo"]
                    == config.model
                )
                & tuning[
                    "Seleccionada"
                ]
            ].iloc[0]

            metric_rows.append(
                {
                    "Serie": series_name,
                    "Modelo": config.model,
                    "MAE": score["MAE"],
                    "RMSE": score["RMSE"],
                    "sMAPE (%)": (
                        score["sMAPE"]
                    ),
                    "RMSE validación": (
                        selected[
                            "RMSE validación"
                        ]
                    ),
                }
            )

            parameter_rows.append(
                {
                    "Serie": series_name,
                    "Modelo": config.model,
                    "Lookback": (
                        config.lookback
                    ),
                    "Horizonte": 1,
                    "Unidades": (
                        config.hidden
                    ),
                    "Capas": (
                        config.layers
                    ),
                    "Dropout": (
                        config.dropout
                    ),
                    "Learning rate": (
                        config.learning_rate
                    ),
                    "Épocas finales": (
                        EPOCHS
                    ),
                }
            )

        all_tuning.append(tuning)

        prediction_rows.append(
            pd.DataFrame(
                {
                    "Fecha": test.index,
                    "Serie": series_name,
                    "Real": test.values,
                    **predictions,
                }
            )
        )

        tuning.to_csv(
            (
                tables
                / f"tuneo_{file_name}.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        plot_tuning(
            series_name,
            tuning,
            (
                figures
                / f"tuneo_{file_name}.png"
            ),
        )

        plot_predictions(
            series_name,
            train,
            test,
            predictions,
            (
                figures
                / (
                    "prediccion_"
                    f"{file_name}.png"
                )
            ),
        )

        plot_losses(
            series_name,
            histories,
            (
                figures
                / f"perdida_{file_name}.png"
            ),
        )

    tuning_results = pd.concat(
        all_tuning,
        ignore_index=True,
    )

    metric_results = (
        pd.DataFrame(metric_rows)
        .sort_values(
            ["Serie", "RMSE"]
        )
    )

    parameter_results = (
        pd.DataFrame(parameter_rows)
        .sort_values(
            ["Serie", "Modelo"]
        )
    )

    prediction_results = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    tuning_results.to_csv(
        (
            tables
            / "lstm_tuning_results.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    metric_results.to_csv(
        (
            tables
            / "lstm_test_metrics.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    parameter_results.to_csv(
        (
            tables
            / "lstm_best_parameters.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    prediction_results.to_csv(
        (
            tables
            / "lstm_predictions.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    table_image(
        metric_results,
        (
            "Métricas de los modelos "
            "LSTM en el conjunto "
            "de prueba"
        ),
        (
            figures
            / "resumen_metricas_lstm.png"
        ),
        {
            "MAE",
            "RMSE",
            "sMAPE (%)",
            "RMSE validación",
        },
    )

    table_image(
        parameter_results,
        (
            "Mejores hiperparámetros "
            "seleccionados"
        ),
        (
            figures
            / "mejores_parametros_lstm.png"
        ),
        {
            "Dropout",
            "Learning rate",
        },
    )

    print(
        "\nMétricas finales"
    )

    print(
        metric_results.to_string(
            index=False
        )
    )

    print(
        "\nResultados guardados en "
        f"{output}"
    )


if __name__ == "__main__":
    main()