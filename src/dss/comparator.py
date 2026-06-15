from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


MODELS = [
    "naive",
    "snaive",
    "sarimax",
    "prophet",
    "lstm"
]


def create_forecast_plot(
    item_id: str,
    store_id: str,
    fold: int,
    phase: int = 1
):

    fig = go.Figure()

    actual_added = False

    for model in MODELS:

        path = Path(
            f"results/{model}/phase{phase}/predictions.csv"
        )

        df = pd.read_csv(path)

        series = df[
            (df["item_id"] == item_id)
            & (df["store_id"] == store_id)
            & (df["fold"] == fold)
        ].copy()

        if series.empty:
            continue

        series["date"] = pd.to_datetime(
            series["date"]
        )

        series = series.sort_values(
            "date"
        )

        if not actual_added:

            fig.add_trace(
                go.Scatter(
                    x=series["date"],
                    y=series["y_true"],
                    mode="lines",
                    name="Actual"
                )
            )

            actual_added = True

        fig.add_trace(
            go.Scatter(
                x=series["date"],
                y=series["y_pred"],
                mode="lines",
                name=model.upper()
            )
        )

    fig.update_layout(
        title=(
            f"{item_id} | "
            f"{store_id} | "
            f"Fold {fold}"
        ),
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified"
    )

    return fig


def get_metrics_table(
    item_id: str,
    store_id: str,
    fold: int,
    phase: int = 1
):

    df = pd.read_csv(
        "results/metrics_comparison.csv"
    )

    result = df[
        (df["phase"] == phase)
        & (df["item_id"] == item_id)
        & (df["store_id"] == store_id)
        & (df["fold"] == fold)
    ]

    return (
        result[
            [
                "model",
                "mae",
                "rmse",
                "mase",
                "mape"
            ]
        ]
        .sort_values("mase")
        .reset_index(drop=True)
    )