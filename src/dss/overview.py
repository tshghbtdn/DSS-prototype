import pandas as pd


def get_overview_metrics(
    metrics_df: pd.DataFrame,
    phase: int
) -> pd.DataFrame:

    phase_df = metrics_df[
        metrics_df["phase"] == phase
    ]

    summary = (
        phase_df
        .groupby("model", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mase=("mase", "mean"),
            mape=("mape", "mean")
        )
    )

    return summary.round(3)