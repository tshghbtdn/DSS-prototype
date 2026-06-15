import pandas as pd


def get_tradeoff_data(
    metrics_df: pd.DataFrame,
    training_df: pd.DataFrame,
    phase: int
) -> pd.DataFrame:

    metrics_phase = metrics_df[
        metrics_df["phase"] == phase
    ]

    metrics_summary = (
        metrics_phase
        .groupby("model", as_index=False)
        .agg(
            mean_mase=("mase", "mean")
        )
    )

    training_summary = (
        training_df
        .groupby("model", as_index=False)
        .agg(
            mean_train_time_s=("train_time_s", "mean"),
            mean_peak_ram_mb=("peak_ram_mb", "mean")
        )
    )

    tradeoff_df = metrics_summary.merge(
        training_summary,
        on="model"
    )

    return tradeoff_df.round(3)


def get_pareto_frontier(
    tradeoff_df: pd.DataFrame
) -> pd.DataFrame:

    frontier_rows = []

    for i, row_a in tradeoff_df.iterrows():

        dominated = False

        for j, row_b in tradeoff_df.iterrows():

            if i == j:
                continue

            if (
                row_b["mean_mase"]
                <= row_a["mean_mase"]
                and
                row_b["mean_train_time_s"]
                <= row_a["mean_train_time_s"]
                and
                (
                    row_b["mean_mase"]
                    < row_a["mean_mase"]
                    or
                    row_b["mean_train_time_s"]
                    < row_a["mean_train_time_s"]
                )
            ):
                dominated = True
                break

        if not dominated:
            frontier_rows.append(row_a.to_dict())

    frontier_df = pd.DataFrame(
        frontier_rows
    )

    frontier_df = frontier_df.sort_values(
        by="mean_mase"
    ).reset_index(
        drop=True
    )

    return frontier_df