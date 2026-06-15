import pandas as pd



def select_best_model(
    metrics_df: pd.DataFrame,
    phase: int = 1
) -> pd.DataFrame:

    phase_df = metrics_df[
        metrics_df["phase"] == phase
    ]

    summary = (
        phase_df
        .groupby(
            ["item_id", "store_id", "model"],
            as_index=False
        )
        .agg(
            mean_mase=("mase", "mean"),
            mean_mae=("mae", "mean")
        )
    )

    results = []

    for (item_id, store_id), group in summary.groupby(
        ["item_id", "store_id"]
    ):

        min_mase = group["mean_mase"].min()

        candidates = group[
            (
                group["mean_mase"]
                - min_mase
            ).abs() < 0.01
        ]

        best = candidates.loc[
            candidates["mean_mae"].idxmin()
        ]

        results.append({
            "item_id": item_id,
            "store_id": store_id,
            "best_model": best["model"],
            "mean_mase": round(best["mean_mase"], 3),
            "mean_mae": round(best["mean_mae"], 3)
        })

    return pd.DataFrame(results)