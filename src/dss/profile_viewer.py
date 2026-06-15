import pandas as pd

from src.dss.model_selector import (
    select_best_model
)


def build_profile_table():

    profile_df = pd.read_csv(
        "data/processed/analysis_profile.csv"
    )

    metrics_df = pd.read_csv(
        "results/metrics_comparison.csv"
    )

    best_models = select_best_model(
        metrics_df,
        phase=1
    )

    profile_df = profile_df.merge(
        best_models[
            [
                "item_id",
                "store_id",
                "best_model"
            ]
        ],
        on=["item_id", "store_id"],
        how="left"
    )

    return profile_df


def build_crosstab(
    profile_df: pd.DataFrame
):

    return pd.crosstab(
        profile_df["demand_type"],
        profile_df["best_model"]
    )