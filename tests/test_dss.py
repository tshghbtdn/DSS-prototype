import pandas as pd
from pathlib import Path
from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)
from src.dss.model_selector import select_best_model
""""
#test model_selector.py
metrics_df = pd.read_csv(
    "results/metrics_comparison.csv"
)

best_models = select_best_model(
    metrics_df,
    phase=1
)
print(best_models.head())
print(best_models.shape)
"""

#test comparator.py
import pandas as pd

from src.dss.comparator import (
    create_forecast_plot,
    get_metrics_table
)

fig = create_forecast_plot(
    item_id="FOODS_1_001",
    store_id="CA_1",
    fold=1,
    phase=1
)

fig.show()

metrics = get_metrics_table(
    item_id="FOODS_1_001",
    store_id="CA_1",
    fold=1,
    phase=1
)

print(metrics)
"""
#test format
MODELS = ["naive", "snaive", "sarimax", "prophet", "lstm"]


def test_files_exist():

    assert Path(
        "data/processed/analysis_profile.csv"
    ).exists()

    assert Path(
        "results/metrics_comparison.csv"
    ).exists()

    for model in MODELS:
        for phase in [1, 2]:

            assert Path(
                f"results/{model}/phase{phase}/predictions.csv"
            ).exists()

            assert Path(
                f"results/{model}/phase{phase}/training_log.csv"
            ).exists()


def test_analysis_profile():

    df = pd.read_csv(
        "data/processed/analysis_profile.csv"
    )

    expected_columns = [
        "item_id",
        "store_id",
        "mean_sales",
        "std_sales",
        "cv",
        "zero_ratio",
        "missing_ratio",
        "trend_slope",
        "total_days",
        "demand_type",
        "profile_scope"
    ]

    assert list(df.columns) == expected_columns

    assert df["mean_sales"].between(
        0.1, 30
    ).all()

    assert df["std_sales"].between(
        0.1, 20
    ).all()

    assert df["zero_ratio"].between(
        0, 0.9
    ).all()

    assert df["missing_ratio"].between(
        0, 0.1
    ).all()

    assert df["trend_slope"].between(
        -0.01, 0.02
    ).all()

    assert set(
        df["demand_type"].unique()
    ).issubset(
        {"regular", "intermittent"}
    )

    assert (
        df["profile_scope"] == "full"
    ).all()


def test_metrics_comparison():

    df = pd.read_csv(
        "results/metrics_comparison.csv"
    )

    expected_columns = [
        "model",
        "phase",
        "item_id",
        "store_id",
        "fold",
        "mae",
        "rmse",
        "mase",
        "mape"
    ]

    assert list(df.columns) == expected_columns

    assert set(
        df["model"].unique()
    ) == {
        "naive",
        "snaive",
        "sarimax",
        "prophet",
        "lstm"
    }

    assert df["phase"].isin(
        [1, 2]
    ).all()

    assert df["mae"].between(
        0.5, 15
    ).all()

    assert df["rmse"].between(
        1, 20
    ).all()

    assert df["mase"].between(
        0.3, 3.0
    ).all()

    mape_non_nan = df["mape"].dropna()

    assert mape_non_nan.between(
        10, 200
    ).all()


def test_predictions():

    df = pd.read_csv(
        "results/lstm/phase1/predictions.csv"
    )

    expected_columns = [
        "date",
        "item_id",
        "store_id",
        "y_true",
        "y_pred",
        "fold",
        "fallback"
    ]

    assert list(df.columns) == expected_columns

    assert df["fold"].isin(
        [1, 2, 3]
    ).all()

    assert (df["y_true"] >= 0).all()

    assert (df["y_pred"] >= 0).all()


def test_sarimax_fallback():

    df = pd.read_csv(
        "results/sarimax/phase1/predictions.csv"
    )

    assert (
        df["fallback"] == True
    ).any()


def test_training_log():

    df = pd.read_csv(
        "results/lstm/phase1/training_log.csv"
    )

    expected_columns = [
        "fold",
        "item_id",
        "store_id",
        "train_time_s",
        "peak_ram_mb",
        "cpu_percent",
        "converged",
        "error_msg"
    ]

    assert list(df.columns) == expected_columns

    assert df["peak_ram_mb"].between(
        50, 2000
    ).all()

    assert df["cpu_percent"].between(
        10, 100
    ).all()

    assert (
        df["train_time_s"] >= 0
    ).all()

if __name__ == "__main__":

    test_files_exist()
    print("✓ test_files_exist")

    test_analysis_profile()
    print("✓ test_analysis_profile")

    test_metrics_comparison()
    print("✓ test_metrics_comparison")

    test_predictions()
    print("✓ test_predictions")

    test_sarimax_fallback()
    print("✓ test_sarimax_fallback")

    test_training_log()
    print("✓ test_training_log")

    print("\nAll tests passed!")
    """