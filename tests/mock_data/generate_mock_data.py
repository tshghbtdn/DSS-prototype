import random
import numpy as np
import pandas as pd
from pathlib import Path

MODELS = ["naive", "snaive", "sarimax", "prophet", "lstm"]
PHASES = [1, 2]
FOLDS = [1, 2, 3] #để giá trị tối thiểu 1-3 (theo yêu cầu) 

SERIES = [
    ("FOODS_1_001", "CA_1", "regular"),
    ("FOODS_1_002", "CA_1", "regular"),
    ("FOODS_1_003", "TX_1", "regular"),
    ("FOODS_1_004", "TX_2", "regular"),
    ("HOUSEHOLD_1_001", "WI_1", "regular"),
    ("HOUSEHOLD_1_002", "WI_2", "regular"),
    ("HOBBIES_1_001", "CA_2", "regular"),
    ("HOBBIES_1_002", "TX_1", "intermittent"),
    ("HOBBIES_1_003", "TX_2", "intermittent"),
    ("HOUSEHOLD_1_003", "WI_3", "intermittent"),
]

MODEL_ERROR = {
    "naive": 5.0,
    "snaive": 4.0,
    "prophet": 3.0,
    "sarimax": 2.0,
    "lstm": 1.5
}

FORECAST_DAYS = 30

TRAIN_TIME_CONFIG = {
    "naive": 0.01,
    "snaive": 0.02,
    "sarimax": (5, 60),
    "prophet": (1, 20),
    "lstm": (100, 500)
}


def generate_analysis_profile():
    rows = []

    for item_id, store_id, demand_type in SERIES:

        mean_sales = round(random.uniform(0.1, 30), 2)
        std_sales = round(random.uniform(0.1, 20), 2)

        if random.random() < 0.1:
            cv = np.nan
        else:
            cv = round(random.uniform(0.2, 5.0), 2)

        rows.append({
            "item_id": item_id,
            "store_id": store_id,
            "mean_sales": mean_sales,
            "std_sales": std_sales,
            "cv": cv,
            "zero_ratio": round(random.uniform(0.0, 0.9), 2),
            "missing_ratio": round(random.uniform(0.0, 0.1), 2),
            "trend_slope": round(random.uniform(-0.01, 0.02), 4),
            "total_days": random.randint(1935, 1945),
            "demand_type": demand_type,
            "profile_scope": "full"
        })

    df = pd.DataFrame(rows)

    output_path = Path("data/processed/analysis_profile.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"analysis_profile.csv generated ({len(df)} rows)")

def generate_y_true(demand_type, mean_sales):

    if demand_type == "regular":

        value = np.random.normal(
            loc=mean_sales,
            scale=max(1, mean_sales * 0.2)
        )

        return round(max(0, value), 2)

    else:

        # 70% ngày không bán được
        if random.random() < 0.7:
            return 0.0

        value = np.random.uniform(
            1,
            max(2, mean_sales)
        )

        return round(value, 2)

def generate_predictions(model, phase, profile_df): #tạo cho 1 model,1phase

    rows = []

    dates = pd.date_range(
        start="2016-03-01",
        periods=FORECAST_DAYS
    )

    for fold in FOLDS:

        for _, profile in profile_df.iterrows():

            item_id = profile["item_id"]
            store_id = profile["store_id"]
            mean_sales = profile["mean_sales"]
            demand_type = profile["demand_type"]

            for date in dates:

                y_true = generate_y_true(
                    demand_type,
                    mean_sales
                )

                error = np.random.normal(
                    0,
                    MODEL_ERROR[model]
                )

                y_pred = round(
                    max(0, y_true + error),
                    2
                )

                fallback = False

                if model == "sarimax":
                    fallback = random.random() < 0.1

                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "item_id": item_id,
                    "store_id": store_id,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "fold": fold,
                    "fallback": fallback
                })

    df = pd.DataFrame(rows)

    output_path = Path(
        f"results/{model}/phase{phase}/predictions.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output_path, index=False)

    print(
        f"{model} phase{phase}: "
        f"{len(df)} rows generated"
    )

def generate_all_predictions():

    profile_df = pd.read_csv(
        "data/processed/analysis_profile.csv"
    )

    for model in MODELS:
        for phase in PHASES:

            generate_predictions(
                model,
                phase,
                profile_df
            )

def generate_metrics_comparison():

    rows = []

    model_base = { #giả định để dashboard trông hợp lý hơn
        "naive": 2.20,
        "snaive": 1.60,
        "prophet": 1.10,
        "sarimax": 0.90,
        "lstm": 0.80
    }

    for model in MODELS:
        for phase in PHASES:
            for item_id, store_id, _ in SERIES:
                for fold in FOLDS:

                    mase = round(
                        max(
                            0.3,
                            min(
                                3.0,
                                model_base[model]
                                + random.uniform(-0.15, 0.15)
                            )
                        ),
                        3
                    )

                    mae = round(
                        random.uniform(0.5, 15),
                        2
                    )

                    rmse = round(
                        random.uniform(
                            max(1.0, mae),
                            20.0
                        ),
                        2
                    )

                    if random.random() < 0.1:
                        mape = np.nan
                    else:
                        mape = round(
                            random.uniform(10, 200),
                            2
                        )

                    rows.append({
                        "model": model,
                        "phase": phase,
                        "item_id": item_id,
                        "store_id": store_id,
                        "fold": fold,
                        "mae": mae,
                        "rmse": rmse,
                        "mase": mase,
                        "mape": mape
                    })

    df = pd.DataFrame(rows)

    output_path = Path(
        "results/metrics_comparison.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"metrics_comparison.csv generated "
        f"({len(df)} rows)"
    )

def generate_training_logs():

    for model in MODELS:

        for phase in PHASES:

            rows = []

            for item_id, store_id, _ in SERIES:

                for fold in FOLDS:
                    # train_time_s
                    if isinstance(TRAIN_TIME_CONFIG[model], tuple):
                        train_time = round(
                            random.uniform(*TRAIN_TIME_CONFIG[model]),
                            2
                        )
                    else:
                        train_time = TRAIN_TIME_CONFIG[model]
                    # peak_ram_mb (50 - 2000)
                    peak_ram = round(
                        random.uniform(50, 2000),
                        2
                    )
                    # cpu_percent (10 - 100)
                    cpu_percent = round(
                        random.uniform(10, 100),
                        2
                    )

                    # converged / error_msg
                    converged = True
                    error_msg = ""

                    # một số ít dòng lỗi timeout
                    if random.random() < 0.05:
                        converged = False
                        error_msg = "timeout"

                    rows.append({
                        "fold": fold,
                        "item_id": item_id,
                        "store_id": store_id,
                        "train_time_s": train_time,
                        "peak_ram_mb": peak_ram,
                        "cpu_percent": cpu_percent,
                        "converged": converged,
                        "error_msg": error_msg
                    })

            df = pd.DataFrame(rows)

            output_path = Path(
                f"results/{model}/phase{phase}/training_log.csv"
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            df.to_csv(
                output_path,
                index=False
            )

            print(
                f"{model} phase{phase} training_log generated "
                f"({len(df)} rows)"
            )

if __name__ == "__main__":
    generate_analysis_profile()
    generate_all_predictions()
    generate_metrics_comparison()
    generate_training_logs()

