import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dss.tradeoff import (
    get_tradeoff_data,
    get_pareto_frontier
)

st.title("Trade-off Dashboard")

try:

    metrics_df = pd.read_csv(
        "results/metrics_comparison.csv"
    )

    phase = st.radio(
        "Phase",
        [1, 2],
        horizontal=True
    )

    training_frames = []

    for model in [
        "naive",
        "snaive",
        "sarimax",
        "prophet",
        "lstm"
    ]:

        df = pd.read_csv(
            f"results/{model}/phase{phase}/training_log.csv"
        )

        df["model"] = model

        training_frames.append(df)

    training_df = pd.concat(
        training_frames,
        ignore_index=True
    )

    tradeoff_df = get_tradeoff_data(
        metrics_df,
        training_df,
        phase
    )

    st.subheader("Trade-off Summary")

    st.dataframe(
        tradeoff_df,
        use_container_width=True
    )

    fig = px.scatter(
        tradeoff_df,
        x="mean_mase",
        y="mean_train_time_s",
        size="mean_peak_ram_mb",
        text="model",
        hover_name="model"
    )

    frontier = get_pareto_frontier(
        tradeoff_df
    )

    fig.add_trace(
        go.Scatter(
            x=frontier["mean_mase"],
            y=frontier["mean_train_time_s"],
            mode="lines+markers",
            name="Pareto Frontier"
        )
    )

    fig.update_layout(
        title="Accuracy vs Training Time",
        xaxis_title="Mean MASE (lower is better)",
        yaxis_title="Mean Training Time (seconds)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except FileNotFoundError:

    st.error(
        "Training log hoặc metrics file không tồn tại."
    )

except Exception as e:

    st.error(str(e))