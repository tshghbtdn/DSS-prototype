import streamlit as st
import pandas as pd
import plotly.express as px

from src.dss.overview import get_overview_metrics


st.set_page_config(
    page_title="Overview",
    layout="wide"
)

st.title("Overview Dashboard")

try:

    metrics_df = pd.read_csv(
        "results/metrics_comparison.csv"
    )

    phase = st.radio(
        "Select Phase",
        [1, 2],
        horizontal=True
    )

    summary = get_overview_metrics(
        metrics_df,
        phase
    )

    best_row = summary.loc[
        summary["mase"].idxmin()
    ]

    st.subheader(
        f"Phase {phase} Performance Summary"
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.success(
        f"🏆 Best Model: "
        f"{best_row['model'].upper()} "
        f"(Mean MASE = {best_row['mase']:.3f})"
    )

    fig = px.bar(
        summary,
        x="model",
        y="mase",
        title="Mean MASE Comparison",
        text_auto=".3f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except FileNotFoundError:

    st.error(
        "metrics_comparison.csv not found."
    )

except Exception as e:

    st.error(
        f"Unexpected error: {e}"
    )