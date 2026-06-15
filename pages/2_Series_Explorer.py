import streamlit as st
import pandas as pd

from src.dss.comparator import (
    create_forecast_plot,
    get_metrics_table
)

st.title("Series Explorer")

try:

    sample = pd.read_csv(
        "results/naive/phase1/predictions.csv"
    )

    item_ids = sorted(
        sample["item_id"].unique()
    )

    item_id = st.selectbox(
        "Item ID",
        item_ids
    )

    stores = sorted(
        sample[
            sample["item_id"] == item_id
        ]["store_id"].unique()
    )

    store_id = st.selectbox(
        "Store ID",
        stores
    )

    fold = st.selectbox(
        "Fold",
        sorted(sample["fold"].unique())
    )

    phase = st.radio(
        "Phase",
        [1, 2],
        horizontal=True
    )

    st.subheader(
        "Forecast Comparison"
    )

    fig = create_forecast_plot(
        item_id=item_id,
        store_id=store_id,
        fold=fold,
        phase=phase
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Metrics Table"
    )

    metrics = get_metrics_table(
        item_id=item_id,
        store_id=store_id,
        fold=fold,
        phase=phase
    )

    st.dataframe(
        metrics,
        use_container_width=True
    )

except FileNotFoundError:

    st.error(
        "Prediction files not found."
    )

except Exception as e:

    st.error(str(e))