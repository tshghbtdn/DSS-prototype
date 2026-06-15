import streamlit as st

from src.dss.recommender import (
    ModelRecommender
)

st.title("Model Recommender")

recommender = ModelRecommender()

demand_type = st.selectbox(
    "Demand Type",
    [
        "regular",
        "intermittent"
    ]
)

cv = st.slider(
    "CV",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.1
)

mean_sales = st.slider(
    "Mean Sales",
    min_value=0.0,
    max_value=30.0,
    value=10.0,
    step=0.5
)

if st.button("Recommend"):

    result = recommender.recommend(
        demand_type=demand_type,
        cv=cv,
        mean_sales=mean_sales
    )

    st.subheader("Result")

    st.write(
        f"**Recommended Model:** "
        f"{result['model']}"
    )

    st.write(
        f"**Confidence:** "
        f"{result['confidence']}"
    )

    st.write(
        f"**Evidence:** "
        f"{result['evidence']}"
    )

    st.write(
        f"**Reasoning:** "
        f"{result['reasoning']}"
    )