import streamlit as st
import plotly.express as px

from src.dss.profile_viewer import (
    build_profile_table,
    build_crosstab
)

st.title("Profile Viewer")

try:

    profile_df = build_profile_table()

    demand_type = st.selectbox(
        "Demand Type",
        [
            "All",
            "regular",
            "intermittent"
        ]
    )

    if demand_type != "All":

        filtered_df = profile_df[
            profile_df["demand_type"]
            == demand_type
        ]

    else:

        filtered_df = profile_df

    st.subheader(
        "Series Profiles"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    st.subheader(
        "Demand Type × Best Model"
    )

    crosstab = build_crosstab(
        filtered_df
    )

    st.dataframe(
        crosstab,
        use_container_width=True
    )

    chart_df = (
        crosstab
        .reset_index()
        .melt(
            id_vars="demand_type",
            var_name="best_model",
            value_name="count"
        )
    )

    fig = px.bar(
        chart_df,
        x="demand_type",
        y="count",
        color="best_model",
        barmode="group",
        title="Best Models by Demand Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except FileNotFoundError:

    st.error(
        "analysis_profile.csv hoặc metrics file không tồn tại."
    )

except Exception as e:

    st.error(str(e))