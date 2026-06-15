import streamlit as st

st.set_page_config(
    page_title="Forecast Model DSS",
    page_icon="📈",
    layout="wide"
)

st.title("Forecast Model Decision Support System")

st.markdown("""
Welcome to the DSS prototype for forecasting model selection.

Use the navigation menu on the left to explore:

- Overview
- Series Explorer
- Model Recommender
- Trade-off Dashboard
- Profile Viewer
""")