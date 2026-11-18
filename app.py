"""Streamlit dashboard for the congestion prediction prototype (synthetic data).

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from predict_congestion import FEATURES, load_data, train_model  # noqa: E402

st.set_page_config(page_title="Port Congestion Predictor", layout="wide")
st.title("Intelligent Port Logistics AI")
st.caption("Prototype trained on synthetic data. Not for real operational decisions.")


@st.cache_resource
def get_model():
    df = load_data()
    model, _ = train_model(df)
    return df, model


try:
    df, model = get_model()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

left, right = st.columns(2)

with left:
    st.subheader("Predict congestion for an hour")
    hour = st.slider("Hour of day", 0, 23, 9)
    day = st.selectbox(
        "Day of week",
        options=list(range(7)),
        format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
    )
    trucks = st.slider("Scheduled trucks", 0, 60, 30)
    vessels = st.slider("Vessels in port", 0, 5, 2)
    customs = st.slider("Average customs delay (min)", 5, 90, 30)

    row = [[hour, day, trucks, vessels, customs]]
    prob = model.predict_proba(row)[0][1]
    st.metric("Estimated congestion risk", f"{prob:.0%}")
    if prob >= 0.6:
        st.warning("High risk: consider moving some truck slots to quieter hours.")
    else:
        st.success("Lower risk for these conditions.")

with right:
    st.subheader("Average queue length by hour (synthetic data)")
    st.bar_chart(df.groupby("hour")["queue_length"].mean())
