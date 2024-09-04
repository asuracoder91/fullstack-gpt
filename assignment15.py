import streamlit as st
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")

st.title(today)

st.selectbox(
    "Choose your model",
    (
        "GPT-3",
        "GPT-4",
        "GPT-4o",
    ),
)
