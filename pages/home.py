import streamlit as st
from src.utils import load_csv
import pathlib

st.set_page_config(page_title="MLReg", layout="wide")


# logo
with st.columns([1, 1.5, 1])[1]:
    st.image("assets/logo.svg", width="stretch")

# title
# st.markdown(
#     """
#     <div style="text-align: center;">
#         <h2 style="margin-bottom: 5px; font-family: "Montserrat">
#         Interactive Machine Learning Regression App
#         </h2>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )
st.html("assets/home.html", unsafe_allow_javascript=True)


st.markdown("### Start here")


# --- File upload ---
with st.expander("Upload CSV file", expanded=True):
    uploaded = st.file_uploader(
        "Upload a CSV file (first row should be headers)",
        type=["csv"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        df = load_csv(uploaded)
        if df is not None:
            st.session_state.df = df
            st.success(
                f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns."
            )

if st.session_state.df is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()


# st.page_link(
#     "pages/1_train.py",
#     label="Go to Train page",
# )
# st.page_link(
#     "pages/2_predict.py",
#     label="Go to Predict page",
# )
