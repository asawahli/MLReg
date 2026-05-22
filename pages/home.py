import streamlit as st
from src.utils import load_csv


st.set_page_config(page_title="MLReg", layout="wide")

st.html(
    """
    <div style="padding:18px;border-radius:8px; border:1px solid black">
        <h1 style="color:#0099FF; margin-bottom:5px"> 
            <span style="font-size:60px"> MLReg </span>
        </h1>
        <h3 style="color:#0099FF; margin-top:0px">
            <span style="font-size:20px">Interactive Machine Learning Regression App</span>
        </h3>

        <p style="font-size:15px;">  
            Use this app to:
        </p>
      <ul >
        <li>Upload and explore your dataset</li>
        <li>Select features and target variables.</li>
        <li>Apply preprocessing and train models.</li>
        <li>Save trained models for the current session.</li>
        <li>Upload a models and make predictions.</li>
      </ul>
    </div>
    """,
    # unsafe_allow_html=True,
)


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
