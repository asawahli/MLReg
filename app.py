import streamlit as st
from src.session_state import init_session_state

# hide_st_style = """
# <style>
# .stToolbarActionButton {visibility: hidden;}
# """

# st.markdown(hide_st_style, unsafe_allow_html=True)

init_session_state()

st.set_page_config(page_title="MLreg", layout="wide", page_icon="assets/icon.svg")
with st.sidebar:
    st.write("")

pages = [
    st.Page("pages/home.py", title="Home"),
    st.Page("pages/eda.py", title="Explore Data"),
    st.Page("pages/preprocessing.py", title="Preprocessing"),
    st.Page("pages/train_model.py", title="Train Models"),
    st.Page("pages/models_summary.py", title="Model Summary"),
    st.Page("pages/predict.py", title="Predict"),
    st.Page("pages/setting.py", title="Setting"),
    st.Page("pages/guidelines.py", title="Guide"),
]


pg = st.navigation(pages, position="top")
pg.run()


footer = """
<style>
footer {visibility: hidden;}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #888888;
    text-align: center;
    font-size: 14px;
}
</style>
<div class="footer">
    <p>Developed by Abdulrahman | <a href="mailto:your-email@example.com" style="color: #888888;">Contact</a></p>
</div>
"""

# st.markdown(footer, unsafe_allow_html=True)
