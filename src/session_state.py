import streamlit as st
from src.ai_setting import get_logfare_models


def init_session_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "models" not in st.session_state:
        st.session_state.models = {}  # key -> {model, metrics, params, timestamp}
    if "split" not in st.session_state:
        st.session_state.split = {}
    if "last_train" not in st.session_state:
        st.session_state.last_train = None
    if "current_view" not in st.session_state:
        st.session_state.current_view = None
    if "model_train_state" not in st.session_state:
        st.session_state.model_train_state = None
    if "ai_summary" not in st.session_state:
        st.session_state.ai_summary = None
    if "ai_init" not in st.session_state:
        st.session_state.ai_init = {
            "logfare_models": get_logfare_models(),
            "google_models": [],
        }
    # if "api_google" not in st.session_state:
    #     st.session_state.api_google = None
    if "saved_api_google" not in st.session_state:
        st.session_state.saved_api_google = ""
    # if "api_logfare" not in st.session_state:
    #     st.session_state.api_logfare = None
    if "saved_api_logfare" not in st.session_state:
        st.session_state.saved_api_logfare = ""
    if "ai_settings" not in st.session_state:
        st.session_state.ai_settings = None
