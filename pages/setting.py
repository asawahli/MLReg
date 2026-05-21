import streamlit as st
import requests
from openai import OpenAI
from google import genai
from src.ai_setting import get_logfare_models, get_google_models
from src.session_state import init_session_state
import pandas as pd

init_session_state()


@st.dialog(" ")
def save_settings(provider, api, ai_model):
    ai_settings = {
        "provider": provider,
        "api": api,
        "model": ai_model,
    }
    st.session_state["ai_settings"] = ai_settings
    st.success("Settings saved!")


_, col1, _ = st.columns([0.5, 5, 0.5])
with col1.container(border=True, width="stretch"):
    st.subheader("AI Client")

    provider = st.selectbox("AI provider", ["logfare.ai", "gemini"])
    if provider == "logfare.ai":
        api = st.text_input(
            "API",
            key="api_logfare_widget",
            value=st.session_state.saved_api_logfare,
            type="password",
        )
        if api != st.session_state.saved_api_logfare:
            st.session_state.saved_api_logfare = api
        ai_model = st.selectbox(
            "Model",
            st.session_state.ai_init["logfare_models"],
        )
        st.warning(
            "* This application uses Logfare AI services to process requests. Inputs, outputs, and related data may be logged and stored by Logfare for research and operational purposes in accordance with their policies. Users should avoid submitting sensitive or confidential information.\n * Only results of analysis used in AI prompt. The original dataset not used in prompts"
        )
    elif provider == "gemini":
        api = st.text_input(
            "API",
            value=st.session_state.saved_api_google,
            key="api_google_widget",
            type="password",
        )
        if api != st.session_state.saved_api_google:
            st.session_state.saved_api_google = api
            get_google_models()
        ai_model = st.selectbox(
            "Model",
            st.session_state.ai_init["google_models"],
        )

    _, _, col = st.columns([2, 2, 1])
    col.button(
        "Save Settings",
        on_click=lambda: save_settings(provider, api, ai_model),
    )



