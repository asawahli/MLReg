from google import genai
from openai import OpenAI
from prompts import ai_prompts
import streamlit as st
import requests


def get_logfare_models():
    url = "https://logfare.ai/v1/models"
    url_response = requests.get(url)
    try:
        models = url_response.json()
        models_id = [model["id"] for model in models["data"]]
    except Exception:
        print(f"Error: {url_response.status_code} - {url_response.text}")
        models_id = []
    return models_id


def get_google_models():
    try:
        api = st.session_state.saved_api_google
        if not api:
            return
        gclient = genai.Client(api_key=api)
        gmodels = [
            gclient.models.list()[i].name.split("/")[-1]
            for i in range(len(gclient.models.list()))
            if "generateContent" in gclient.models.list()[i].supported_actions
        ]
        st.session_state.ai_init["google_models"] = gmodels
    except Exception:
        st.session_state.ai_init["google_models"] = []
    # return gmodels


## AI response:
def ai_support(inputs, type, figs=None):
    """
    provider: [logfare.ai, gemini]
    ai_model: ai_model
    inputs:  inputs
    type: string ['describe', 'corr', 'residuals', 'summary']
    """
    if not st.session_state.ai_settings:
        raise Exception("AI settings not configured.")

    provider, api, ai_model = st.session_state.ai_settings.values()

    if provider == "logfare.ai":
        # api = st.session_state["saved_api_logfare"]
        client = OpenAI(base_url="https://logfare.ai/v1", api_key=api)
    elif provider == "gemini":
        # api = st.session_state["saved_api_google"]
        client = OpenAI(
            api_key=api,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    content = ai_prompts(inputs=inputs, analysis_type=type, figs=figs)

    response = client.chat.completions.create(
        model=ai_model,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )
    return response.choices[0].message.content
