import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats


def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read CSV: {e}")
        return None


@st.dialog(" ")
def store_model(model_name_input):
    if st.session_state.get("last_train") is None:
        st.warning(
            "No model trained in this session to store. Train first, then store."
        )
    else:
        key = model_name_input
        # store a copy of the pipeline and metrics
        entry = {
            "type": st.session_state.last_train.get("type"),
            "pipeline": st.session_state.last_train.get("pipeline"),
            "metrics_test": st.session_state.last_train.get("metrics_test"),
            "metrics_train": st.session_state.last_train.get("metrics_train"),
            "params": st.session_state.last_train.get("params"),
            "timestamp": st.session_state.last_train.get("timestamp"),
        }
        st.session_state.models[key] = entry
        st.success(f"Stored model under key: '{key}'")
        st.session_state["ai_summary"] = None


# @st.cache_data
def describe(df: pd.DataFrame):
    stats = df.describe().T
    stats["skewness"] = df.skew()
    stats["kurtosis"] = df.kurtosis()
    return stats


def build_pipeline(model_obj, use_scaler=True):
    steps = []
    # Impute numeric with median
    steps.append(("imputer", SimpleImputer(strategy="median")))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model_obj))
    return Pipeline(steps)


def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
