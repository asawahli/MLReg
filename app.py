# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from scipy import stats
import requests
from openai import OpenAI
from google import genai


from prompts import ai_prompts
# from radar import radar_factory


st.set_page_config(page_title="Interactive ML Regression App", layout="wide")


# --- Helper functions ---
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


def reset_model_train_state():
    st.session_state.model_train_state = None


# Constants
markers = {
    "Default": None,
    "Circle": "o",
    "Point": ".",
    "Pixel": ",",
    "Square": "s",
    "Pentagon": "p",
    "Star": "*",
    "Plus": "+",
    "X": "x",
    "Dimond": "D",
}

params = {
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.family": "serif",
}
plt.rcParams.update(params)
sns.set_palette(sns.color_palette())


def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read CSV: {e}")
        return None


# @st.cache_data
def describe(df: pd.DataFrame):
    stats = df.describe().T
    stats["skewness"] = df.skew()
    stats["kurtosis"] = df.kurtosis()
    return stats


def customize_plot(df, xdata, ydata, scatter_kwargs=None, ax_kwargs=None):
    fig, ax = plt.subplots()

    ax.scatter(df[xdata].values, df[ydata], **scatter_kwargs)

    if ax_kwargs:
        for k, v in ax_kwargs.items():
            setter = getattr(ax, f"set_{k}", None)
            if callable(setter):
                setter(v)

    return fig


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


def plot_predictions(y_true, y_pred, model_name, type="Test"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, linestyle="--", color="red")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.axis("square")
    ax.set_title(f"{model_name} — Predicted vs Actual ({type})")
    return fig


def plot_residuals(y_true, y_pred, model_name, type="Test"):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title(f"{model_name} — Residuals Distribution ({type})")
    return fig


def plot_residuals_vs_fitted(y_true, y_pred, model_name, type="Test"):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()

    sns.residplot(
        x=y_pred, y=residuals, lowess=True, line_kws={"color": "red", "lw": 2}, ax=ax
    )

    ax.set_xlabel("Fitted (Predicted) Values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{model_name} — Residuals vs Fitted ({type})")
    return fig


def plot_qq(y_true, y_pred, model_name, type="Test"):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    (theoretical_q, ordered_res), (slope, intercept, r) = stats.probplot(
        residuals, dist="norm"
    )
    ax.scatter(theoretical_q, ordered_res, alpha=0.4, edgecolor="k")
    line_x = [min(theoretical_q), max(theoretical_q)]
    line_y = [slope * x + intercept for x in line_x]

    ax.plot(line_x, line_y, color="red", ls="--", lw=2)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Ordered Residuals")
    ax.set_title(f"{model_name} — Q-Q Plot of Residuals ({type})")
    return fig


def feature_importances_if_any(model, feature_names):
    # Works for RandomForest regressors
    try:
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "named_steps") and "model" in model.named_steps:
            m = model.named_steps["model"]
            if hasattr(m, "feature_importances_"):
                importances = m.feature_importances_
        if importances is not None:
            fi = pd.Series(importances, index=feature_names).sort_values(
                ascending=False
            )
            fi_df = fi.reset_index().rename(
                columns={"index": "feature", 0: "importance"}
            )
            fig, ax = plt.subplots(figsize=(6, max(3, len(fi) * 0.3)))
            fi.plot(kind="barh", ax=ax)
            ax.invert_yaxis()

            return fi_df, fig
        else:
            return None, None
    except Exception as e:
        st.error(f"Could not plot feature importance: {e}")
        return None, None


def plot_summary(train, test, metric="r2"):
    m_train = train[["model_key", metric]].copy()
    m_train["dataset"] = "Train"

    m_test = test[["model_key", metric]].copy()
    m_test["dataset"] = "Test"

    combined_m = pd.concat([m_train, m_test])

    fig, ax = plt.subplots()
    sns.barplot(
        data=combined_m, y="model_key", x=metric, hue="dataset", ax=ax, edgecolor="k"
    )
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("")
    if metric == "r2":
        ax.set_xlim(0, 1)

    return fig


## AI response:
def ai_support(provider, ai_model, inputs, type, figs=None):
    """
    provider: [logfare.ai, gemini]
    ai_model: ai_model
    inputs:  inputs
    type: string ['describe', 'corr', 'residuals', 'summary']
    """
    if provider == "logfare.ai":
        api = st.session_state["saved_api_logfare"]
        client = OpenAI(base_url="https://logfare.ai/v1", api_key=api)
    elif provider == "gemini":
        api = st.session_state["saved_api_google"]
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


@st.fragment
def update_summary():
    for xx in ["train", "test"]:
        st.subheader(
            f"\t\t _Model summary for :blue[{xx.capitalize()}] data_",
            divider="blue",
        )
        rows = []
        for k, v in st.session_state.models.items():
            # st.json(k)
            # st.json(v)
            row = {
                "model_key": k,
                "type": v.get("type"),
                "r2": v[f"metrics_{xx}"].get("r2"),
                "mse": v[f"metrics_{xx}"].get("mse"),
                "rmse": v[f"metrics_{xx}"].get("rmse"),
                "mae": v[f"metrics_{xx}"].get("mae"),
                "params": str(v.get("params")),
                "stored_at": pd.to_datetime(v.get("timestamp"), unit="s"),
            }
            rows.append(row)
        summary_df = pd.DataFrame(rows).sort_values(by="r2", ascending=False)
        st.session_state[f"summary_{xx}"] = summary_df
        st.dataframe(st.session_state[f"summary_{xx}"].reset_index(drop=True))


@st.dialog(" ")
def store_model():
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


# --- Initialize session state ---
init_session_state()

# --- Title and description ---
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
            Upload a CSV file (first row must be column headers). 
            Explore your dataset, select features and target variables, apply preprocessing, 
            split data into training and testing sets, and interactively train regression models.
        </p>
        <p style="font-size:15px;">  
            You can store trained models during the session, compare performance metrics, 
            and remove stored models as needed.
        </p>
      <ul >
        <li>CSV format required: first row must contain column headers.</li>
        <li>Only numeric columns are processed automatically; non-numeric data must be converted externally.</li>
        <li>Stored models exist only for the current session (Streamlit session state).</li>
        <li>AI-powered explanations are available for model results and diagnostics.</li>
      </ul>
    </div>
    """,
    # unsafe_allow_html=True,
)

with st.sidebar:
    st.write("AI Client")

    provider = st.selectbox("AI provider", ["logfare.ai", "gemini"])
    if provider == "logfare.ai":
        current_logfare_api = st.text_input(
            "API",
            key="api_logfare_widget",
            value=st.session_state.saved_api_logfare,
            type="password",
        )
        if current_logfare_api != st.session_state.saved_api_logfare:
            st.session_state.saved_api_logfare = current_logfare_api
        ai_model = st.selectbox(
            "Model",
            st.session_state.ai_init["logfare_models"],
        )
        st.warning(
            "* This application uses Logfare AI services to process requests. Inputs, outputs, and related data may be logged and stored by Logfare for research and operational purposes in accordance with their policies. Users should avoid submitting sensitive or confidential information.\n * Only results of analysis used in AI prompt. The original dataset not used in prompts"
        )
    elif provider == "gemini":
        current_google_api = st.text_input(
            "API",
            value=st.session_state.saved_api_google,
            key="api_google_widget",
            type="password",
        )
        if current_google_api != st.session_state.saved_api_google:
            st.session_state.saved_api_google = current_google_api
            get_google_models()
        ai_model = st.selectbox(
            "Model",
            st.session_state.ai_init["google_models"],
        )

st.markdown("---")

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

df = st.session_state.df.copy()
df = df.select_dtypes(include=[np.number])

# --- Data exploration container ---
with st.container():
    st.subheader("Data Exploration")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Data Preview",
            "Summary Stats",
            "Distributions Plots",
            "Features vs Target",
            "Customize Plot",
            "Correlation Plot",
        ]
    )

    with tab1:  # Data Preview
        st.markdown("**Preview**")
        st.dataframe(df)
        st.markdown(f"**Size:** {df.shape[0]} rows × {df.shape[1]} columns")

    with tab2:  # Summary Stats
        st.markdown("**Descriptive statistics**")
        try:
            describe_stat = describe(df)
            st.dataframe(describe_stat)
            with st.expander("AI Explanation", True, key="expand_sam"):
                if st.button("Generate Text", key="button_sam"):
                    try:
                        with st.spinner("In progress...", show_time=True):
                            st.markdown(
                                ai_support(
                                    provider=provider,
                                    ai_model=ai_model,
                                    inputs=describe_stat.to_string(),
                                    type="describe",
                                )
                            )
                    except Exception as e:
                        st.error(f"Error: {e}")
        except Exception as e:
            st.write("Could not compute describe():", e)

        st.markdown("**Column types & null counts**")
        info_df = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "non-null": df.notnull().sum(),
                "null": df.isnull().sum(),
            }
        )
        st.dataframe(info_df)

    with tab3:  # Distributions Plots
        st.markdown("**Visual exploration**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns in the uploaded dataset to plot.")
        else:
            col = st.selectbox(
                "Select numeric column to visualize", numeric_cols, index=0
            )
            dis_color = st.color_picker("Color", "#5587D7", key="disb_color")
            st.write("Histogram and boxplot")
            col1, col2 = st.columns([1, 1])
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.histplot(
                df[col].dropna(),
                kde=True,
                ax=ax,
                color=dis_color,
            )
            ax.set_title(f"Histogram: {col}")
            col1.pyplot(fig)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                df,
                x=col,
                ax=ax,
                color=dis_color,
            )
            ax.set_title(f"Box Plot: {col}")
            col2.pyplot(fig)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(
                df[col].dropna(),
                orient="h",
                color=dis_color,
                ax=ax,
            )
            ax.set_title(f"Violin Plot: {col}")
            col1.pyplot(fig)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.ecdfplot(
                df[col].dropna(),
                ax=ax,
                color=dis_color,
            )
            ax.set_title(f"ECDF (Cumulative Distribution): {col}")
            fig.tight_layout()

            col2.pyplot(fig)

    with tab4:
        output = st.selectbox("Target", df.columns, index=len(df.columns) - 1)
        cols = st.columns(3, vertical_alignment="bottom")
        tar_color = cols[0].color_picker(
            "Color",
            "#5587D7",
        )
        tar_alpha = cols[1].number_input("alpha", 0.0, 1.00, 1.00)

        # cols = st.columns(3)
        # @st.cache_data
        def plot_features(df, output):
            inputs = df.columns.drop(output)

            for i in range(0, len(inputs), 3):
                cols = st.columns(3)
                for j, input in enumerate(inputs[i : i + 3]):
                    fig, ax = plt.subplots()
                    sns.scatterplot(
                        df,
                        x=input,
                        y=output,
                        ax=ax,
                        color=tar_color,
                        edgecolor="k",
                        alpha=tar_alpha,
                    )
                    # ax.scatter(df[input].values, df[output].values)

                    cols[j].pyplot(fig)

        plot_features(df, output)
    with tab5:
        cols = st.columns(4)
        x_data = cols[0].selectbox("X axis", df.columns, index=0)
        y_data = cols[1].selectbox("Y axis", df.columns, index=len(df.columns) - 1)
        color = cols[2].color_picker("Marker Color", "#90D5FF")
        edgecolor = cols[3].color_picker("Marker Edge Color", "#000000")

        xlabel = cols[0].text_input("X Label", x_data)
        ylabel = cols[1].text_input("Y Label", y_data)
        markersize = cols[2].number_input("Marker Size", value=None)
        markershape = cols[3].selectbox("marker", markers.keys(), 0)
        alpha = cols[0].slider("transparency ", 0, 100, 100, 1) / 100

        with st.container(border=False):
            cols = st.columns(4)
            xmin = cols[0].number_input("X axis min", value=None)
            xmax = cols[1].number_input("X axis max", value=None)
            ymin = cols[2].number_input("Y axis min", value=None)
            ymax = cols[3].number_input("Y axis max", value=None)
        # if st.button("Plot"):
        fig = customize_plot(
            df,
            xdata=x_data,
            ydata=y_data,
            scatter_kwargs={
                "color": color,
                "edgecolor": edgecolor,
                "s": markersize,
                "marker": markers[markershape],
                "alpha": alpha,
            },
            ax_kwargs={
                "xlabel": xlabel,
                "ylabel": ylabel,
                "xlim": (xmin, xmax),
                "ylim": (ymin, ymax),
            },
        )
        _, col, _ = st.columns([1, 2, 1])
        col.pyplot(fig)
    with tab6:
        corr_method = st.selectbox("Method ", ["pearson", "kendall", "spearman"])
        corr_var = st.multiselect(
            "Variables to be included", options=df.columns, default=df.columns
        )
        corr = df[corr_var].corr(method=corr_method)
        cmap_dic = {
            "vlag": "vlag",
            "coolwarm": "coolwarm",
            "Spectral": "Spectral",
            "Red, Blue": "RdBu",
            "Red, Yellow, Blue": "RdYlBu",
            "Red, Yellow Green": "RdYlGn",
        }
        columns = st.columns(3)
        min_cor = columns[0].number_input("Min Value", -1.0, 1.0, -1.0)
        max_cor = columns[1].number_input("Max Value", -1.0, 1.0, 1.0)
        center_cor = columns[2].number_input("Center Value", -1.0, 1.0, 0.0)
        cmap_select = st.selectbox("Color map", cmap_dic.keys(), 0)
        st.dataframe(corr)
        _, col, _ = st.columns([1, 3, 1])
        fig, ax = plt.subplots(figsize=(corr.shape[0], corr.shape[0] / 2))
        sns.heatmap(
            corr,
            ax=ax,
            annot=True,
            # cmap="vlag",
            cmap=cmap_dic[cmap_select],
            vmin=min_cor,
            vmax=max_cor,
            center=center_cor,
        )
        col.pyplot(fig)

        with st.expander("AI Explanation", False, key="expand_cor"):
            if st.button("Generate Text", key="button_cor"):
                try:
                    with st.spinner("In progress...", show_time=True):
                        st.markdown(
                            ai_support(
                                provider=provider,
                                ai_model=ai_model,
                                inputs=df.corr(method=corr_method),
                                type="corr",
                            )
                        )
                except Exception as e:
                    st.error(f"Error: {e}")


st.markdown("---")

# --- Preprocessing container ---
with st.container():
    st.subheader("Preprocessing & Train/Test Split")

    # Feature/target selection
    all_cols = df.columns.tolist()
    target_col = st.selectbox(
        "Select target column (y)", all_cols, index=len(all_cols) - 1
    )
    feature_cols = st.multiselect(
        "Select feature columns (X). If none selected, all numeric columns except target will be used",
        options=[c for c in all_cols if c != target_col],
        default=[
            c
            for c in df.select_dtypes(include=[np.number]).columns.tolist()
            if c != target_col
        ],
    )

    feature_cols = [col for col in df.columns if col in feature_cols]

    if not feature_cols:
        st.warning(
            "No feature columns selected. Select at least one feature column to proceed."
        )
    # Missing values Handling
    helper_ma = """
            ### Missing Value Handling Options

            | Option | Description | 
            |-------|-------------|
            | **Drop rows (remove missing)** | Remove rows containing missing cells |
            | **Fill with Median** | Replace missing values with the column median | 
            | **Fill with Mean** | Replace missing values with the column mean | 
            | **Fill with Mode** | Replace missing values with the most frequent value | 
            | **Fill with Zero** | Replace missing values with 0 |
            | **Forward Fill (ffill)** | Carry the previous value forward |
            | **Backward Fill (bfill)** | Use the next value to fill missing entries | 
        """
    missing_methods = {
        "Drop rows": lambda df: df.dropna(),
        "Fill with Median": lambda df: df.fillna(df.median()),
        "Fill with Mean": lambda df: df.fillna(df.mean()),
        "Fill with Mode": lambda df: df.fillna(df.mode().iloc[0]),
        "Fill with Zero": lambda df: df.fillna(0),
        "Forward Fill": lambda df: df.ffill(),
        "Backward Fill": lambda df: df.bfill(),
    }
    missing_choice = st.selectbox(
        "Missing Values Fill",
        missing_methods.keys(),
        0,
        help=helper_ma,
    )

    test_size = st.slider(
        "Test set size (fraction)", min_value=0.05, max_value=0.5, value=0.2, step=0.01
    )
    # random_state = st.number_input("Random state (integer)", value=42, step=1)
    random_state = 42
    apply_split = st.button("Apply split / preview train/test")

    if apply_split:
        # Keep only numeric features automatically (imputer will handle missing)
        df_temp = missing_methods[missing_choice](df)
        X = df_temp[feature_cols]
        y = df_temp[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )
        st.session_state.split = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "test_size": test_size,
            "random_state": int(random_state),
            "feature_cols": feature_cols,
            "target_col": target_col,
        }
        st.success(
            f"Split applied: train={X_train.shape[0]} rows, test={X_test.shape[0]} rows."
        )
        st.dataframe(pd.concat([X_train.head(), y_train.head()], axis=1))

# If no split in session_state, create a default one (not applied until user clicks)
if not st.session_state.split:
    # default split using numeric features (if any)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    default_features = [c for c in numeric_cols if c != df.columns[-1]]
    default_target = df.columns[-1]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df[default_features], df[default_target], test_size=0.2, random_state=42
        )
        st.session_state.split = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "test_size": 0.2,
            "random_state": 42,
            "feature_cols": default_features,
            "target_col": default_target,
        }
    except Exception:
        # can't auto-split (no numeric features), keep empty split - user must select
        pass

# If split still empty, stop
if not st.session_state.split:
    st.warning(
        "No valid train/test split available. Ensure you have numeric features and a numeric target."
    )
    st.stop()

split = st.session_state.split

st.markdown("---")

# --- Model training container ---

# DICTIONARY OF REGRESSOR
regressors = {
    "linear": "Linear Regression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "randforst": "Random Forest",
    "svr": "Support Vector Regression",
    "lsvr": "Linear Support Vector Regression",
    "nusvr": "Nu Support Vector Regression",
}


with st.container():
    st.subheader("Model Training")

    model_type = st.selectbox(
        "Choose regression algorithm",
        options=[
            "Linear Regression",
            "Lasso",
            "Ridge",
            "Elastic Net",
            "Random Forest",
            "KNN (K-Nearest Neighbors)",
            "SVR (Support Vector Regression)",
            "Gradient Boosting",
            "XGBoost",
            "Neural Network (MLP)",
        ],
        on_change=reset_model_train_state,
    )

    # dynamic hyperparameters
    st.markdown(":red[_**Hyperparameters**_]")
    # Defaults
    params = {}
    if model_type == "Linear Regression":
        st.markdown("No hyperparameters for basic Linear Regression.")
        use_scaler = st.checkbox(
            "Use Scaler",
            value=True,
            help="Standardize features by removing the mean and scaling to unit variance.",
        )
        # use_scaler = True

    elif model_type == "Lasso":
        st.text(
            "Lasso regression (Least Absolute Shrinkage and Selection Operator) is a regression analysis method that performs both variable selection and regularization"
        )
        alpha = st.number_input(
            "alpha (regularization)",
            min_value=0.0,
            value=0.1,
            step=0.1,
            help="Constant that multiplies the L1 term, controlling regularization strength",
        )
        params["alpha"] = float(alpha)

        use_scaler = True
        use_scaler = st.checkbox(
            "Use Scaler",
            value=True,
            help="Standardize features by removing the mean and scaling to unit variance.",
        )
    elif model_type == "Ridge":
        st.text(
            "Ridge regression (L2 regularization) is a technique used to improve regression models by reducing overfitting and handling multicollinearity"
        )
        alpha = st.number_input("alpha", min_value=0.0, value=1.0, step=0.1)
        params["alpha"] = float(alpha)
        use_scaler = True
        use_scaler = st.checkbox(
            "Use Scaler",
            value=True,
            help="Standardize features by removing the mean and scaling to unit variance.",
        )
    elif model_type == "Elastic Net":
        st.text(
            "ElasticNet is a linear regression regularization technique that combines both L1 (Lasso) and L2 (Ridge) penalties to enhance prediction accuracy and handle multicollinearity",
        )
        alpha = st.number_input(
            "alpha",
            min_value=0.0,
            value=1.0,
            step=0.1,
            help="Constant that multiplies the penalty terms",
        )
        l1_ratio = st.number_input(
            "L1 ratio", min_value=0.0, value=0.5, max_value=1.0, step=0.1, help=""
        )
        params.update({"alpha": alpha, "l1_ratio": l1_ratio})
        use_scaler = st.checkbox(
            "Use Scaler",
            value=True,
            help="Standardize features by removing the mean and scaling to unit variance.",
        )
    elif model_type == "Random Forest":
        n_estimators = st.number_input(
            "n_estimators", min_value=10, max_value=2000, value=100, step=10
        )
        max_depth = st.number_input(
            "max_depth", min_value=0, max_value=100, value=0, step=1
        )
        min_samples_leaf = st.number_input(
            "min_samples_leaf", min_value=1, max_value=50, value=1, step=1
        )
        params.update(
            {
                "n_estimators": int(n_estimators),
                "max_depth": int(max_depth) if int(max_depth) > 0 else None,
                "min_samples_leaf": int(min_samples_leaf),
            }
        )
        use_scaler = False  # trees don't need scaling
        use_scaler = st.checkbox(
            "Use Scaler",
            value=False,
            help="Standardize features by removing the mean and scaling to unit variance.",
        )
    elif model_type == "KNN (K-Nearest Neighbors)":
        st.text(
            "KNN (K-Nearest Neighbors) regression is a non-parametric learning algorithm that predicts a continuous target value by averaging the values of the closest data points (neighbors) in the feature space."
        )
        n_neighbors = st.number_input(
            "Number of neighbors", min_value=1, value=5, step=1
        )
        help = """
        ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.\\
        ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away."""
        weights = st.selectbox("Weight", ["uniform", "distance"], 0, help=help)
        leaf_size = st.number_input(
            "leaf size",
            min_value=1,
            value=30,
        )

        params.update(
            {
                "n_neighbors": int(n_neighbors),
                "weights": weights,
                "leaf_size": int(leaf_size),
            }
        )
        use_scaler = st.checkbox(
            "Use Scaler",
            value=False,
            help="Standardize features by removing the mean and scaling to unit variance.",
        )

    elif model_type == "SVR":
        st.text("Support Vector Regression.")
        kernel = st.selectbox("kernel", options=["rbf", "linear", "poly", "sigmoid"])
        C = st.number_input("C (regularization)", min_value=0.01, value=1.0, step=0.01)
        gamma = st.number_input(
            "gamma (Kernel coefficient)",
            min_value=0.0,
            value=1 / (split["X_train"].shape[1] * np.var(split["X_train"].values)),
            # step=0.0001,
            format="%0.4f",
        )
        degree = st.number_input(
            "degree: (polynomial kernel function (‘poly’))", 2, value=3, step=1
        )
        params.update(
            {"C": float(C), "kernel": kernel, "gamma": gamma, "degree": degree}
        )
        use_scaler = True

        use_scaler = st.checkbox(
            "Use Scaler",
            value=True,
            help="Standardize features by removing the mean and scaling to unit variance.",
        )
    elif model_type == "XGBoost":
        n_estimators = st.number_input(
            "n_estimators", min_value=10, max_value=2000, value=100, step=10
        )
        learning_rate = st.number_input(
            "learining_rate", min_value=0.01, max_value=1.0, value=0.01, step=0.01
        )
        max_depth = st.number_input(
            "max_depth", min_value=1, max_value=20, value=3, step=1
        )
        params.update(
            {
                "n_estimators": int(n_estimators),
                "learning_rate": float(learning_rate),
                "max_depth": int(max_depth),
            }
        )
        use_scaler = False
        use_scaler = st.checkbox(
            "Use Scaler",
            value=False,
            help="Standardize features by removing the mean and scaling to unit variance.",
        )
    elif model_type == "Gradient Boosting":
        n_estimators = st.number_input(
            "n_estimators", min_value=10, max_value=2000, value=100, step=10
        )
        learning_rate = st.number_input(
            "learning_rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01
        )
        max_depth = st.number_input(
            "max_depth", min_value=1, max_value=20, value=3, step=1
        )
        params.update(
            {
                "n_estimators": int(n_estimators),
                "learning_rate": float(learning_rate),
                "max_depth": int(max_depth),
            }
        )
        use_scaler = False
        use_scaler = st.checkbox(
            "Use Scaler",
            value=False,
            help="Standardize features by removing the mean and scaling to unit variance.",
        )
    elif model_type == "Neural Network (MLP)":
        st.text("")
        hidden_layers = st.text_input(
            "Hidden Layer Sizes (comma separated, e.g., 100,50)", value="100"
        )
        activation = st.selectbox(
            "Activation Function", options=["relu", "tanh", "logistic", "identity"]
        )
        max_iter = st.number_input(
            "Max Iterations", min_value=100, max_value=2000, value=500, step=100
        )
        try:
            hl_sizes = tuple([int(x.strip()) for x in hidden_layers.split(",")])
        except Exception:
            hl_sizes = (100,)
            st.warning("Invalid hidden layer sizes. Defaulting to (100,).")
        params.update(
            {
                "hidden_layer_sizes": hl_sizes,
                "activation": activation,
                "max_iter": int(max_iter),
            }
        )
        use_scaler = True
        use_scaler = st.checkbox(
            "Use Scaler",
            value=True,
            help="Standardize features by removing the mean and scaling to unit variance.",
        )
    # Model name for storage key
    model_name_input = st.text_input(
        "Model name (used as key when storing). Default = selected algorithm",
        value=model_type,
        help="Note: Exciting stored will be overwritten",
    )

    # Buttons: Train, Store, Delete
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    # with col1:
    if col1.button("Train model (fit & show metrics)"):
        # build the model object
        try:
            X_train = split["X_train"][split["feature_cols"]]
            X_test = split["X_test"][split["feature_cols"]]
            y_train = split["y_train"]
            y_test = split["y_test"]

            if model_type == "Linear Regression":
                model_obj = LinearRegression()
            elif model_type == "Ridge":
                model_obj = Ridge(**params)
            elif model_type == "Lasso":
                model_obj = Lasso(**params)
            elif model_type == "Elastic Net":
                model_obj = ElasticNet(**params)
            elif model_type == "Random Forest":
                model_obj = RandomForestRegressor(**params)
            elif model_type == "KNN (K-Nearest Neighbors)":
                model_obj = KNeighborsRegressor(**params)
            elif model_type == "SVR":
                model_obj = SVR(**params)
            elif model_type == "XGBoost":
                model_obj = XGBRegressor(**params)
            elif model_type == "Gradient Boosting":
                model_obj = GradientBoostingRegressor(**params)
            elif model_type == "Neural Network (MLP)":
                model_obj = MLPRegressor(**params)

            pipeline = build_pipeline(model_obj, use_scaler=use_scaler)
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            preds_train = pipeline.predict(X_train)

            # Meterics for Train
            metrics_train = compute_metrics(y_train, preds_train)
            metrics_test = compute_metrics(y_test, preds)

            # st.dataframe(metrics, "content")

            # Plots
            # predictions
            fig_ptrain = plot_predictions(
                y_train, preds_train, model_name_input, type="Train"
            )
            fig_ptest = plot_predictions(y_test, preds, model_name_input, type="Test")
            # residual histogram
            fig_r_train = plot_residuals(
                y_train, preds_train, model_name_input, type="Train"
            )
            fig_r_test = plot_residuals(y_test, preds, model_name_input, type="Test")

            # residual vs fitted
            fig_rf_train = plot_residuals_vs_fitted(
                y_train, preds_train, model_name_input, type="Train"
            )
            fig_rf_test = plot_residuals_vs_fitted(
                y_test, preds, model_name_input, type="Test"
            )
            # QQ plot
            fig_qq_train = plot_qq(y_train, preds_train, model_name_input, type="Train")
            fig_qq_test = plot_qq(y_test, preds, model_name_input, type="Test")

            # Feature importances
            fi_df, fig_f = feature_importances_if_any(pipeline, split["feature_cols"])

            # save result to session state
            st.session_state.current_view = {
                "metrics_train": metrics_train,
                "metrics_test": metrics_test,
                "fig_ptrain": fig_ptrain,
                "fig_ptest": fig_ptest,
                "fig_r_train": fig_r_train,
                "fig_r_test": fig_r_test,
                "fig_rf_train": fig_rf_train,
                "fig_rf_test": fig_rf_test,
                "fig_qq_train": fig_qq_train,
                "fig_qq_test": fig_qq_test,
                "fig_f": fig_f,
                "fi_df": fi_df,
                "ai_explain": None,
            }

            # keep last trained model in session state temporarily
            st.session_state.last_train = {
                "name": model_name_input,
                "type": model_type,
                "pipeline": pipeline,
                "metrics_test": metrics_test,
                "metrics_train": metrics_train,
                "params": params,
                "timestamp": time.time(),
            }
            # change model train state
            st.session_state.model_train_state = 1
        except Exception as e:
            st.error(f"Error during training: {e}")

    if st.session_state.current_view is not None:
        cv = st.session_state.current_view

        if st.session_state.model_train_state is not None:
            st.success("Model trained.")
        else:
            st.warning(f"{model_type} model not trained")
        st.markdown(f"## {model_type} model result")
        cols = st.columns(2)
        cols[0].markdown("**Metrics on train set**")
        cols[0].json(cv["metrics_train"])
        cols[1].markdown("**Metrics on test set**")
        cols[1].json(cv["metrics_test"])

        # save model as pickle
        try:
            pipeline = st.session_state.last_train["pipeline"]
            b = pickle.dumps(pipeline)
            st.download_button(
                "Download Model",
                data=b,
                file_name=f"{st.session_state.last_train['name']}.model",
                mime="application/octet-stream",
                help="Download Model",
            )

        except Exception as e:
            st.error(f"Error: {e}")

        diag_tab1, diag_tab2, diag_tab3 = st.tabs(
            ["Predictions", "Residuals", "Feature Importance"]
        )

        with diag_tab1:
            st.markdown("**Actual vs. Predicted Values**")
            cols = st.columns(2, vertical_alignment="bottom")
            cols[0].pyplot(cv["fig_ptrain"])
            cols[1].pyplot(cv["fig_ptest"])

        with diag_tab2:
            cols = st.columns(3, vertical_alignment="bottom")
            cols[0].pyplot(cv["fig_r_train"])
            cols[1].pyplot(cv["fig_rf_train"])
            cols[2].pyplot(cv["fig_qq_train"])
            st.markdown(":blue[---]")
            # if view_set == "Test Data":
            # cols = st.columns(3, vertical_alignment="bottom")
            cols[0].pyplot(cv["fig_r_test"])
            cols[1].pyplot(cv["fig_rf_test"])
            cols[2].pyplot(cv["fig_qq_test"])

        cols = st.columns(3, vertical_alignment="bottom")
        with diag_tab3:
            if cv["fig_f"] is not None:
                st.dataframe(cv["fi_df"])
                _, col11, _ = st.columns([1, 4, 1])
                col11.pyplot(cv["fig_f"])
                # col22.pyplot(fig_rad)
            else:
                st.info("Feature importance not available for this model type.")

        with st.expander("AI Explanation ", False, key="expand_figs"):
            if st.button("Generate Text", key="button_figs"):
                with st.spinner("In progress...", show_time=True):
                    resp = ai_support(
                        provider=provider,
                        ai_model=ai_model,
                        inputs=None,
                        type="residuals",
                        figs=[
                            cv["fig_ptrain"],
                            cv["fig_ptest"],
                            cv["fig_r_train"],
                            cv["fig_rf_train"],
                            cv["fig_qq_train"],
                            cv["fig_r_test"],
                            cv["fig_rf_test"],
                            cv["fig_qq_test"],
                        ],
                    )
                    st.session_state.current_view["ai_explain"] = resp
            if cv.get("ai_explain") is not None:
                st.markdown(st.session_state.current_view["ai_explain"])

    col2.button("Store model (save into session dictionary)", on_click=store_model)

st.markdown("---")

# --- Stored models summary container ---
with st.container():
    st.subheader("Stored Models Summary")
    if not st.session_state.models:
        st.info(
            "No models stored yet. Train and store a model to see the summary here."
        )
    else:
        # Build summary DataFrame for display
        # update_summary()
        for xx in ["train", "test"]:
            st.subheader(
                f"\t\t _Model summary for :blue[{xx.capitalize()}] data_",
                divider="blue",
            )
            rows = []
            for k, v in st.session_state.models.items():
                # st.json(k)
                # st.json(v)
                row = {
                    "model_key": k,
                    "type": v.get("type"),
                    "r2": v[f"metrics_{xx}"].get("r2"),
                    "mse": v[f"metrics_{xx}"].get("mse"),
                    "rmse": v[f"metrics_{xx}"].get("rmse"),
                    "mae": v[f"metrics_{xx}"].get("mae"),
                    "params": str(v.get("params")),
                    "stored_at": pd.to_datetime(v.get("timestamp"), unit="s"),
                }
                rows.append(row)
            summary_df = pd.DataFrame(rows).sort_values(by="r2", ascending=False)
            st.session_state[f"summary_{xx}"] = summary_df
            st.dataframe(
                st.session_state[f"summary_{xx}"]
                .reset_index(drop=True)
                .style.highlight_max(subset=["r2"], color="lightgreen")
                .highlight_min(subset=["mse", "rmse", "mae"], color="lightgreen")
            )
        col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
        key_to_delete = col1.selectbox(
            "Delete model", options=list(st.session_state.models.keys())
        )
        if col2.button(
            "Delete",
        ):
            del st.session_state.models[key_to_delete]
            st.session_state["ai_summary"] = None
            st.rerun()
            col3.success(f"Deleted model '{key_to_delete}'.")
            # update_summary()

        col1, col2 = st.columns([1, 1])
        # col.pyplot(fig)
        fig_r2 = plot_summary(
            st.session_state["summary_train"], st.session_state["summary_test"], "r2"
        )
        fig_mse = plot_summary(
            st.session_state["summary_train"], st.session_state["summary_test"], "mse"
        )
        fig_rmse = plot_summary(
            st.session_state["summary_train"], st.session_state["summary_test"], "rmse"
        )
        fig_mae = plot_summary(
            st.session_state["summary_train"], st.session_state["summary_test"], "mae"
        )

        col1.pyplot(fig_r2)
        col2.pyplot(fig_mse)
        col1.pyplot(fig_rmse)
        col2.pyplot(fig_mae)

        with st.expander("AI Explanation", False, key="expand_summ"):
            if st.button("Generate Text", key="button_summ"):
                with st.spinner("In progress...", show_time=True):
                    resp_sum = ai_support(
                        provider=provider,
                        ai_model=ai_model,
                        inputs=[
                            st.session_state["summary_train"].to_string(),
                            st.session_state["summary_test"].to_string(),
                        ],
                        type="summary",
                        figs=None,
                    )
                    st.session_state["ai_summary"] = resp_sum
            if st.session_state["ai_summary"] is not None:
                st.markdown(st.session_state["ai_summary"])
st.markdown("---")
