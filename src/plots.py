import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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


def customize_plot(df, xdata, ydata, scatter_kwargs=None, ax_kwargs=None):
    fig, ax = plt.subplots()

    ax.scatter(df[xdata].values, df[ydata], **scatter_kwargs)

    if ax_kwargs:
        for k, v in ax_kwargs.items():
            setter = getattr(ax, f"set_{k}", None)
            if callable(setter):
                setter(v)

    return fig


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
