import streamlit as st
import numpy as np
import pandas as pd
from src.utils import build_pipeline, compute_metrics, store_model
from src.plots import *
from src.ai_setting import ai_support
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import time
import pickle


# --- Model training container ---
def reset_model_train_state():
    st.session_state.model_train_state = None


# @st.dialog(" ")
# def store_model():
#     if st.session_state.get("last_train") is None:
#         st.warning(
#             "No model trained in this session to store. Train first, then store."
#         )
#     else:
#         key = model_name_input
#         # store a copy of the pipeline and metrics
#         entry = {
#             "type": st.session_state.last_train.get("type"),
#             "pipeline": st.session_state.last_train.get("pipeline"),
#             "metrics_test": st.session_state.last_train.get("metrics_test"),
#             "metrics_train": st.session_state.last_train.get("metrics_train"),
#             "params": st.session_state.last_train.get("params"),
#             "timestamp": st.session_state.last_train.get("timestamp"),
#         }
#         st.session_state.models[key] = entry
#         st.success(f"Stored model under key: '{key}'")
#         st.session_state["ai_summary"] = None


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

if st.session_state.df is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

split = st.session_state.split

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

    elif model_type == "SVR (Support Vector Regression)":
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
            elif model_type == "SVR (Support Vector Regression)":
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
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
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
            st.success(f" {st.session_state.last_train['type']}Model trained.")
        else:
            st.warning(f"{model_type} model not trained")
        st.markdown(f"## {st.session_state.last_train['type']} model result")
        cols = st.columns(2)
        cols[0].markdown("**Metrics on train set**")
        # cols[0].json(cv["metrics_train"])
        cols[0].dataframe(
            pd.DataFrame(
                list(cv["metrics_train"].items()), columns=["Metric", "Value"]
            ),
            column_config={
                "Value": st.column_config.NumberColumn(format="%0.3f"),
                "Metric": st.column_config.TextColumn("Metric", width="small"),
            },
            hide_index=True,
        )
        cols[1].markdown("**Metrics on test set**")
        # cols[1].json(cv["metrics_test"])
        cols[1].dataframe(
            pd.DataFrame(list(cv["metrics_test"].items()), columns=["Metric", "Value"]),
            column_config={
                "Value": st.column_config.NumberColumn(format="%0.3f"),
                "Metric": st.column_config.TextColumn("Metric", width="small"),
            },
            hide_index=True,
        )
        # save model as pickle
        # try:
        #     pipeline = st.session_state.last_train["pipeline"]
        #     b = pickle.dumps(pipeline)
        #     st.download_button(
        #         "Download Model",
        #         data=b,
        #         file_name=f"{st.session_state.last_train['name']}.model",
        #         mime="application/octet-stream",
        #         help="Download Model",
        #     )

        # except Exception as e:
        #     st.error(f"Error: {e}")

        diag_tab1, diag_tab2, diag_tab3, diag_tab4 = st.tabs(
            ["Predictions", "Residuals", "Feature Importance", "SHAP (beta)"]
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

        # cols = st.columns(3, vertical_alignment="bottom")
        with diag_tab3:
            if cv["fig_f"] is not None:
                st.dataframe(cv["fi_df"])
                _, col11, _ = st.columns([1, 4, 1])
                col11.pyplot(cv["fig_f"])
                # col22.pyplot(fig_rad)
            else:
                st.info("Feature importance not available for this model type.")

        with diag_tab4:
            from src.shap import explain, plot_beeswarm

            st.warning(
                "Currently supports Linear Models only. More algorithms coming soon!"
            )
            if st.session_state.last_train["type"] in [
                "Linear Regression",
                "Ridge",
                "Lasso",
                "Elastic Net",
            ]:
                type = "linear"
            elif st.session_state.last_train["type"] in [
                "Random Forest",
                "XGBoost",
                "Gradient Boosting",
            ]:
                type = "tree"

            elif st.session_state.last_train["type"] in [
                "SVR (Support Vector Regression)",
                "KNN (K-Nearest Neighbors)",
            ]:
                type = "kernal"
            #
            #  "Neural Network (MLP)"

            try:
                shap_values = explain(
                    type,
                    st.session_state.last_train["pipeline"][-1],
                    st.session_state.current_view["X_test"],
                )
                with st.columns([1, 3, 1])[1]:
                    fig = plot_beeswarm(shap_values)
                    st.pyplot(fig)
            except Exception as e:
                st.error(e)

        with st.expander("AI Explanation ", False, key="expand_figs"):
            if st.button("Generate Text", key="button_figs"):
                with st.spinner("In progress...", show_time=True):
                    resp = ai_support(
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

    col2.button(
        "Store model (save into session dictionary)",
        on_click=lambda: store_model(model_name_input),
    )

    try:
        pipeline = st.session_state.last_train["pipeline"]
        b = pickle.dumps(pipeline)
        col3.download_button(
            "Download Model",
            data=b,
            file_name=f"{st.session_state.last_train['name']}.model",
            mime="application/octet-stream",
            help="Download Model",
            key="fownload_3",
        )

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")

# # --- Stored models summary container ---
# with st.container():
#     st.subheader("Stored Models Summary")
#     if not st.session_state.models:
#         st.info(
#             "No models stored yet. Train and store a model to see the summary here."
#         )
#     else:
#         # Build summary DataFrame for display
#         # update_summary()
#         for xx in ["train", "test"]:
#             st.subheader(
#                 f"\t\t _Model summary for :blue[{xx.capitalize()}] data_",
#                 divider="blue",
#             )
#             rows = []
#             for k, v in st.session_state.models.items():
#                 # st.json(k)
#                 # st.json(v)
#                 row = {
#                     "model_key": k,
#                     "type": v.get("type"),
#                     "r2": v[f"metrics_{xx}"].get("r2"),
#                     "mse": v[f"metrics_{xx}"].get("mse"),
#                     "rmse": v[f"metrics_{xx}"].get("rmse"),
#                     "mae": v[f"metrics_{xx}"].get("mae"),
#                     "params": str(v.get("params")),
#                     "stored_at": pd.to_datetime(v.get("timestamp"), unit="s"),
#                 }
#                 rows.append(row)
#             summary_df = pd.DataFrame(rows).sort_values(by="r2", ascending=False)
#             st.session_state[f"summary_{xx}"] = summary_df
#             st.dataframe(
#                 st.session_state[f"summary_{xx}"]
#                 .reset_index(drop=True)
#                 .style.highlight_max(subset=["r2"], color="lightgreen")
#                 .highlight_min(subset=["mse", "rmse", "mae"], color="lightgreen")
#             )
#         col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
#         key_to_delete = col1.selectbox(
#             "Delete model", options=list(st.session_state.models.keys())
#         )
#         if col2.button(
#             "Delete",
#         ):
#             del st.session_state.models[key_to_delete]
#             st.session_state["ai_summary"] = None
#             st.rerun()
#             col3.success(f"Deleted model '{key_to_delete}'.")
#             # update_summary()

#         col1, col2 = st.columns([1, 1])
#         # col.pyplot(fig)
#         fig_r2 = plot_summary(
#             st.session_state["summary_train"], st.session_state["summary_test"], "r2"
#         )
#         fig_mse = plot_summary(
#             st.session_state["summary_train"], st.session_state["summary_test"], "mse"
#         )
#         fig_rmse = plot_summary(
#             st.session_state["summary_train"], st.session_state["summary_test"], "rmse"
#         )
#         fig_mae = plot_summary(
#             st.session_state["summary_train"], st.session_state["summary_test"], "mae"
#         )

#         col1.pyplot(fig_r2)
#         col2.pyplot(fig_mse)
#         col1.pyplot(fig_rmse)
#         col2.pyplot(fig_mae)

#         with st.expander("AI Explanation", False, key="expand_summ"):
#             if st.button("Generate Text", key="button_summ"):
#                 with st.spinner("In progress...", show_time=True):
#                     resp_sum = ai_support(
#                         inputs=[
#                             st.session_state["summary_train"].to_string(),
#                             st.session_state["summary_test"].to_string(),
#                         ],
#                         type="summary",
#                         figs=None,
#                     )
#                     st.session_state["ai_summary"] = resp_sum
#             if st.session_state["ai_summary"] is not None:
#                 st.markdown(st.session_state["ai_summary"])
# st.markdown("---")
#
