import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Predict", layout="wide")
with st.sidebar:
    st.write("")


st.title("Predict Using Uploaded Model")

uploaded_model = st.file_uploader("Upload trained model", type=["model"])

if uploaded_model is not None:
    try:
        model = pickle.load(uploaded_model)
        st.success("Model loaded successfully!")

        tab_p, tab_i = st.tabs(["Prediction", "Model Info"])

        with tab_p:
            features = model.feature_names_in_.tolist()

            st.subheader("Enter values")

            inputs = []
            for feature in features:
                inputs.append(st.number_input(feature, value=0.0))

            if st.button("Predict"):
                x = np.array([inputs])
                pred = model.predict(x)
                st.success(f"Prediction: {pred[0]}")

        with tab_i:
            st.subheader("Model Information")
            if hasattr(model, "named_steps"):
                step_names = list(model.named_steps.keys())
                final_name = step_names[-1]
                final_model = model.named_steps[final_name]

            else:
                final_model = model

            st.write("Model class: ", final_model.__class__.__name__)

            if step_names:
                with st.expander("Pipeline Steps", expanded=False):
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Step": step_names,
                                "Class": [
                                    model.named_steps[step].__class__.__name__
                                    for step in step_names
                                ],
                            },
                        ),
                        hide_index=True,
                    )

            if hasattr(model, "feature_names_in_"):
                with st.expander("Feature Names", expanded=False):
                    st.dataframe(
                        pd.DataFrame({"Feature": model.feature_names_in_}),
                        hide_index=True,
                    )

            if hasattr(final_model, "coef_") or hasattr(final_model, "intercept_"):
                with st.expander("Model Coefficients", expanded=False):
                    # st.write("Model coefficients and intercept:")
                    coef_data = {
                        "Feature": np.append(model.feature_names_in_, "Intercept"),
                        "Coefficient": np.append(
                            getattr(
                                final_model,
                                "coef_",
                                [None] * len(model.feature_names_in_),
                            ),
                            getattr(final_model, "intercept_", None),
                        ),
                    }
                    st.dataframe(pd.DataFrame(coef_data), hide_index=True)

            scaler_found = False
            if hasattr(model, "named_steps"):
                for step_name, step_obj in model.named_steps.items():
                    if (
                        "scaler" in step_name.lower()
                        or step_obj.__class__.__name__.lower().endswith("scaler")
                    ):
                        scaler_found = True
                        with st.expander(f"Scaler Info ({step_name})", expanded=False):
                            st.write(f"Scaler class: {step_obj.__class__.__name__}")
                            attrs = {}
            if hasattr(final_model, "feature_importances_"):
                with st.expander("Feature Importances", expanded=False):
                    importance_data = {
                        "Feature": model.feature_names_in_,
                        "Importance": final_model.feature_importances_,
                    }
                    st.dataframe(pd.DataFrame(importance_data), hide_index=True)
            if hasattr(final_model, "support_vectors_"):
                with st.expander("Support Vectors", expanded=False):
                    st.write(
                        "Support vectors shape: ", final_model.support_vectors_.shape
                    )
                    # st.dataframe(
                    #     pd.DataFrame(final_model.support_vectors_),
                    #     hide_index=True,
                    # )
            if hasattr(final_model, "get_params"):
                with st.expander("Model Parameters (Advanced)", expanded=False):
                    params = final_model.get_params()
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Parameter": list(params.keys()),
                                "Value": list(params.values()),
                            }
                        ),
                        hide_index=True,
                    )
    except Exception as e:
        st.error(f"Could not load model: {e}")
