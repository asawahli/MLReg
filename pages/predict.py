import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Initialize the model key in session state if it doesn't exist yet
if "model" not in st.session_state:
    st.session_state.model = None

st.title("Predict with Your Model")

model_option = st.radio(
    "Choose Model Source:",
    ["Use Model in Session", "Upload New Model"],
    horizontal=True,
)

# Handle file uploading and save it straight to the session state
if model_option == "Upload New Model":
    st.session_state.model = None
    uploaded_model = st.file_uploader("Upload trained model", type=["model"])
    if uploaded_model is not None:
        try:
            st.session_state.model = pickle.load(uploaded_model)
            st.success("New model loaded and saved to session state successfully!")
        except Exception as e:
            st.error(f"Could not load uploaded model: {e}")
else:
    try:
        st.session_state.model = st.session_state.last_train["pipeline"]
    except Exception as e:
        if st.session_state.last_train is None:
            st.info("No model from last training session found. Please train a model.")
        else:
            st.error(f"Could not load model from last training session: {e}")

# Point our operational 'model' variable to whatever is currently in the session state
model = st.session_state.model

# 5. Run your prediction & inspection tabs only if a model is actively loaded
if model is not None:
    try:
        tab_p, tab_i = st.tabs(["Prediction", "Model Info"])

        with tab_p:
            features = model.feature_names_in_.tolist()

            st.subheader("Enter values")

            inputs = []

            for i in range(0, len(features), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(features):
                        feature_name = features[i + j]
                        with cols[j]:
                            inputs.append(st.number_input(feature_name, value=0.0))

            # for feature in features:
            #     inputs.append(st.number_input(feature, value=0.0))

            if st.button("Predict"):
                x = np.array([inputs])
                pred = model.predict(x)
                st.success(f"Prediction: {pred[0]}")

        with tab_i:
            st.subheader("Model Information")

            # Quick Fix: Initialized step_names here to prevent a potential NameError
            # down the line if the model isn't a Pipeline object.
            step_names = []
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
        st.error(f"An error occurred while parsing the model details: {e}")
else:
    # Friendly fallback notice if no model exists in the session state yet
    st.info(
        "No model is currently active in this session. Please select 'Upload New Model' above to begin."
    )
