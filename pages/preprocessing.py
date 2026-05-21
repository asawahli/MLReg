import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


if st.session_state.df is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()


df = st.session_state.df.copy()

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
        df_temp = df.copy()
        df_temp = df_temp[feature_cols + [target_col]]
        df_temp = missing_methods[missing_choice](df_temp)
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
        # st.dataframe(pd.concat([X_train.head(), y_train.head()], axis=1))

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
