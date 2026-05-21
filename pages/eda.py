import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import describe
from src.plots import *
from src.ai_setting import ai_support

if st.session_state.df is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = st.session_state.df.copy()
df = df.select_dtypes(include=[np.number])

st.title("Data Exploration")
with st.container():
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
        st.markdown("**Data Health**")
        info_df = pd.DataFrame(
            {
                # "dtype": df.dtypes.astype(str),
                "Available": df.notnull().sum(),
                "Missing": df.isnull().sum(),
                "Completion %": (df.notnull().sum() / len(df) * 100).round(2),
            }
        )
        st.dataframe(info_df)
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
                                    inputs=describe_stat.to_string(),
                                    type="describe",
                                )
                            )
                    except Exception as e:
                        st.error(f"Error: {e}")
        except Exception as e:
            st.write("Could not compute describe():", e)

        # st.markdown("**Data Health**")
        # info_df = pd.DataFrame(
        #     {
        #         #"dtype": df.dtypes.astype(str),
        #         "Available": df.notnull().sum(),
        #         "Missing": df.isnull().sum(),
        #         "Completion %": (df.notnull().sum()/len(df) * 100).round(2)
        #     }
        # )
        # st.dataframe(info_df)

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
        tar_alpha = cols[1].slider("Opacity ", 0, 100, 100, 1, key="Opacity01") / 100

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
        color = cols[2].color_picker("Marker Color", "#5587D7")
        edgecolor = cols[3].color_picker("Marker Edge Color", "#000000")

        xlabel = cols[0].text_input("X Label", x_data)
        ylabel = cols[1].text_input("Y Label", y_data)
        markersize = cols[2].number_input("Marker Size", value=None)
        markershape = cols[3].selectbox("marker", markers.keys(), 0)
        alpha = cols[0].slider("Opacity ", 0, 100, 100, 1, key="Opacity02") / 100

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
                                inputs=df.corr(method=corr_method),
                                type="corr",
                            )
                        )
                except Exception as e:
                    st.error(f"Error: {e}")
