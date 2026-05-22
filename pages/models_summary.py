import streamlit as st
from src.ai_setting import ai_support
from src.plots import plot_summary
import pandas as pd


# --- Stored models summary container ---
st.title("Models Summary")
with st.container():
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
        tabs = st.tabs(["Train Metrics", "Test Metrics", "Overall"])
        with tabs[0]:
            col_1, col_2 = st.columns([1, 1])
            color = col_1.radio(
                "Color",
                ["Single Color", "Multi Color"],
                key="colorful_summ_train",
                horizontal=True,
            )
            if color == "Multi Color":
                singlecolor = False
            else:
                singlecolor = col_2.color_picker(
                    "Pick a color for the bars", "#1f77b4", key="color_summ_train"
                )
            col1, col2 = st.columns([1, 1])
            # col.pyplot(fig)
            fig_r2 = plot_summary(
                st.session_state["summary_train"], None, "r2", colorfull=singlecolor
            )
            fig_mse = plot_summary(
                st.session_state["summary_train"], None, "mse", colorfull=singlecolor
            )
            fig_rmse = plot_summary(
                st.session_state["summary_train"], None, "rmse", colorfull=singlecolor
            )
            fig_mae = plot_summary(
                st.session_state["summary_train"], None, "mae", colorfull=singlecolor
            )

            col1.pyplot(fig_r2)
            col2.pyplot(fig_mse)
            col1.pyplot(fig_rmse)
            col2.pyplot(fig_mae)
        with tabs[1]:
            col_1, col_2 = st.columns([1, 1])
            color = col_1.radio(
                "Color",
                ["Single Color", "Multi Color"],
                key="colorful_summ_test",
                horizontal=True,
            )
            if color == "Multi Color":
                singlecolor = False
            else:
                singlecolor = col_2.color_picker(
                    "Pick a color for the bars", "#1f77b4", key="color_summ_test"
                )
            col1, col2 = st.columns([1, 1])
            # col.pyplot(fig)
            fig_r2 = plot_summary(
                st.session_state["summary_test"], None, "r2", colorfull=singlecolor
            )
            fig_mse = plot_summary(
                st.session_state["summary_test"], None, "mse", colorfull=singlecolor
            )
            fig_rmse = plot_summary(
                st.session_state["summary_test"], None, "rmse", colorfull=singlecolor
            )
            fig_mae = plot_summary(
                st.session_state["summary_test"], None, "mae", colorfull=singlecolor
            )

            col1.pyplot(fig_r2)
            col2.pyplot(fig_mse)
            col1.pyplot(fig_rmse)
            col2.pyplot(fig_mae)
        with tabs[2]:
            col1, col2 = st.columns([1, 1])
            # col.pyplot(fig)
            fig_r2 = plot_summary(
                st.session_state["summary_train"],
                st.session_state["summary_test"],
                "r2",
            )
            fig_mse = plot_summary(
                st.session_state["summary_train"],
                st.session_state["summary_test"],
                "mse",
            )
            fig_rmse = plot_summary(
                st.session_state["summary_train"],
                st.session_state["summary_test"],
                "rmse",
            )
            fig_mae = plot_summary(
                st.session_state["summary_train"],
                st.session_state["summary_test"],
                "mae",
            )

            col1.pyplot(fig_r2)
            col2.pyplot(fig_mse)
            col1.pyplot(fig_rmse)
            col2.pyplot(fig_mae)

        with st.expander("AI Explanation", False, key="expand_summ"):
            if st.button("Generate Text", key="button_summ"):
                with st.spinner("In progress...", show_time=True):
                    resp_sum = ai_support(
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
