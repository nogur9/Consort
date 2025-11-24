"""Streamlit entry point for the CONSORT dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from app_config import (
    CONSORT_GROUPS,
    DEFAULT_ARMS,
    MAX_WAITING_DAYS_DEFAULT,
    METRIC_TABS,
)
from data_pipeline import build_patient_dataset
from visualizations import (
    plot_cumulative_incidence_with_risk,
    plot_metric_bar,
    plot_waiting_histogram,
    summarize_by_arm,
)


st.set_page_config(page_title="CONSORT Analysis Dashboard", page_icon="📊", layout="wide")

CUSTOM_CSS = """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
@st.cache_data
def load_processed_data(data_bytes: Optional[bytes]) -> pd.DataFrame:
    """Load and cache the fully processed dataset."""
    return build_patient_dataset(data_source=data_bytes)




# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def render_sidebar_filters(
    df: pd.DataFrame,
) -> Tuple[str, int, Optional[datetime], List[str], List[str]]:
    st.sidebar.header("🔧 Filters & Settings")
    selected_group = st.sidebar.selectbox("CONSORT group", CONSORT_GROUPS, index=0)
    slider_cap = max(3650, int(MAX_WAITING_DAYS_DEFAULT))
    default_wait = min(int(MAX_WAITING_DAYS_DEFAULT), slider_cap)
    max_wait = st.sidebar.number_input(
        "Max waiting days",
        min_value=0,
        max_value=slider_cap,
        value=default_wait,
        help="Remove entries with waiting duration longer than this threshold.",
    )

    intake_date_filter = None
    if "intake_date" in df.columns and df["intake_date"].notna().any():
        intake_dates = pd.to_datetime(df["intake_date"])
        min_date = intake_dates.min()
        max_date = intake_dates.max()
        if pd.notna(min_date) and pd.notna(max_date):
            intake_date_filter = st.sidebar.date_input(
                "Minimum intake date",
                value=min_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )

    group_options = sorted(df["group"].dropna().unique().tolist())
    if df["group"].isna().any() and "Missing Group" not in group_options:
        group_options.append("Missing Group")
    if not group_options:
        group_options = DEFAULT_ARMS
    default_groups = [g for g in DEFAULT_ARMS if g in group_options]
    if not default_groups:
        default_groups = group_options
    selected_groups = st.sidebar.multiselect(
        "Groups",
        options=group_options,
        default=default_groups,
        help="Filter the analysis to a subset of groups.",
    )

    if "Clinic" in df.columns:
        clinic_options = sorted(df["Clinic"].dropna().unique().tolist())
    else:
        clinic_options = []

    selected_clinics = (
        st.sidebar.multiselect(
            "Clinics",
            options=clinic_options,
            default=clinic_options,
            help="Filter the analysis to a specific clinic.",
        )
        if clinic_options
        else []
    )

    return selected_group, max_wait, intake_date_filter, selected_groups, selected_clinics


def apply_filters(
    df: pd.DataFrame,
    selected_group: str,
    max_wait: int,
    intake_threshold,
    group_selection: List[str],
    clinic_selection: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    analysis_arms = group_selection.copy() if group_selection else DEFAULT_ARMS.copy()

    filtered = df[df["group"].isin(analysis_arms) | df["group"].isna()].copy()
    filtered["group"] = filtered["group"].fillna("Missing Group")

    if clinic_selection and "Clinic" in filtered.columns:
        filtered = filtered[filtered["Clinic"].isin(clinic_selection)]

    if group_selection:
        filtered = filtered[filtered["group"].isin(group_selection)]

    if selected_group in filtered.columns:
        filtered = filtered[filtered[selected_group]]

    if max_wait:
        filtered = filtered[~(filtered["waiting_duration"] > max_wait)]

    if intake_threshold and "intake_date" in filtered.columns:
        intake_ts = pd.to_datetime(filtered["intake_date"])
        filtered = filtered[intake_ts >= pd.to_datetime(intake_threshold)]

    if not analysis_arms:
        analysis_arms = sorted(filtered["group"].unique())

    return filtered, analysis_arms


def render_sidebar_metrics(df: pd.DataFrame, analysis_arms):
    st.sidebar.subheader("Data snapshot")
    st.sidebar.metric("Total patients", len(df))

    for arm in analysis_arms:
        count = len(df[df["group"] == arm])
        if count:
            st.sidebar.metric(arm, count)

    missing_count = len(df[df["group"] == "Missing Group"])
    if missing_count:
        st.sidebar.metric("Missing Group", missing_count)


def build_summary(df: pd.DataFrame, analysis_arms):
    summary = summarize_by_arm(df, analysis_arms)
    total_row = {
        "group": "Total",
        "count": len(df),
        "waiting_duration_mean": df["waiting_duration"].mean(skipna=True),
        "did_started_therapy": df["did_started_therapy"].sum(skipna=True),
        "suitable_for_pp": df["suitable_for_pp"].sum(skipna=True),

        "waiting_duration_median": df["waiting_duration"].median(skipna=True),
        "did_started_therapy_mean": df["did_started_therapy"].mean(skipna=True),
        "suitable_for_pp_mean": df["suitable_for_pp"].mean(skipna=True),
        "waiting_duration_std": df["waiting_duration"].std(skipna=True),

    }
    summary_display = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)
    return summary, summary_display.round(2), total_row


def render_metric_tabs(summary_df, df_filtered, totals, analysis_arms):
    base_titles = [conf["title"] for conf in METRIC_TABS]
    tab_titles = base_titles + ["Cumulative Incidence", "Waiting Histogram"]
    tabs = st.tabs(tab_titles)

    metric_tabs = tabs[: len(METRIC_TABS)]
    for conf, tab in zip(METRIC_TABS, metric_tabs):
        with tab:
            st.write(conf["description"])
            fig = plot_metric_bar(
                summary_df,
                conf["column"],
                conf["label"],
                show_total=True,
                total_value=totals.get(conf["column"]),
            )
            st.pyplot(fig)

            if conf["column"] == "waiting_duration_mean":
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Mean waiting",
                    f"{totals['waiting_duration_mean']:.1f} days"
                    if totals["waiting_duration_mean"] is not None
                    else "N/A",
                )
                col2.metric(
                    "Median waiting",
                    f"{totals['waiting_duration_median']:.1f} days"
                    if totals["waiting_duration_median"] is not None
                    else "N/A",
                )
                col3.metric(
                    "Std deviation",
                    f"{totals['waiting_duration_std']:.1f} days"
                    if totals["waiting_duration_std"] is not None
                    else "N/A",
                )

            if conf["column"] == "did_started_therapy_mean" and len(df_filtered):
                started = int(df_filtered["did_started_therapy"].sum())
                not_started = len(df_filtered) - started
                st.metric("Started therapy", f"{started} ({started/len(df_filtered)*100:.1f}%)")
                st.metric(
                    "Not started",
                    f"{not_started} ({not_started/len(df_filtered)*100:.1f}%)",
                )

            if (
                conf["column"] == "suitable_for_pp"
                and "suitable_for_pp" not in df_filtered.columns
            ):
                st.info("`suitable_for_pp` field is not available in this dataset.")

    with tabs[-2]:
        st.write("Event probability over time helps compare how quickly groups start therapy.")
        plotted, ci_fig = plot_cumulative_incidence_with_risk(df_filtered, analysis_arms)
        if plotted and ci_fig:
            st.pyplot(ci_fig)
        else:
            st.info("Not enough data to compute cumulative incidence for the selected filters.")

    with tabs[-1]:
        st.write("Waiting-time distribution per group.")
        hist_fig = plot_waiting_histogram(df_filtered, analysis_arms)
        if hist_fig:
            st.pyplot(hist_fig)
        else:
            st.info("Not enough waiting-time data to draw the histogram for the selected filters.")


def render_download_button(summary_display: pd.DataFrame, selected_group: str):
    csv_bytes = summary_display.to_csv(index=False)
    st.download_button(
        label="💾 Download summary (CSV)",
        data=csv_bytes,
        file_name=f"consort_summary_{selected_group}_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
    )


def render_raw_data_sections(df_filtered: pd.DataFrame, groups: List[str]):
    st.subheader("🗂 Raw patient records")
    ordered = groups or sorted(df_filtered["group"].unique())
    for group in ordered:
        group_df = df_filtered[df_filtered["group"] == group]
        if group_df.empty:
            continue
        st.markdown(f"**{group}** — {len(group_df)} patients")
        st.dataframe(group_df, use_container_width=True)


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
def main():

    st.markdown('<h1 class="main-header">📊 CONSORT Analysis Dashboard</h1>',
                unsafe_allow_html=True)

    # centered uploader
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        data_bytes = st.file_uploader(
            "Upload Excel file",
            type=["xlsx"],
            help="Upload a master workbook (.xlsx)."
        )

    # stop the app until file uploaded
    if data_bytes is None:
        st.info("⬆️ Please upload an Excel file to continue.")
        st.stop()

    # --- only runs AFTER upload ---

    try:
        with st.spinner("Loading and preprocessing data..."):
            df = load_processed_data(data_bytes)
    except ValueError as e:
        # Handle date validation errors
        if "Invalid date values" in str(e):
            st.error("❌ **Date Validation Error**")
            st.error(str(e))
            st.info("💡 **Tip**: Please fix the invalid date values in your Excel file and try again. Dates should be in a standard format (e.g., YYYY-MM-DD, DD/MM/YYYY) or left empty.")
            return
        else:
            # Re-raise other ValueErrors
            raise
    except Exception as e:
        st.error(f"❌ **Error loading data**: {str(e)}")
        st.info("Please check your data file and try again.")
        return

    (
        selected_group,
        max_wait,
        intake_threshold,
        group_selection,
        clinic_selection,
    ) = render_sidebar_filters(df)
    df_filtered, analysis_arms = apply_filters(
        df, selected_group, max_wait, intake_threshold, group_selection, clinic_selection
    )
    render_sidebar_metrics(df_filtered, analysis_arms)

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
        return

    summary_df, summary_display, totals = build_summary(df_filtered, analysis_arms)

    st.subheader(f"📋 Summary statistics — {selected_group}")
    st.dataframe(summary_display, use_container_width=True)

    st.subheader("📊 Visual insights")
    render_metric_tabs(summary_df, df_filtered, totals, analysis_arms)

    st.subheader("Raw data")
    render_raw_data_sections(df_filtered, analysis_arms)


    st.subheader("💾 Export")
    render_download_button(summary_display, selected_group)

if __name__ == "__main__":
    main()

