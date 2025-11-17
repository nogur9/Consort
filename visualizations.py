"""Visualization helpers for the CONSORT dashboard."""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def summarize_by_arm(subset: pd.DataFrame, arms: List[str]) -> pd.DataFrame:
    """Aggregate key metrics per treatment arm."""
    rows = []
    for arm in arms:
        arm_df = subset[subset["group"] == arm]
        if arm_df.empty:
            rows.append(
                {
                    "group": arm,
                    "count": 0,
                    "waiting_duration_mean": np.nan,
                    "waiting_duration_std": np.nan,
                    "waiting_duration_median": np.nan,
                    "did_started_therapy_mean": np.nan,
                    "suiteable_for_pp_mean": np.nan,
                }
            )
            continue

        if "count" in arm_df.columns and arm_df["count"].notna().any():
            count_sum = arm_df["count"].sum(skipna=True)
        else:
            count_sum = len(arm_df)

        suiteable_mean = (
            arm_df["suiteable_for_pp"].mean(skipna=True)
            if "suiteable_for_pp" in arm_df.columns
            else np.nan
        )

        rows.append(
            {
                "group": arm,
                "count": int(count_sum) if not pd.isna(count_sum) else 0,
                "waiting_duration_mean": arm_df["waiting_duration"].mean(skipna=True),
                "waiting_duration_std": arm_df["waiting_duration"].std(skipna=True),
                "waiting_duration_median": arm_df["waiting_duration"].median(
                    skipna=True
                ),
                "did_started_therapy_mean": arm_df["did_started_therapy"].mean(
                    skipna=True
                ),
                "suiteable_for_pp_mean": suiteable_mean,
            }
        )

    return pd.DataFrame(rows)


def plot_metric_bar(
    summary_df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    show_total: bool = True,
    total_value: Optional[float] = None,
) -> plt.Figure:
    """Create a bar chart for a given metric."""
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_df = summary_df.copy()
    if show_total and total_value is not None:
        total_row = pd.DataFrame([{"group": "Total", metric_col: total_value}])
        plot_df = pd.concat([plot_df, total_row], ignore_index=True)

    plot_df.plot(
        x="group",
        y=metric_col,
        kind="bar",
        ax=ax,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    ax.set_title(metric_label, fontsize=14, fontweight="bold")
    ax.set_xlabel("Group", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.legend().set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    return fig


def plot_cumulative_incidence_with_risk(
    subset: pd.DataFrame, arms: List[str]
) -> Tuple[bool, Optional[plt.Figure]]:
    """
    Use lifelines' automatic plotting to show cumulative incidence
    curves and an at-risk table for the provided arms.
    """
    surv_df = subset[["group", "waiting_duration", "did_started_therapy"]].copy()
    surv_df = surv_df.dropna(subset=["waiting_duration", "did_started_therapy"])
    surv_df = surv_df[~(surv_df["waiting_duration"] < 0)]

    if surv_df.empty:
        return False, None

    kmf_models = []
    fig, ax = plt.subplots(figsize=(10, 6))
    for arm in arms:
        arm_df = surv_df[surv_df["group"] == arm]
        if arm_df.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            arm_df["waiting_duration"],
            event_observed=arm_df["did_started_therapy"],
            label=arm,
        )
        kmf.plot_cumulative_density(ax=ax)
        kmf_models.append(kmf)

    if not kmf_models:
        plt.close(fig)
        return False, None

    # add_at_risk_counts(*kmf_models, ax=ax)
    ax.set_title(
        "Cumulative incidence (probability of starting therapy)", fontsize=14
    )
    ax.set_xlabel("Waiting duration (days)", fontsize=12)
    ax.set_ylabel("Probability of event", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return True, fig


def plot_waiting_histogram(
    subset: pd.DataFrame, arms: List[str], bins: int = 30
) -> Optional[plt.Figure]:
    """Plot waiting-time histogram with hue per group."""
    data = subset[subset["waiting_duration"].notna() & subset["group"].isin(arms)]
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for arm in arms:
        arm_df = data[data["group"] == arm]
        if arm_df.empty:
            continue
        ax.hist(
            arm_df["waiting_duration"],
            bins=bins,
            alpha=0.6,
            label=arm,
            edgecolor="black",
        )

    ax.set_title("Waiting time distribution by group", fontsize=14)
    ax.set_xlabel("Waiting duration (days)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

