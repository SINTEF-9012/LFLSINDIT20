"""
CNC Machine Data Visualization Dashboard

A Streamlit app for visualizing CNC workpiece data with interactive charts.
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Add sindit/src/ to sys.path so that "sindit" package is importable
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.normpath(_src_dir))

from sindit.processing.cnc_preprocessing import (
    load_multiple_workpieces,
    CNC_FILE_TYPES,
    CNC_DIR,
)
import glob


@st.cache_data(show_spinner=False)
def cached_load_workpieces(of_ids_str, start_str, end_str, resample_interval):
    """
    Streamlit cache wrapper around load_multiple_workpieces.
    of_ids_str: comma-separated OF identifiers, e.g. "OF10001,OF10005"

    Always loads all three file types (state + power + vibration) so the
    standalone viz dashboard has access to all signals.
    """
    time_range = [start_str, end_str]
    of_ids = of_ids_str.split(",")
    # Load all three file types — the standalone viz always needs everything
    query_types = ['state', 'power', 'vibration']
    return load_multiple_workpieces(of_ids, query_types, time_range, resample_interval)


def visualize_workpiece_activity(workpiece_data, workpiece_name, resample_interval="5S"):
    """
    Visualize workpiece activity with position data, active-machine shading,
    tool change annotations, and program change annotations.

    Parameters:
        workpiece_data    : DataFrame with CNC signals
        workpiece_name    : str, OF identifier used in chart title
        resample_interval : str, pandas offset string used during loading
    """
    logging.info("start visualize_workpiece_activity")

    if workpiece_data is None or workpiece_data.empty:
        st.warning(f"No data available for {workpiece_name}")
        return None

    # Kaleido serializes the Plotly figure to JSON for its headless browser.
    # Both timezone-aware AND plain pd.Timestamp objects cause "not JSON serializable"
    # errors. Convert the timestamp column to plain Python strings so that every
    # downstream use (traces, vlines, annotations) is already a string.
    if "timestamp" in workpiece_data.columns:
        workpiece_data = workpiece_data.copy()
        ts_col = workpiece_data["timestamp"]
        if hasattr(ts_col, "dt"):
            if ts_col.dt.tz is not None:
                ts_col = ts_col.dt.tz_convert("UTC").dt.tz_localize(None)
            workpiece_data["timestamp"] = ts_col.dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            workpiece_data["timestamp"] = ts_col.astype(str)

    position_mappings = {
        "X": "Offset_X",
        "Y": "Offset_Y",
        "Z": "Offset_Z",
    }

    fig = go.Figure()
    position_traces = []

    # Add axis offset traces
    for pos_name, col_name in position_mappings.items():
        if col_name in workpiece_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=workpiece_data["timestamp"],
                    y=workpiece_data[col_name],
                    mode="lines",
                    name=f"{pos_name} Position",
                    line=dict(width=2),
                )
            )
            position_traces.append(col_name)

    if not position_traces:
        st.warning(f"No suitable data columns found for visualization of {workpiece_name}")
        return None

    # Compute y-axis range for annotation placement
    y_values = []
    for col in position_traces:
        if col in workpiece_data.columns:
            y_values.extend(workpiece_data[col].dropna().tolist())
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
    else:
        y_min, y_max = 0, 1

    CNC_ACTIVITY_COLUMNS = {
        "Spindle_Speed_Actual": {"color": "#1f77b4", "name": "Spindle Speed (rpm)"},
        "Feed_Rate_Actual":     {"color": "#ff7f0e", "name": "Actual Feed Rate (mm/min)"},
        "Power_Active":         {"color": "#2ca02c", "name": "Active Power (W)"},
        "Vibration_Severity_X": {"color": "#d62728", "name": "Vibration Severity X"},
        "Vibration_Severity_Y": {"color": "#9467bd", "name": "Vibration Severity Y"},
    }

    # Translucent red fill when machine is active (spindle or feed > 0).
    # Single fill trace is much faster than thousands of vrects.
    active_cols = [c for c in ["Spindle_Speed_Actual", "Feed_Rate_Actual"] if c in workpiece_data.columns]
    if active_cols:
        active_mask = workpiece_data[active_cols].gt(0).any(axis=1).astype(float)
        active_mask[active_mask == 0] = float("nan")
        fig.add_trace(
            go.Scatter(
                x=workpiece_data["timestamp"],
                y=active_mask * y_max,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(255, 100, 100, 0.12)",
                line=dict(width=0),
                name="Machine Active",
                showlegend=True,
            )
        )

    # Plot each CNC signal as a line trace
    for col, config in CNC_ACTIVITY_COLUMNS.items():
        if col in workpiece_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=workpiece_data["timestamp"],
                    y=workpiece_data[col],
                    mode="lines",
                    name=config["name"],
                    line=dict(color=config["color"], width=1.5),
                )
            )
            position_traces.append(col)

    # Tool change annotations
    if "Tool_Number" in workpiece_data.columns:
        # .diff() returns NaN on the first row, so drop it before filtering
        tool_changes = workpiece_data[workpiece_data["Tool_Number"].diff() != 0].copy()
        tool_changes = tool_changes.dropna(subset=["Tool_Number"])

        if not tool_changes.empty:
            for idx, row in tool_changes.iterrows():
                new_tool = row["Tool_Number"]
                time_point = row["timestamp"]

                prev_idx = idx - 1
                try:
                    new_tool_str = f"Tool{int(new_tool)}"
                    if prev_idx in workpiece_data.index:
                        old_tool = workpiece_data.loc[prev_idx, "Tool_Number"]
                        old_tool_str = f"Tool{int(old_tool)}" if pd.notna(old_tool) else "?"
                        label = f"{old_tool_str} → {new_tool_str}"
                    else:
                        label = new_tool_str
                except (ValueError, TypeError):
                    # new_tool is NaN or not convertible — skip this annotation
                    continue

                fig.add_annotation(
                    x=time_point,
                    y=y_max * 0.9,
                    text=label,
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="green",
                    bgcolor="lightgreen",
                    bordercolor="green",
                    borderwidth=1,
                    font=dict(size=9, color="darkgreen"),
                )
                fig.add_vline(x=time_point, line_width=1, line_dash="dash", line_color="green")

    # Program change annotations
    if "Program_Name" in workpiece_data.columns:
        # Filter out NaN rows — work only on rows with a real program name
        prog_df = workpiece_data[workpiece_data["Program_Name"].notna()].copy()

        if not prog_df.empty:
            # Detect real program changes (NaN → NaN transitions are ignored)
            prog_changes = prog_df[prog_df["Program_Name"] != prog_df["Program_Name"].shift()]

            for i, (idx, row) in enumerate(prog_changes.iterrows()):
                new_prog = row["Program_Name"]
                time_point = row["timestamp"]

                if i == 0:
                    label = f"START: {new_prog}"
                else:
                    prev_idx = prog_changes.index[i - 1]
                    old_prog = prog_df.loc[prev_idx, "Program_Name"]
                    label = f"{old_prog} → {new_prog}"

                fig.add_annotation(
                    x=time_point,
                    y=y_max * 0.8,
                    text=label,
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="blue",
                    bgcolor="lightblue",
                    bordercolor="blue",
                    borderwidth=1,
                    font=dict(size=9, color="darkblue"),
                )

    fig.update_layout(
        title=f"{workpiece_name} — Activity Analysis",
        xaxis_title="Timestamp",
        yaxis_title="Value",
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
    )
    logging.info("end visualize_workpiece_activity")

    return fig


def visualize_environmental_data(env_data):
    """
    Plot CNC temperature sensors over time.
    Each available temperature column gets its own subplot.
    """
    if env_data is None or env_data.empty:
        st.warning("No environmental data available")
        return None

    # CNC temperatures available in the TYZBPS file
    env_sensors = {
        "Temperature_Head": {"color": "#e74c3c", "name": "Head Temperature (°C)"},
        "Temperature_Room": {"color": "#3498db", "name": "Room Temperature (°C)"},
        "Temperature_Y":    {"color": "#2ecc71", "name": "Y-axis Temperature (°C)"},
        "Temperature_Z":    {"color": "#f39c12", "name": "Z-axis Temperature (°C)"},
    }

    # Keep only sensors present in the DataFrame
    available = {k: v for k, v in env_sensors.items() if k in env_data.columns}
    n_rows = len(available)

    if n_rows == 0:
        st.warning("No temperature columns found in the data.")
        return None

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[cfg["name"] for cfg in available.values()],
    )

    # Dynamically assign row numbers from 1 to n_rows
    # (hardcoded 1-4 would break if some sensors are missing)
    for row_idx, (sensor, config) in enumerate(available.items(), start=1):
        fig.add_trace(
            go.Scatter(
                x=env_data["timestamp"],
                y=env_data[sensor],
                mode="lines",
                name=config["name"],
                line=dict(color=config["color"], width=2),
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text="°C", row=row_idx, col=1)

    fig.update_layout(
        title="CNC Temperatures — Time Series",
        height=250 * n_rows,
        showlegend=False,
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
    )
    fig.update_xaxes(title_text="Timestamp", row=n_rows, col=1)

    return fig


def display_workpiece_overview_summary(workpiece_data):
    """Display CNC summary statistics for one workpiece (OF)."""
    if workpiece_data is None or workpiece_data.empty:
        return

    with st.expander("Workpiece Summary"):
        col1, col2, col3, col4 = st.columns(4)

        # Percentage of time spindle is active (Spindle_Speed_Actual > 0)
        if "Spindle_Speed_Actual" in workpiece_data.columns:
            total = workpiece_data["Spindle_Speed_Actual"].notna().sum()
            active = (workpiece_data["Spindle_Speed_Actual"] > 0).sum()
            pct = (active / total * 100) if total > 0 else 0
            with col1:
                st.metric("Spindle Active", f"{pct:.1f}%")
                st.caption(f"{active}/{total} intervals")

        # Average spindle speed during active cutting
        if "Spindle_Speed_Actual" in workpiece_data.columns:
            mean_speed = workpiece_data.loc[
                workpiece_data["Spindle_Speed_Actual"] > 0, "Spindle_Speed_Actual"
            ].mean()
            with col2:
                st.metric(
                    "Avg. Spindle Speed",
                    f"{mean_speed:.0f} rpm" if not pd.isna(mean_speed) else "—",
                )

        # Number of tool changes
        if "Tool_Number" in workpiece_data.columns:
            tool_changes = workpiece_data["Tool_Number"].diff().ne(0).sum()
            with col3:
                st.metric("Tool Changes", int(tool_changes))

        # Percentage of time chatter was detected
        if "Chatter_Detection_OnOff_X" in workpiece_data.columns:
            total = workpiece_data["Chatter_Detection_OnOff_X"].notna().sum()
            chatter = workpiece_data["Chatter_Detection_OnOff_X"].eq(True).sum()
            pct_chatter = (chatter / total * 100) if total > 0 else 0
            with col4:
                st.metric("Chatter Detected", f"{pct_chatter:.1f}%")
                st.caption(f"{chatter} intervals")


def main():
    st.title("CNC Machine Dashboard")
    st.markdown("Interactive visualization of CNC workpiece data and temperature sensors")

    st.sidebar.header("Configuration")

    # Time range selection
    st.sidebar.subheader("Time Range")
    default_start_date = datetime(2025, 9, 1).date()
    default_start_time = datetime(2025, 9, 1, 6, 0, 0).time()
    default_end_date = datetime(2025, 9, 2).date()
    default_end_time = datetime(2025, 9, 2, 0, 0, 0).time()

    start_date = st.sidebar.date_input("Start Date", value=default_start_date)
    start_time_input = st.sidebar.time_input("Start Time", value=default_start_time)
    start_time = datetime.combine(start_date, start_time_input)

    end_date = st.sidebar.date_input("End Date", value=default_end_date)
    end_time_input = st.sidebar.time_input("End Time", value=default_end_time)
    end_time = datetime.combine(end_date, end_time_input)

    resample_interval = st.sidebar.selectbox(
        "Resample Interval", options=["1s", "5s", "10s", "30s", "1min", "5min"], index=1
    )

    # Workpiece selection — scan the CNC folder for available OFs
    # OFs are stored in subdirectories: data/cnc/OF10001/OF10001_*.parquet
    st.sidebar.subheader("Workpieces")
    of_pattern = os.path.join(CNC_DIR, "*", f"*_{CNC_FILE_TYPES[0]}.parquet")
    available_workpieces = sorted(set(
        os.path.basename(f).split("_")[0]
        for f in glob.glob(of_pattern)
    ))
    if not available_workpieces:
        st.sidebar.warning(f"No OF files found in {CNC_DIR}")
        available_workpieces = ["OF10001"]

    selected_workpieces = st.sidebar.multiselect(
        "Select Workpieces to Analyze",
        options=available_workpieces,
        default=available_workpieces[:1],
    )

    if not selected_workpieces:
        st.warning("Please select at least one workpiece to analyze.")
        return

    if st.sidebar.button("Load Data", type="primary"):
        # Clear previous data before loading the new selection
        st.session_state.pop("data", None)

        with st.spinner("Loading and processing data..."):
            if end_time <= start_time:
                st.error("End time must be after Start time.")
            else:
                data = cached_load_workpieces(
                    ",".join(selected_workpieces),
                    start_time.isoformat(),
                    end_time.isoformat(),
                    resample_interval,
                )

                # Identify OFs with no data in the selected time range
                missing = [of for of in selected_workpieces if of not in data]
                if missing:
                    st.warning(f"No data for {', '.join(missing)} in the selected time range.")

                if data:
                    st.session_state["data"] = data
                    st.session_state["time_range"] = [start_time, end_time]
                    st.session_state["resample_interval"] = resample_interval
                    st.success(f"Loaded {len(data)} workpiece(s) — {start_time} → {end_time}")
                else:
                    st.error(
                        f"No data found between {start_time} and {end_time}. Check the selected time range."
                    )

    if "data" in st.session_state:
        data = st.session_state["data"]
        resample_interval = st.session_state.get("resample_interval", "5S")

        st.header("Workpiece Analysis")

        tab_names = list(data.keys())
        tabs = st.tabs(tab_names)

        for i, (workpiece, workpiece_data) in enumerate(data.items()):
            with tabs[i]:
                st.subheader(workpiece)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", len(workpiece_data))
                with col2:
                    st.metric("Columns", len(workpiece_data.columns))

                with st.expander("Data Preview"):
                    st.dataframe(workpiece_data.head())
                    csv = workpiece_data.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download full CSV",
                        data=csv,
                        file_name=f"{workpiece.lower()}_data.csv",
                        mime="text/csv",
                    )

                fig = visualize_workpiece_activity(workpiece_data, workpiece, resample_interval)
                if fig:
                    st.plotly_chart(fig, width="stretch", theme=None)

                st.subheader("Temperatures")
                fig_env = visualize_environmental_data(workpiece_data)
                if fig_env:
                    st.plotly_chart(fig_env, width="stretch", theme=None)

                display_workpiece_overview_summary(workpiece_data)

    else:
        st.info("Configure settings in the sidebar and click 'Load Data' to begin analysis.")


if __name__ == "__main__":
    main()
