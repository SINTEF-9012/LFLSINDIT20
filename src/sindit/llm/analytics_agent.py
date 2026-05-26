"""
Analytics Agent for CNC Manufacturing Data Analysis.

Provides analytics capabilities for CNC workpiece data (OF10001–OF10005),
including machine state monitoring, axis power analysis, vibration analysis,
and LLM-powered explanations.

Data structure (3 files per workpiece):
  - TYZBPS  : General machine state (spindle, feed, position, energy, tool — 34 cols)
  - BXCZ3M  : Axis power and operation mode (Power_X1/X2/Y/Z, Operation_Status — 7 cols)
  - 7N4ZJ8  : Vibration and chatter detection (severity, harmonics, peaks — 61 cols)

Available workpieces and time ranges:
  - OF10001 : 2025-09-01 06:15 → 2025-09-02 00:22
  - OF10002 : 2025-09-02 01:10 → 2025-09-02 16:17
  - OF10003 : 2025-09-02 16:58 → 2025-09-03 08:41
  - OF10004 : 2025-10-22 00:19 → 2025-10-22 13:23
  - OF10005 : 2026-03-13 10:18 → 2026-03-16 00:54
"""

import glob
import os
import re
import sys
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_dir)

from ..processing.cnc_preprocessing import (
    CNC_DIR,
    CNC_FILE_TYPES,
    load_multiple_workpieces,
)
from .llm_image_explanation import (
    ChartImageSaver,
    LLMImageAnalyzer,
    generate_comprehensive_process_explanation,
)
from ..processing.viz import visualize_workpiece_activity


# ---------------------------------------------------------------------------
# Actual data date bounds (UTC, naive datetimes used throughout)
# ---------------------------------------------------------------------------
_DATA_START = datetime(2025, 9, 1, 6, 15, 0)   # OF10001 begin
_DATA_END = datetime(2026, 3, 16, 0, 54, 0)   # OF10005 end


class AnalyticsAgent:
    """
    Analytics Agent for CNC manufacturing data.

    Capabilities:
    1. Load and cache workpiece data (OF10001–OF10005) from parquet files.
    2. Compute machine metrics: activity, spindle, feed, power, position, vibration.
    3. Parse natural language queries for time ranges and workpiece references.
    4. Generate text summaries and LLM-powered chart explanations.

    Performance:
    - Data and metrics are cached (1 h TTL).
    - All metric calculations use vectorized pandas/NumPy operations.
    - lru_cache on pure query-parsing functions.
    """

    # ------------------------------------------------------------------
    # Pre-compiled regex patterns
    # ------------------------------------------------------------------
    _TIME_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r'last\s+(\d+)\s+minutes?',  re.IGNORECASE), "minutes"),
        (re.compile(r'last\s+(\d+)\s+hours?',    re.IGNORECASE), "hours"),
        (re.compile(r'last\s+(\d+)\s+days?',     re.IGNORECASE), "days"),
        (re.compile(r'past\s+(\d+)\s+minutes?',  re.IGNORECASE), "minutes"),
        (re.compile(r'past\s+(\d+)\s+hours?',    re.IGNORECASE), "hours"),
        (re.compile(r'past\s+(\d+)\s+days?',     re.IGNORECASE), "days"),
        (re.compile(r'(\d+)\s+minutes?\s+ago',   re.IGNORECASE), "minutes"),
        (re.compile(r'(\d+)\s+hours?\s+ago',     re.IGNORECASE), "hours"),
        (re.compile(r'(\d+)\s+days?\s+ago',      re.IGNORECASE), "days"),
    ]

    _DATE_PATTERNS: List[re.Pattern] = [
        re.compile(r'(\d{1,2})/(\d{1,2})/(\d{4})'),
        re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})'),
        re.compile(r'(\d{1,2})-(\d{1,2})-(\d{4})'),
        re.compile(r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)\s+(\d{4})', re.IGNORECASE),
        re.compile(r'([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)\s+(\d{4})',             re.IGNORECASE),
    ]

    _MONTH_MAP: Dict[str, int] = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3,    'mar': 3,
        'april':   4, 'apr': 4, 'may':      5,            'june':  6,    'jun': 6,
        'july':    7, 'jul': 7, 'august':   8, 'aug': 8,  'september': 9,'sep': 9,
        'october':10, 'oct':10, 'november':11, 'nov':11,  'december': 12,'dec':12,
    }

    # ------------------------------------------------------------------
    # Column sets for fast lookup
    # ------------------------------------------------------------------
    _POS_COLS: frozenset = frozenset([
        'Position_MCS_X', 'Position_MCS_Y', 'Position_MCS_Z',
        'Position_MCS_A', 'Position_MCS_C',
    ])
    _POWER_COLS: frozenset = frozenset([
        'Power_Active', 'Power_Apparent', 'Power_Reactive',
        'Power_Spindle', 'Power_X1', 'Power_X2', 'Power_Y', 'Power_Z',
    ])

    # ------------------------------------------------------------------
    # Query classification keywords by file type
    # ------------------------------------------------------------------
    # TYZBPS — general machine state
    _TYZBPS_KEYWORDS: frozenset = frozenset([
        'spindle', 'feed', 'position', 'energy', 'temperature', 'tool',
        'program', 'override', 'speed', 'offset', 'head', 'angular',
        'state', 'general', 'boring', 'auto',
    ])
    # BXCZ3M — axis power and operation mode
    _BXCZ3M_KEYWORDS: frozenset = frozenset([
        'power', 'axis', 'consumption', 'load', 'mode', 'operation',
        'drive', 'x1', 'x2', 'effort',
    ])
    # 7N4ZJ8 — vibration and chatter detection
    _7N4ZJ8_KEYWORDS: frozenset = frozenset([
        'vibration', 'chatter', 'frequency', 'harmonic', 'peak',
        'amplitude', 'severity', 'detection', 'spectral', 'spectrum',
        'resonance',
    ])

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialize the Analytics Agent."""
        self.cnc_file_types = CNC_FILE_TYPES

        # Discover available workpieces from the data directory
        of_pattern = os.path.join(CNC_DIR, "*", f"*_{CNC_FILE_TYPES[0]}.parquet")
        self.available_workpieces: List[str] = sorted(set(
            os.path.basename(f).split("_")[0]
            for f in glob.glob(of_pattern)
        ))

        self.default_time_range: Tuple[datetime, datetime] = self._get_default_time_range()

        # Runtime data stores
        self.loaded_data: Dict[str, pd.DataFrame] = {}
        self._data_cache: Dict[tuple, Dict[str, pd.DataFrame]] = {}
        self._metrics_cache: Dict[tuple, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[tuple, float] = {}

        # Per-file incremental cache: (of_id, query_type, start_str, end_str) → raw DataFrame
        # Allows reusing already-loaded files when query_types changes between calls
        self._per_file_cache: Dict[tuple, pd.DataFrame] = {}

        # Last Plotly figure generated — used by chatbot UI to display the chart
        self.last_figure = None

        # Chart output directory
        self.charts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "generated_charts",
        )
        os.makedirs(self.charts_dir, exist_ok=True)

        # Lazy-loaded LLM/chart utilities
        self._llm_analyzer: Optional[LLMImageAnalyzer] = None
        self._chart_saver: Optional[ChartImageSaver] = None

        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @property
    def llm_analyzer(self) -> LLMImageAnalyzer:
        if self._llm_analyzer is None:
            self._llm_analyzer = LLMImageAnalyzer()
        return self._llm_analyzer

    @property
    def chart_saver(self) -> ChartImageSaver:
        if self._chart_saver is None:
            self._chart_saver = ChartImageSaver()
        return self._chart_saver

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear all in-memory caches."""
        self._data_cache.clear()
        self._metrics_cache.clear()
        self._cache_timestamps.clear()
        self.parse_time_range_from_query.cache_clear()
        self._detect_workpieces_from_query_cached.cache_clear()
        self.logger.info("All caches cleared.")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache size and lru_cache hit/miss statistics."""
        return {
            "data_cache_size":    len(self._data_cache),
            "metrics_cache_size": len(self._metrics_cache),
            "parse_time_cache":   self.parse_time_range_from_query.cache_info()._asdict(),
            "detect_wp_cache":    self._detect_workpieces_from_query_cached.cache_info()._asdict(),
        }

    # ------------------------------------------------------------------
    # Default time range
    # ------------------------------------------------------------------

    def _get_default_time_range(self) -> Tuple[datetime, datetime]:
        """Default time range: OF10001 full duration."""
        return datetime(2025, 9, 1, 6, 15, 0), datetime(2025, 9, 2, 0, 22, 0)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_available_workpieces(self) -> List[str]:
        """Return the list of available workpiece IDs (e.g. ['OF10001', ...])."""
        return self.available_workpieces

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Mapping: query_type → parquet file suffix
    # ------------------------------------------------------------------
    _QT_TO_FT: Dict[str, str] = {
        'state':     'TYZBPS',
        'power':     'BXCZ3M',
        'vibration': '7N4ZJ8',
    }

    def _load_single_file_cached(
        self,
        of_id: str,
        query_type: str,
        time_range: Tuple,
    ) -> pd.DataFrame:
        """
        Load one parquet file for a given workpiece and query_type.

        Uses a per-file cache keyed by (of_id, query_type, start, end) so that
        the same file is never read from disk twice within a session, even when
        query_types changes between calls (incremental loading).
        """
        from ..processing.cnc_preprocessing import load_and_filter_data
        tr_key = (str(time_range[0]), str(time_range[1]))
        cache_key = (of_id, query_type) + tr_key

        tr_start = tr_key[0]
        tr_end = tr_key[1]
        print(f"[AnalyticsAgent] Loading file | workpiece={of_id} | type={query_type} "
              f"| filter={tr_start} → {tr_end}")

        if cache_key in self._per_file_cache:
            print(f"[AnalyticsAgent] Cache HIT for {of_id}/{query_type}")
            return self._per_file_cache[cache_key]

        ft = self._QT_TO_FT.get(query_type)
        if ft is None:
            return pd.DataFrame()

        pattern = os.path.join(CNC_DIR, of_id, f"{of_id}_*_{ft}.parquet")
        matches = glob.glob(pattern)
        if not matches:
            self.logger.warning("File not found: %s %s", of_id, ft)
            return pd.DataFrame()

        df = load_and_filter_data(matches[0], time_range)
        print(f"Loaded: {os.path.basename(matches[0])} → {len(df)} rows")
        self._per_file_cache[cache_key] = df
        return df

    def load_workpiece_data(
        self,
        workpieces: List[str],
        query_types: List[str],
        time_range: Optional[Tuple[datetime, datetime]] = None,
        resample_interval: str = "5S",
        force_reload: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load merged CNC data for one or more workpieces with incremental per-file cache.

        Each parquet file (TYZBPS / BXCZ3M / 7N4ZJ8) is cached individually by
        (of_id, query_type, time_range). Re-calling with a different query_types
        set reuses already-cached files and only reads the new ones from disk.

        Args:
            workpieces:        List of OF identifiers, e.g. ['OF10001'].
            query_types:       File types to load: any of 'state', 'power', 'vibration'.
            time_range:        (start, end) datetimes. Defaults to OF10001 range.
            resample_interval: Pandas offset string (default '5S').
            force_reload:      Clear the per-file cache for these files before loading.

        Returns:
            Dict mapping workpiece ID → merged DataFrame.
        """
        from ..processing.cnc_preprocessing import resample_cnc_dataframe, merge_workpiece_files

        if time_range is None:
            time_range = self.default_time_range

        start_str = time_range[0].strftime('%Y-%m-%d %H:%M') if hasattr(time_range[0], 'strftime') else str(time_range[0])
        end_str = time_range[1].strftime('%Y-%m-%d %H:%M') if hasattr(time_range[1], 'strftime') else str(time_range[1])
        print(f"[AnalyticsAgent] load_workpiece_data | workpieces={workpieces} "
              f"| query_types={query_types} | time_range={start_str} → {end_str}")

        if force_reload:
            # Invalidate per-file cache for this time range
            tr_key = (str(time_range[0]), str(time_range[1]))
            stale = [k for k in self._per_file_cache if k[2:] == tr_key]
            for k in stale:
                del self._per_file_cache[k]

        data_dict: Dict[str, pd.DataFrame] = {}

        for of_id in workpieces:
            print(f"Loading workpiece: {of_id}")
            raw: Dict[str, pd.DataFrame] = {}

            for qt in query_types:
                print(f"I load {self._QT_TO_FT.get(qt, qt)}")
                df = self._load_single_file_cached(of_id, qt, time_range)
                if df is not None and not df.empty:
                    raw[qt] = df

            if not raw:
                self.logger.warning("No data found for %s", of_id)
                continue

            resampled: Dict[str, pd.DataFrame] = {}
            for qt, df in raw.items():
                print(f"Resampling {qt}...")
                resampled[qt] = resample_cnc_dataframe(df, resample_interval)

            print("Merging files...")
            merged = merge_workpiece_files(resampled)
            if merged is not None:
                # Log the actual timestamp range present in the merged data
                ts_col = merged['timestamp'] if 'timestamp' in merged.columns else None
                if ts_col is not None and not ts_col.empty:
                    actual_start = ts_col.min()
                    actual_end = ts_col.max()
                    print(f"[AnalyticsAgent] Done: {of_id} → {len(merged)} rows, "
                          f"{len(merged.columns)} columns | "
                          f"actual range: {actual_start} → {actual_end}")
                else:
                    print(f"Done: {of_id} → {len(merged)} rows, {len(merged.columns)} columns")
                data_dict[of_id] = merged

        self.loaded_data.update(data_dict)
        return data_dict

    # ------------------------------------------------------------------
    # Metrics — numeric dict (used by cache / dashboard)
    # ------------------------------------------------------------------

    def get_workpiece_metrics(self, workpiece: str) -> Dict[str, Any]:
        """
        Return a dict of computed metrics for a loaded workpiece.

        The workpiece must have been loaded first via load_workpiece_data().
        """
        if workpiece not in self.loaded_data:
            return {"error": f"No data loaded for workpiece {workpiece}"}

        data = self.loaded_data[workpiece]
        if data.empty:
            return {"error": f"Empty data for workpiece {workpiece}"}

        cache_key = (workpiece, id(data))
        if cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]

        t0 = time.time()
        try:
            metrics = self._calculate_workpiece_metrics(data)
            self._metrics_cache[cache_key] = metrics
            self.logger.debug("Metrics for %s computed in %.3fs", workpiece, time.time() - t0)
            return metrics
        except Exception as exc:
            self.logger.error("Error computing metrics for %s: %s", workpiece, exc)
            return {"error": "Failed to compute metrics"}

    def _calculate_workpiece_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute all numeric metrics for a workpiece DataFrame.

        Uses vectorized operations throughout for performance.

        Metric groups:
        - activity  : % of time the machine was running (composite mask)
        - spindle   : mean / max actual and commanded speed
        - feed      : mean / max actual feed rate (active periods only)
        - power     : mean / max / std for each power column
        - position  : mean / min / max / range for each MCS axis
        - vibration : mean / max / std of Vibration_Severity_X and _Y
        - chatter   : event count and % for X and Y axes
        """
        metrics = {}

        # ── 1. Activity mask (machine is running if any condition is true) ──
        activity_conditions = []
        for col, threshold in [
            ("Spindle_Speed_Actual", 0),
            ("Feed_Rate_Actual",     0),
            ("Power_Active",         0),
            ("Operation_Status",     0),
        ]:
            if col in data.columns:
                activity_conditions.append(data[col] > threshold)

        if activity_conditions:
            active_mask = activity_conditions[0]
            for cond in activity_conditions[1:]:
                active_mask = active_mask | cond
            metrics["active_pct"] = round(float(active_mask.mean()) * 100, 2)
            metrics["active_rows"] = int(active_mask.sum())
            metrics["total_rows"] = len(data)

        # ── 2. Spindle ──
        for col in ("Spindle_Speed_Actual", "Spindle_Speed_Commanded"):
            if col in data.columns and data[col].notna().any():
                s = data[col].agg(["mean", "max"])
                key = col.lower()
                metrics[f"{key}_mean"] = round(float(s["mean"]), 2)
                metrics[f"{key}_max"] = round(float(s["max"]),  2)

        # ── 3. Feed rate (active only to exclude idle zeros) ──
        if "Feed_Rate_Actual" in data.columns:
            active_feed = data.loc[data["Feed_Rate_Actual"] > 0, "Feed_Rate_Actual"]
            if not active_feed.empty:
                s = active_feed.agg(["mean", "max"])
                metrics["feed_rate_mean_active"] = round(float(s["mean"]), 2)
                metrics["feed_rate_max"] = round(float(s["max"]),  2)

        # ── 4. Power columns ──
        for col in self._POWER_COLS:
            if col in data.columns and data[col].notna().any():
                s = data[col].describe()
                key = col.lower()
                metrics[f"{key}_mean"] = round(float(s["mean"]), 2)
                metrics[f"{key}_max"] = round(float(s["max"]),  2)
                metrics[f"{key}_std"] = round(float(s["std"]),  2)

        # ── 5. Position axes ──
        pos_cols = [c for c in data.columns if c in self._POS_COLS]
        for col in pos_cols:
            if data[col].notna().any():
                s = data[col].agg(["mean", "min", "max"])
                key = col.lower()
                metrics[f"{key}_mean"] = round(float(s["mean"]), 2)
                metrics[f"{key}_min"] = round(float(s["min"]),  2)
                metrics[f"{key}_max"] = round(float(s["max"]),  2)
                metrics[f"{key}_range"] = round(float(s["max"] - s["min"]), 2)

        # ── 6. Vibration severity ──
        for col in ("Vibration_Severity_X", "Vibration_Severity_Y"):
            if col in data.columns and data[col].notna().any():
                s = data[col].agg(["mean", "max", "std"])
                key = col.lower()
                metrics[f"{key}_mean"] = round(float(s["mean"]), 4)
                metrics[f"{key}_max"] = round(float(s["max"]),  4)
                metrics[f"{key}_std"] = round(float(s["std"]),  4)

        # ── 7. Chatter detection ──
        for axis in ("X", "Y"):
            col = f"Chatter_Detection_OnOff_{axis}"
            if col in data.columns and data[col].notna().any():
                chatter_active = data[col] == 1
                metrics[f"chatter_{axis.lower()}_events"] = int(chatter_active.sum())
                metrics[f"chatter_{axis.lower()}_pct"] = round(float(chatter_active.mean()) * 100, 2)

        return metrics

    # ------------------------------------------------------------------
    # Summary — formatted text (used by query() and LLM)
    # ------------------------------------------------------------------

    def generate_data_summary(
        self,
        workpiece: str,
        data: pd.DataFrame,
        time_range: Tuple[datetime, datetime],
    ) -> str:
        """
        Build a human-readable text summary for the given workpiece and data.

        Returns:
            Multi-line markdown string with header and metric sections.
        """
        try:
            start, end = time_range
            duration_min = (end - start).total_seconds() / 60

            header = (
                f"**{workpiece} Analysis "
                f"({start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')})**\n\n"
                f"**Time Range:**\n"
                f"- Duration: {duration_min:.1f} min\n"
                f"- Data points: {len(data)}\n"
                f"- Resampling: 5-second intervals\n\n"
            )
            return header + self._analyze_workpiece_metrics(data)
        except Exception:
            return f"Error generating summary for {workpiece}."

    def _analyze_workpiece_metrics(self, data: pd.DataFrame) -> str:
        """
        Build the body of the text summary from computed metrics.

        Sections: machine activity, spindle & feed, power, position, vibration.
        """
        metrics = self._calculate_workpiece_metrics(data)
        lines = []

        # ── Activity ──
        if "active_pct" in metrics:
            lines.append("**Machine Activity:**")
            lines.append(
                f"- Active: {metrics['active_pct']:.1f}% "
                f"({metrics['active_rows']} / {metrics['total_rows']} samples)"
            )

        # ── Spindle & feed ──
        spindle_lines = []
        for key in ("spindle_speed_actual_mean", "spindle_speed_actual_max",
                    "spindle_speed_commanded_mean"):
            if key in metrics:
                label = key.replace("_", " ").title()
                spindle_lines.append(f"- {label}: {metrics[key]:.0f} rpm")
        for key in ("feed_rate_mean_active", "feed_rate_max"):
            if key in metrics:
                label = key.replace("_", " ").title()
                spindle_lines.append(f"- {label}: {metrics[key]:.1f} mm/min")
        if spindle_lines:
            lines.append("\n**Spindle & Feed:**")
            lines.extend(spindle_lines)

        # ── Power ──
        power_lines = []
        for col in ("power_active", "power_spindle", "power_x1", "power_x2",
                    "power_y", "power_z"):
            mean_key = f"{col}_mean"
            max_key = f"{col}_max"
            if mean_key in metrics:
                label = col.replace("_", " ").title()
                power_lines.append(
                    f"- {label}: mean {metrics[mean_key]:.2f} W, "
                    f"max {metrics.get(max_key, 0):.2f} W"
                )
        if power_lines:
            lines.append("\n**Power Consumption:**")
            lines.extend(power_lines)

        # ── Position ──
        pos_lines = []
        for axis in ("x", "y", "z", "a", "c"):
            range_key = f"position_mcs_{axis}_range"
            mean_key = f"position_mcs_{axis}_mean"
            if range_key in metrics:
                label = f"MCS {axis.upper()}"
                pos_lines.append(
                    f"- {label}: mean {metrics[mean_key]:.1f}, "
                    f"range {metrics[range_key]:.1f} mm"
                )
        if pos_lines:
            lines.append("\n**Axis Positions:**")
            lines.extend(pos_lines)

        # ── Vibration ──
        vib_lines = []
        for axis in ("x", "y"):
            mean_key = f"vibration_severity_{axis}_mean"
            max_key = f"vibration_severity_{axis}_max"
            if mean_key in metrics:
                vib_lines.append(
                    f"- Severity {axis.upper()}: mean {metrics[mean_key]:.4f}, "
                    f"max {metrics.get(max_key, 0):.4f}"
                )
        for axis in ("x", "y"):
            evt_key = f"chatter_{axis}_events"
            pct_key = f"chatter_{axis}_pct"
            if evt_key in metrics and metrics[evt_key] > 0:
                vib_lines.append(
                    f"- Chatter {axis.upper()}: {metrics[evt_key]} events "
                    f"({metrics.get(pct_key, 0):.1f}% of time)"
                )
        if vib_lines:
            lines.append("\n**Vibration & Chatter:**")
            lines.extend(vib_lines)

        return "\n".join(lines) if lines else "No metrics available."

    def generate_process_data_summary(
        self,
        workpiece: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> str:
        """Load workpiece data (if needed) and return its text summary."""
        if time_range is None:
            time_range = self.default_time_range
        if workpiece not in self.loaded_data:
            self.load_workpiece_data([workpiece], time_range)
        data = self.loaded_data.get(workpiece)
        if data is None or data.empty:
            return f"No data available for {workpiece}."
        return self.generate_data_summary(workpiece, data, time_range)

    # ------------------------------------------------------------------
    # Query entry point
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Process a natural language query about CNC workpiece data.

        Args:
            query_text: Free-text question from the user.
            time_range: Optional explicit time range; parsed from query if absent.

        Returns:
            Dict with keys: query, time_range, workpiece, query_type,
            data_summary, chart_path, llm_analysis, success.
        """
        try:
            if time_range is None:
                time_range = self.parse_time_range_from_query(query_text)

            print(f"[AnalyticsAgent] query() | resolved time_range: "
                  f"{time_range[0].strftime('%Y-%m-%d %H:%M')} → "
                  f"{time_range[1].strftime('%Y-%m-%d %H:%M')}")

            # Reject queries fully outside the available data window
            if time_range[0] > _DATA_END or time_range[1] < _DATA_START:
                start_str = time_range[0].strftime("%Y-%m-%d %H:%M")
                end_str = time_range[1].strftime("%Y-%m-%d %H:%M")
                return {
                    "error": "No data available for the requested time period.",
                    "guidance": (
                        f"The range {start_str} → {end_str} is outside the available "
                        f"data window ({_DATA_START.date()} → {_DATA_END.date()})."
                    ),
                    "query": query_text,
                    "time_range": time_range,
                    "success": False,
                }

            print("Je classifie")
            query_types = self._classify_query_types(query_text)
            print("query_types : ", query_types)
            workpieces_to_load = self._detect_workpieces_from_query(query_text)
            print("workpieces_to_load : ", workpieces_to_load)

            self.load_workpiece_data(workpieces_to_load, query_types, time_range)

            primary_wp = workpieces_to_load[0]
            primary_data = self.loaded_data.get(primary_wp)

            if primary_data is None or primary_data.empty:
                start_str = time_range[0].strftime("%Y-%m-%d %H:%M")
                end_str = time_range[1].strftime("%Y-%m-%d %H:%M")
                return {
                    "error": f"No data found for {primary_wp} in {start_str} → {end_str}.",
                    "guidance": "Try a wider time range or check the workpiece ID.",
                    "query": query_text,
                    "time_range": time_range,
                    "workpiece": primary_wp,
                    "query_types": query_types,
                    "success": False,
                }

            data_summary = self.generate_data_summary(primary_wp, primary_data, time_range)
            chart_path = self._save_chart_for_llm(primary_wp, primary_data)
            llm_analysis = self._generate_llm_analysis(
                chart_path, data_summary, query_text, primary_wp
            )

            return {
                "query": query_text,
                "time_range": time_range,
                "workpiece": primary_wp,
                "query_types": query_types,
                "data_summary": data_summary,
                "chart_path": chart_path,
                "llm_analysis": llm_analysis,
                "success": True,
            }

        except Exception as exc:
            self.logger.exception("Unhandled error in query(): %s", exc)
            return {
                "error":      "Internal error while processing the query.",
                "query":      query_text,
                "time_range": time_range,
                "success":    False,
            }

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    # General analysis keywords — when present with no specific file-type keywords,
    # load all three files for a complete picture.
    _GENERAL_KEYWORDS: frozenset = frozenset([
        'analyse', 'analyze', 'analysis', 'overview', 'summary',
        'all', 'complete', 'full', 'everything', 'global', 'overall',
        'what', 'show', 'give', 'tell', 'describe', 'explain',
    ])

    def _classify_query_types(self, query_text: str) -> str:
        """
        Classify a query into the CNC file types that must be loaded.

        Rules (evaluated in order):
        1. If specific signal keywords are found → load only those file type(s).
        2. If no specific keywords but general analysis words are found → load all 3.
        3. If nothing matches (fallback) → load all 3 (safe default for unknown queries).

        Returns:
            List containing one or more of: "state", "power", "vibration"
        """
        words = set(query_text.lower().split())
        query_types = []
        if words & self._7N4ZJ8_KEYWORDS:
            query_types.append("vibration")
        if words & self._BXCZ3M_KEYWORDS:
            query_types.append("power")
        if words & self._TYZBPS_KEYWORDS:
            query_types.append("state")

        if query_types:
            # Specific signals identified — load only what is needed
            return query_types

        # No specific signal keyword found.
        # General words like "analyse", "overview", "show me" → load everything.
        # Unknown query → also load everything (safe default).
        return ["state", "power", "vibration"]

    # ------------------------------------------------------------------
    # Workpiece detection
    # ------------------------------------------------------------------

    @lru_cache(maxsize=128)
    def _detect_workpieces_from_query_cached(self, query_text: str) -> Tuple[str, ...]:
        """Cached wrapper — returns a tuple so lru_cache can hash it."""
        return tuple(self._detect_workpieces_from_query_impl(query_text))

    def _detect_workpieces_from_query(self, query_text: str) -> List[str]:
        """Return the list of workpiece IDs relevant to the query."""
        return list(self._detect_workpieces_from_query_cached(query_text))

    def _detect_workpieces_from_query_impl(self, query_text: str) -> List[str]:
        """
        Scan the query for explicit workpiece references (e.g. 'OF10001').

        Falls back to the first available workpiece if none is mentioned.
        """
        query_upper = query_text.upper()
        found = [wp for wp in self.available_workpieces if wp.upper() in query_upper]

        if found:
            seen = set()
            return [wp for wp in found if wp not in seen and not seen.add(wp)]  # type: ignore[func-returns-value]

        return [self.available_workpieces[0]] if self.available_workpieces else []

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------

    def _save_chart_for_llm(self, workpiece: str, data: pd.DataFrame) -> Optional[str]:
        """
        Render a workpiece Plotly chart, store it as self.last_figure for the UI,
        and save it as PNG for the LLM vision pipeline.

        Returns the PNG file path, or None if saving failed.
        """
        try:
            fig = visualize_workpiece_activity(data, workpiece, "5S")
            # Always store the Plotly figure so the UI can display it directly
            self.last_figure = fig

            if fig is None:
                return None

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = os.path.join(self.charts_dir, f"{workpiece.lower()}_{ts}.png")
            if self.chart_saver.save_chart(fig, chart_path):
                return chart_path
        except Exception as exc:
            self.logger.warning("Chart generation failed for %s: %s", workpiece, exc)
        return None

    def get_interactive_chart(
        self,
        query_text: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ):
        """Return a Plotly figure for the primary workpiece in the query."""
        try:
            if time_range is None:
                time_range = self.parse_time_range_from_query(query_text)
            workpieces = self._detect_workpieces_from_query(query_text)
            self.load_workpiece_data(workpieces, time_range)
            data = self.loaded_data.get(workpieces[0])
            if data is None or data.empty:
                return None
            return visualize_workpiece_activity(data, workpieces[0], "5S")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    def _generate_llm_analysis(
        self,
        chart_path: Optional[str],
        data_summary: str,
        query_text: str,
        workpiece: str,
    ) -> str:
        """Generate an LLM explanation, using the chart image when available."""
        if chart_path and os.path.exists(chart_path):
            return generate_comprehensive_process_explanation(
                chart_path, data_summary, None
            )
        if not data_summary.strip():
            return (
                f"No activity was detected for {workpiece} in the requested period.\n"
                "Possible causes: machine was idle, data was not collected, "
                "or the time range does not overlap with available data."
            )
        return f"Analysis for query '{query_text}':\n\n{data_summary}"

    # ------------------------------------------------------------------
    # Time range parsing
    # ------------------------------------------------------------------

    @lru_cache(maxsize=64)
    def parse_time_range_from_query(self, query: str) -> Tuple[datetime, datetime]:
        """
        Extract a (start, end) datetime pair from a natural language query.

        Handles:
        - Relative: 'last 10 minutes', 'past 2 hours', '3 days ago'
        - Month-day range: 'september 1 to september 2'
        - Absolute date: '2025-09-01', '01/09/2025', '1st september 2025'
        - With optional time: 'from 06:00 to 18:00', 'at 14:30'

        Falls back to the default time range when nothing matches.
        """
        q = query.lower()

        # ── Relative time ──
        for pattern, unit in self._TIME_PATTERNS:
            m = pattern.search(q)
            if m:
                amount = int(m.group(1))
                now = datetime.now()
                delta = {
                    "minutes": timedelta(minutes=amount),
                    "hours":   timedelta(hours=amount),
                    "days":    timedelta(days=amount),
                }[unit]
                return now - delta, now

        # ── Full datetime range: "YYYY-MM-DD HH:MM to YYYY-MM-DD HH:MM" ──
        # e.g. "from 2025-10-22 06:00 to 2025-10-22 13:00"
        # Must be checked BEFORE the bare absolute-date patterns, which would
        # consume only the date part and lose the time information.
        m = re.search(
            r'(\d{4}-\d{2}-\d{2})\s+(\d{1,2}):(\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\s+(\d{1,2}):(\d{2})',
            q,
        )
        if m:
            try:
                start = datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}", "%Y-%m-%d %H:%M")
                end = datetime.strptime(f"{m.group(4)} {m.group(5)}:{m.group(6)}", "%Y-%m-%d %H:%M")
                print(f"[TimeParser] Full datetime range: {start} → {end}")
                return (start, end)
            except ValueError:
                pass

        # ── One date + time range: "YYYY-MM-DD HH:MM to HH:MM" ──
        # e.g. "from 2025-10-22 00:19 to 06:00"  (both times on the same day)
        m = re.search(
            r'(\d{4}-\d{2}-\d{2})\s+(\d{1,2}):(\d{2})\s+to\s+(\d{1,2}):(\d{2})',
            q,
        )
        if m:
            try:
                date_str = m.group(1)
                start = datetime.strptime(f"{date_str} {m.group(2)}:{m.group(3)}", "%Y-%m-%d %H:%M")
                end = datetime.strptime(f"{date_str} {m.group(4)}:{m.group(5)}", "%Y-%m-%d %H:%M")
                print(f"[TimeParser] Date + time range (same day): {start} → {end}")
                return (start, end)
            except ValueError:
                pass

        # ── Month-day range ("september 1 to september 2") ──
        m = re.search(r'(\w+)\s+(\d{1,2})\s+to\s+(\w+)\s+(\d{1,2})', q, re.IGNORECASE)
        if m:
            s_month = self._MONTH_MAP.get(m.group(1))
            e_month = self._MONTH_MAP.get(m.group(3))
            if s_month and e_month:
                try:
                    # Infer year from data range context
                    year = _DATA_START.year
                    return (
                        datetime(year, s_month, int(m.group(2)), 0,  0,  0),
                        datetime(year, e_month, int(m.group(4)), 23, 59, 59),
                    )
                except ValueError:
                    pass

        # ── Absolute date ──
        date_match = None
        date_fmt = None
        for i, pattern in enumerate(self._DATE_PATTERNS):
            m = pattern.search(q)
            if m:
                date_match, date_fmt = m, i
                break

        if date_match is None:
            # No absolute date found — but there might still be a "from HH:MM to HH:MM"
            # time-only pattern (e.g. "from 06:00 to 06:30").
            # Use the default start date as the base date and try to extract the time window.
            base_date_for_time = _DATA_START.replace(hour=0, minute=0, second=0)
            start_t, end_t = self._parse_time_in_query(q, base_date_for_time)
            if start_t and end_t:
                print(f"[TimeParser] Time-only range detected (no date in query): "
                      f"{start_t.strftime('%Y-%m-%d %H:%M')} → {end_t.strftime('%Y-%m-%d %H:%M')}")
                return (start_t, end_t)
            print(f"[TimeParser] No time pattern found — using default range: "
                  f"{self.default_time_range[0]} → {self.default_time_range[1]}")
            return self.default_time_range

        groups = date_match.groups()
        try:
            if date_fmt == 3:                        # "1st december 2025"
                day, month_name, year = int(groups[0]), groups[1].lower(), int(groups[2])
                month = self._MONTH_MAP.get(month_name)
            elif date_fmt == 4:                      # "december 1st 2025"
                month_name, day, year = groups[0].lower(), int(groups[1]), int(groups[2])
                month = self._MONTH_MAP.get(month_name)
            elif len(groups) == 3:
                if len(groups[0]) == 4:              # YYYY-MM-DD
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                else:                                # DD/MM/YYYY
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
            else:
                return self.default_time_range

            if not month:
                return self.default_time_range

            base_date = datetime(year, month, day)
        except (ValueError, TypeError):
            return self.default_time_range

        start, end = self._parse_time_in_query(q, base_date)
        return (start, end) if start and end else self.default_time_range

    def _parse_time_in_query(
        self,
        query_lower: str,
        base_date: datetime,
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Extract HH:MM bounds from a query string relative to a known base date.

        Patterns tried in order:
        1. "from HH:MM to HH:MM"
        2. "HH:MM to HH:MM"
        3. "between HH:MM and HH:MM"
        4. "at HH:MM"  → [HH:MM, HH:MM + 30 min]
        5. Any two times found → [min, max]
        6. Any one time → [time, time + 1 h]
        7. Nothing → [00:00, 23:59] on base_date
        """
        range_patterns = [
            re.compile(r"from\s+(\d{1,2}):(\d{2})\s+to\s+(\d{1,2}):(\d{2})"),
            re.compile(r"(\d{1,2}):(\d{2})\s+to\s+(\d{1,2}):(\d{2})"),
            re.compile(r"between\s+(\d{1,2}):(\d{2})\s+and\s+(\d{1,2}):(\d{2})"),
        ]
        for pat in range_patterns:
            m = pat.search(query_lower)
            if m:
                sh, sm, eh, em = map(int, m.groups())
                return (
                    base_date.replace(hour=sh, minute=sm, second=0),
                    base_date.replace(hour=eh, minute=em, second=0),
                )

        # "at HH:MM"
        m = re.search(r"at\s+(\d{1,2}):(\d{2})", query_lower)
        if m:
            h, mn = map(int, m.groups())
            start = base_date.replace(hour=h, minute=mn, second=0)
            return start, start + timedelta(minutes=30)

        # Any isolated times
        times = [
            base_date.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0)
            for m in re.finditer(r"(\d{1,2}):(\d{2})", query_lower)
        ]
        if len(times) >= 2:
            return min(times), max(times)
        if len(times) == 1:
            return times[0], times[0] + timedelta(hours=1)

        # Whole day
        return (
            base_date.replace(hour=0,  minute=0,  second=0),
            base_date.replace(hour=23, minute=59, second=59),
        )
