import logging
import os
import glob
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# Path to the CNC data directory.
# __file__ = lfl/backend/agents/processing/cnc_preprocessing.py
# Go up 3 levels to reach lfl/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
CNC_DIR = os.path.join(project_root, 'data', 'cnc')

# The 3 CSV file types for each OF
CNC_FILE_TYPES = ['TYZBPS', 'BXCZ3M', '7N4ZJ8']


# Columns that contain categorical/discrete values, resampled with last() instead of mean()
CNC_CATEGORICAL_COLUMNS = {
    'Program_Name',
    'Program_Block_Number',
    'Tool_Number',
    'Head_Angular_On',
    'Head_Auto_On',
    'Head_Boring_On',
    'Operation_Mode',
    'Operation_Status',
    'Chatter_Detection_OnOff_X',
    'Chatter_Detection_OnOff_Y',
}


def convert_all_cnc_data():
    for file_type in CNC_FILE_TYPES:
        of_pattern = os.path.join(CNC_DIR, "*", f"*_{file_type}.csv")
        for f in glob.glob(of_pattern):
            convert_csv_to_parquet(f)


def convert_csv_to_parquet(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df = df.sort_values('timestamp')

    out_path = file_path.replace('.csv', '.parquet')
    df.to_parquet(out_path, index=False)
    logging.info(f"Converted: {out_path}")


def load_and_filter_data(file_path, time_range):
    """
    Load a CNC Parquet file and filter rows by time range.

    Uses PyArrow predicate pushdown to filter at the row-group level before
    loading into memory — much faster than reading the full file when the
    time range covers only a fraction of the data.

    Parameters:
        file_path  : absolute path to the .parquet file
        time_range : tuple (start, end) as ISO strings or datetime objects

    Returns:
        Filtered DataFrame, or empty DataFrame if file not found
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()

    # Prepare time bounds with UTC timezone
    start_time = pd.to_datetime(time_range[0])
    end_time = pd.to_datetime(time_range[1])
    if start_time.tz is None:
        start_time = start_time.tz_localize('UTC')
    if end_time.tz is None:
        end_time = end_time.tz_localize('UTC')

    try:
        # PyArrow predicate pushdown: skip row groups outside the time window
        # without loading them into memory. Requires the parquet file to have
        # row-group statistics (which pandas write_parquet produces by default).
        table = pq.read_table(
            file_path,
            filters=[
                ('timestamp', '>=', start_time),
                ('timestamp', '<=', end_time),
            ],
        )
        return table.to_pandas()
    except Exception:
        # Fallback: read everything and filter in pandas
        df = pd.read_parquet(file_path)
        return df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]


def load_requested_workpiece_data(of_id, query_types, time_range):
    """
    Load the 3 Parquet files for a given Work Order.

    Parameters:
        of_id      : OF identifier, e.g. "OF10001"
        time_range : tuple (start, end) as ISO strings or datetime objects

    Returns:
        dict { 'TYZBPS': DataFrame, 'BXCZ3M': DataFrame, '7N4ZJ8': DataFrame }
        or None if no files were found at all

    Example:
        data = load_requested_workpiece_data(
            "OF10001",
            ("2025-09-01T06:00:00", "2025-09-01T18:00:00")
        )
    """
    data = {}
    logging.info("I load the requested workpiece data, only this query types files :")
    logging.info(query_types)
    if query_types:
        query_types_to_file_types = {
            'state': 'TYZBPS', 'power': 'BXCZ3M', 'vibration':'7N4ZJ8'
        }
    
        for file_type in query_types:
            corresponding_file_type = query_types_to_file_types[file_type]
            logging.info(f"I load {corresponding_file_type}")
            # Files are stored in a subfolder named after the OF
            # e.g. data/cnc/OF10001/OF10001_G_BQC_S8CF2G_TYZBPS.parquet
            pattern = os.path.join(CNC_DIR, of_id, f"{of_id}_*_{corresponding_file_type}.parquet")
            matches = glob.glob(pattern)

            if not matches:
                logging.error(f"File not found: OF={of_id}, type={corresponding_file_type}")
                continue

            file_path = matches[0]
            df = load_and_filter_data(file_path, time_range)
            data[file_type] = df
            logging.info(f"Loaded: {os.path.basename(file_path)} → {len(df)} rows")

    return data if data else None


def resample_cnc_dataframe(df, resample_interval='5s'):
    """
    Resample a multi-column CNC DataFrame.
      - Numeric columns     → mean
      - Categorical columns → last value in the interval

    Parameters:
        df                : DataFrame with a 'timestamp' column
        resample_interval : pandas offset string, e.g. '5S', '1min'

    Returns:
        Resampled DataFrame with 'timestamp' as a regular column,
        or None if input is empty/invalid
    """
    if df is None or df.empty or 'timestamp' not in df.columns:
        return None

    df_indexed = df.set_index('timestamp').sort_index()

    numeric_cols  = [c for c in df_indexed.columns if c not in CNC_CATEGORICAL_COLUMNS]
    categorical_cols = [c for c in df_indexed.columns if c in CNC_CATEGORICAL_COLUMNS]

    resampled_parts = []

    if numeric_cols:
        resampled_parts.append(df_indexed[numeric_cols].resample(resample_interval).mean())

    if categorical_cols:
        resampled_parts.append(df_indexed[categorical_cols].resample(resample_interval).last())

    if not resampled_parts:
        return None

    result = pd.concat(resampled_parts, axis=1).reset_index()
    return result


def merge_workpiece_files(data):
    """
    Merge the 3 resampled CNC DataFrames (TYZBPS, BXCZ3M, 7N4ZJ8) on timestamp.

    Parameters:
        data : dict { 'TYZBPS': df, 'BXCZ3M': df, '7N4ZJ8': df }

    Returns:
        Single merged DataFrame, or None if no valid DataFrames were provided
    """
    dfs = {k: v for k, v in data.items() if v is not None and not v.empty}

    if not dfs:
        return None

    merged = dfs[list(dfs.keys())[0]]
    for key in list(dfs.keys())[1:]:
        merged = merged.merge(dfs[key], on='timestamp', how='outer')

    merged = merged.sort_values('timestamp').reset_index(drop=True)
    return merged


def load_resampled_workpiece(of_id, query_types, time_range, resample_interval='5s'):
    """
    Full pipeline for one OF: load → resample → merge.

    Parameters:
        of_id             : e.g. "OF10001"
        time_range        : tuple (start, end) as ISO strings or datetime objects
        resample_interval : pandas offset string, e.g. '5S', '1min'

    Returns:
        Single merged DataFrame with all CNC signals aligned on timestamp,
        or None if loading failed

    Example:
        df = load_resampled_workpiece(
            "OF10001",
            ("2025-09-01T06:00:00", "2025-09-02T00:00:00"),
            resample_interval="5S"
        )
    """
    logging.info(f"Loading workpiece: {of_id}")
    raw_data = load_requested_workpiece_data(of_id, query_types, time_range)

    if not raw_data:
        logging.warning(f"No data found for {of_id}")
        return None

    resampled_data = {}
    for file_type, df in raw_data.items():
        logging.info(f"Resampling {file_type}...")
        resampled_data[file_type] = resample_cnc_dataframe(df, resample_interval)

    logging.info("Merging files...")
    merged = merge_workpiece_files(resampled_data)

    if merged is not None:
        logging.info(f"Done: {of_id} → {len(merged)} rows, {len(merged.columns)} columns")

    return merged


def load_multiple_workpieces(of_ids, query_types, time_range, resample_interval='5s'):
    """
    Load and process multiple OFs.

    Parameters:
        of_ids            : list of OF identifiers, e.g. ["OF10001", "OF10005"]
        time_range        : tuple (start, end) as ISO strings or datetime objects
        resample_interval : pandas offset string, e.g. '5S', '1min'

    Returns:
        dict { "OF10001": DataFrame, "OF10005": DataFrame, ... }

    Example:
        workpieces = load_multiple_workpieces(
            ["OF10001", "OF10005"],
            ("2025-09-01T00:00:00", "2025-09-02T00:00:00")
        )
    """
    results = {}
    for of_id in of_ids:
        df = load_resampled_workpiece(of_id, query_types, time_range, resample_interval)
        if df is not None:
            results[of_id] = df
    return results
