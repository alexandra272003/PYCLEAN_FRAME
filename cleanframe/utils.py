"""
cleanframe.utils
================
Column matching helpers.

Key design decisions vs your original:
  - normalize_columns() returns a NEW df — never mutates the original
  - find_column() only does exact + case-insensitive match
    NO partial match — "salary" must not silently match "salary_usd"
  - auto_strategy() decides null fill from dtype + column name patterns
"""

import pandas as pd


def find_column(df_columns, target: str):
    """
    Find a column in df by name.

    Priority order:
      1. Exact match         "age"        → "age"
      2. Case-insensitive    "Age"        → "age"
      3. Strip whitespace    " age "      → "age"

    NO partial matching — "salary" does NOT match "salary_usd".
    That was a bug in the original — silent wrong-column selection.

    Parameters
    ----------
    df_columns : Index or list
        df.columns
    target : str
        Column name declared in Schema

    Returns
    -------
    str or None
        Actual column name in df, or None if not found.
    """
    # normalise target once
    t = target.strip().lower()

    for col in df_columns:
        if col.strip().lower() == t:
            return col

    return None  # not found — no partial match, ever


def auto_strategy(col_name: str, series: pd.Series) -> str:
    """
    Choose the best null-fill strategy automatically.
    Called when Col(null="auto") is declared.

    Rules (in priority order):
      datetime-like names    → ffill  (time series)
      object / category      → mode   (most frequent)
      numeric, skew > 1.0    → median (robust to outliers)
      numeric, skew ≤ 1.0    → mean   (normal distribution)
      fallback               → median (safe default)

    Parameters
    ----------
    col_name : str
    series   : pd.Series  (may contain nulls — that's fine)

    Returns
    -------
    str — one of: "ffill" "mode" "median" "mean"
    """
    name = col_name.lower()

    # time-related column names → forward fill
    time_keywords = {"date", "time", "timestamp", "dt", "day", "month", "year"}
    if any(kw in name for kw in time_keywords):
        return "ffill"

    # categorical / text → mode
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_string_dtype(series):
        return "mode"

    # numeric — use skewness to decide
    if pd.api.types.is_numeric_dtype(series):
        try:
            skew = abs(series.dropna().skew())
            return "median" if skew > 1.0 else "mean"
        except Exception:
            return "median"

    # datetime dtype → ffill
    if pd.api.types.is_datetime64_any_dtype(series):
        return "ffill"

    return "median"  # safe fallback
