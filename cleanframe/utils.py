"""
cleanframe/utils.py
Utility helpers: column name matching, auto strategy detection.
"""
from __future__ import annotations

import re
import pandas as pd

# Column name patterns that suggest a time-series / ordered fill strategy
_DATETIME_PATTERNS = re.compile(
    r"(date|time|ts|timestamp|created|updated|at|day|month|year|period)",
    re.IGNORECASE,
)

# Column name patterns that suggest categorical data → use mode
_CATEGORICAL_PATTERNS = re.compile(
    r"(cat|category|type|kind|class|status|flag|label|tag|group|tier|"
    r"dept|department|gender|sex|country|city|region|state|code|id)",
    re.IGNORECASE,
)


def find_column(df: pd.DataFrame, name: str) -> str | None:
    """
    Find the actual column name in df that matches `name`.

    Matching rules (in priority order):
    1. Exact match
    2. Case-insensitive match (strip whitespace)
    No partial matching — 'salary' will NOT match 'salary_usd'.

    Returns the actual column name if found, else None.
    """
    # 1. Exact
    if name in df.columns:
        return name

    # 2. Case-insensitive + strip
    name_norm = name.strip().lower()
    for col in df.columns:
        if col.strip().lower() == name_norm:
            return col

    return None


def auto_strategy(series: pd.Series, col_name: str) -> str:
    """
    Choose the best null-fill strategy automatically based on:
    - Column dtype
    - Column name patterns
    - Numeric skewness

    Returns one of: "ffill", "mode", "median", "mean"
    """
    # Datetime dtype → forward fill preserves temporal order
    if pd.api.types.is_datetime64_any_dtype(series):
        return "ffill"

    # Datetime-like name → forward fill
    if _DATETIME_PATTERNS.search(col_name):
        return "ffill"

    # Object / string → mode (most frequent category)
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return "mode"

    # Categorical-sounding name → mode
    if _CATEGORICAL_PATTERNS.search(col_name):
        return "mode"

    # Numeric — check skewness
    clean = series.dropna()
    if len(clean) < 3:
        return "median"  # not enough data to assess, safe default

    try:
        skew = float(clean.skew())
    except Exception:
        return "median"

    # Skewed distribution → median is robust to outliers
    if abs(skew) > 1.0:
        return "median"

    return "mean"


def collect_issues(df: pd.DataFrame, schema) -> list[dict]:
    """
    Scan df against schema and return a list of quality issues.
    Used by @validate and mode='raise'.
    """
    issues = []
    for col_name, col_rule in schema:
        actual = find_column(df, col_name)

        if actual is None:
            if col_rule.required:
                issues.append({
                    "column": col_name,
                    "problem": "missing_required_column",
                    "detail": f"Column '{col_name}' is required but not found in DataFrame.",
                })
            continue

        series = df[actual]

        # Null check
        null_count = int(series.isna().sum())
        if null_count > 0 and col_rule.null not in ("ignore", None):
            issues.append({
                "column": actual,
                "problem": "nulls_present",
                "detail": f"{null_count} null value(s) found.",
            })

        # Clip check
        if col_rule.clip is not None and pd.api.types.is_numeric_dtype(series):
            lo, hi = col_rule.clip
            out_of_range = int(((series < lo) | (series > hi)).sum())
            if out_of_range > 0:
                issues.append({
                    "column": actual,
                    "problem": "out_of_range",
                    "detail": f"{out_of_range} value(s) outside clip range [{lo}, {hi}].",
                })

    return issues
