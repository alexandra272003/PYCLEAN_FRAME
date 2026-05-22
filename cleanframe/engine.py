"""
cleanframe/engine.py
The engine that applies a Schema to a DataFrame.
All pandas operations live here — the decorators are just wrappers around this.
"""
from __future__ import annotations

import warnings
import pandas as pd

from .schema import Schema, Col
from .utils import find_column, auto_strategy, collect_issues
from .exceptions import DataQualityError


def apply_schema(
    df: pd.DataFrame,
    schema: Schema,
    mode: str = "fix",
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Apply a Schema to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame. Original is never mutated.
    schema : Schema
        The cleaning contract.
    mode : str
        "fix"   — clean silently (default)
        "warn"  — clean and emit UserWarning for each change
        "raise" — raise DataQualityError if any issues exist (no cleaning)

    Returns
    -------
    (cleaned_df, steps)
        cleaned_df : pd.DataFrame — a new, cleaned DataFrame
        steps : list[dict] — structured log of every change made
    """
    if mode == "raise":
        issues = collect_issues(df, schema)
        if issues:
            raise DataQualityError(issues)
        return df.copy(), []

    result = df.copy()
    steps: list[dict] = []

    pd.set_option('future.no_silent_downcasting', True)  # ← add this

    for col_name, col_rule in schema:
        actual = find_column(result, col_name)

        # Column not found in DataFrame
        if actual is None:
            if col_rule.required:
                raise KeyError(
                    f"Required column '{col_name}' not found in DataFrame. "
                    f"Available columns: {list(result.columns)}"
                )
            steps.append({
                "action": "skip",
                "column": col_name,
                "reason": "column not found in DataFrame",
            })
            continue

        # ── 1. Null handling ────────────────────────────────────────────────
        null_count = int(result[actual].isna().sum())

        if null_count > 0 and col_rule.null != "ignore":
            strategy = col_rule.null

            # Resolve "auto"
            if strategy == "auto":
                strategy = auto_strategy(result[actual], actual)

            # Apply strategy
            if strategy == "mean":
                fill_val = result[actual].mean()
                result[actual] = result[actual].fillna(fill_val).infer_objects(copy=False)
                _record_fill(steps, actual, "mean", fill_val, null_count, mode)

            elif strategy == "median":
                fill_val = result[actual].median()
                result[actual] = result[actual].fillna(fill_val).infer_objects(copy=False)
                _record_fill(steps, actual, "median", fill_val, null_count, mode)

            elif strategy == "mode":
                mode_vals = result[actual].mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else None
                if fill_val is not None:
                    result[actual] = result[actual].fillna(fill_val).infer_objects(copy=False)
                _record_fill(steps, actual, "mode", fill_val, null_count, mode)

            elif strategy == "drop":
                before_len = len(result)
                result = result.dropna(subset=[actual]).reset_index(drop=True)
                dropped = before_len - len(result)
                step = {
                    "action": "fill_null",
                    "column": actual,
                    "strategy": "drop",
                    "rows_dropped": dropped,
                    "code": f"df = df.dropna(subset=['{actual}']).reset_index(drop=True)",
                }
                steps.append(step)
                if mode == "warn":
                    warnings.warn(
                        f"cleanframe [@clean]: dropped {dropped} rows with null '{actual}'",
                        UserWarning, stacklevel=6,
                    )

            elif strategy == "ffill":
                result[actual] = result[actual].ffill().infer_objects(copy=False)
                _record_fill(steps, actual, "ffill", None, null_count, mode)

            elif strategy == "bfill":
                result[actual] = result[actual].bfill().infer_objects(copy=False)
                _record_fill(steps, actual, "bfill", None, null_count, mode)

            elif strategy == "ignore":
                pass  # intentionally skip

            else:
                # Scalar fill value
                result[actual] = result[actual].fillna(strategy)
                _record_fill(steps, actual, f"scalar({strategy!r})", strategy, null_count, mode)

        # ── 2. Clip ─────────────────────────────────────────────────────────
        if col_rule.clip is not None:
            if pd.api.types.is_numeric_dtype(result[actual]):
                lo, hi = col_rule.clip
                out_of_range = int(((result[actual] < lo) | (result[actual] > hi)).sum())
                if out_of_range > 0:
                    result[actual] = result[actual].clip(lower=lo, upper=hi)
                    step = {
                        "action": "clip",
                        "column": actual,
                        "clip_min": lo,
                        "clip_max": hi,
                        "before": out_of_range,
                        "code": f"df['{actual}'] = df['{actual}'].clip(lower={lo}, upper={hi})",
                    }
                    steps.append(step)
                    if mode == "warn":
                        warnings.warn(
                            f"cleanframe [@clean]: clipped {out_of_range} value(s) in '{actual}' to [{lo}, {hi}]",
                            UserWarning, stacklevel=6,
                        )
            # Non-numeric column with clip — silently skip (don't raise)

        # ── 3. Dtype coercion ────────────────────────────────────────────────
        if col_rule.dtype is not None:
            try:
                old_dtype = str(result[actual].dtype)
                result[actual] = _coerce_dtype(result[actual], col_rule.dtype)
                new_dtype = str(result[actual].dtype)
                if old_dtype != new_dtype:
                    steps.append({
                        "action": "dtype",
                        "column": actual,
                        "from": old_dtype,
                        "to": new_dtype,
                        "code": f"df['{actual}'] = df['{actual}'].astype('{col_rule.dtype}')",
                    })
            except Exception as e:
                warnings.warn(
                    f"cleanframe: could not coerce '{actual}' to dtype '{col_rule.dtype}': {e}",
                    UserWarning, stacklevel=6,
                )

        # ── 4. Rename ────────────────────────────────────────────────────────
        if col_rule.rename is not None:
            result = result.rename(columns={actual: col_rule.rename})
            steps.append({
                "action": "rename",
                "column": actual,
                "to": col_rule.rename,
                "code": f"df = df.rename(columns={{'{actual}': '{col_rule.rename}'}})",
            })

    return result, steps


# ── Helpers ───────────────────────────────────────────────────────────────────

def _record_fill(steps, col, strategy, fill_val, null_count, mode):
    """Record a null-fill step and optionally warn."""
    if strategy in ("ffill", "bfill"):
        code = f"df['{col}'] = df['{col}'].{strategy}()"
    elif fill_val is not None:
        code = f"df['{col}'] = df['{col}'].fillna({fill_val!r})"
    else:
        code = f"# '{col}': no fill value available (e.g. all-null column)"

    step = {
        "action": "fill_null",
        "column": col,
        "strategy": strategy,
        "fill_value": fill_val,
        "nulls_filled": null_count,
        "code": code,
    }
    steps.append(step)

    if mode == "warn":
        warnings.warn(
            f"cleanframe [@clean]: filled {null_count} null(s) in '{col}' using {strategy}"
            + (f" (value={fill_val!r})" if fill_val is not None else ""),
            UserWarning, stacklevel=7,
        )


def _coerce_dtype(series: pd.Series, dtype: str) -> pd.Series:
    """Coerce a Series to the target dtype string."""
    mapping = {
        "int":      lambda s: pd.to_numeric(s, errors="coerce").astype("Int64"),
        "float":    lambda s: pd.to_numeric(s, errors="coerce").astype("float64"),
        "str":      lambda s: s.astype(str),
        "bool":     lambda s: s.astype(bool),
        "category": lambda s: s.astype("category"),
        "datetime": lambda s: pd.to_datetime(s, errors="coerce"),
    }
    return mapping[dtype](series)