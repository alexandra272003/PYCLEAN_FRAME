"""
cleanframe/decorators.py
The 5 decorators: @clean, @validate, @audit, @profile, @pipeline
"""
from __future__ import annotations

import inspect
import warnings
import functools
import pandas as pd

from .schema import Schema
from .engine import apply_schema
from .result import CleanResult
from .utils import collect_issues
from .exceptions import DataQualityError


# ── Helper: find the DataFrame argument ───────────────────────────────────────

def _get_df_arg(func, args, kwargs) -> tuple[pd.DataFrame, str | int]:
    """
    Find the pd.DataFrame argument from args/kwargs.
    Returns (dataframe, key) where key is the kwarg name or positional index.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # Check kwargs first
    for k, v in kwargs.items():
        if isinstance(v, pd.DataFrame):
            return v, k

    # Check positional args
    for i, v in enumerate(args):
        if isinstance(v, pd.DataFrame):
            return v, i

    raise TypeError(
        f"cleanframe: could not find a pd.DataFrame argument in call to "
        f"'{func.__name__}'. Positional params: {params}"
    )


def _inject_df(args, kwargs, key, new_df) -> tuple[tuple, dict]:
    """Replace the DataFrame in args/kwargs with new_df."""
    if isinstance(key, str):
        kwargs = {**kwargs, key: new_df}
    else:
        args = list(args)
        args[key] = new_df
        args = tuple(args)
    return args, kwargs


# ── @clean ────────────────────────────────────────────────────────────────────

def clean(schema: Schema, mode: str = "fix"):
    """
    Decorator that cleans the DataFrame before the function runs.

    Parameters
    ----------
    schema : Schema
        The cleaning contract (Col rules per column).
    mode : str
        "fix"   — clean silently (default)
        "warn"  — clean and emit UserWarning per change
        "raise" — raise DataQualityError instead of cleaning

    Returns
    -------
    CleanResult
        .df           — cleaned DataFrame
        .explain()    — human-readable report
        .to_code()    — equivalent pandas code
        .return_value — what your function returned

    Examples
    --------
    >>> schema = Schema(age=Col(null="median", clip=(0, 120)))
    >>> @clean(schema)
    ... def train(df):
    ...     model.fit(df)
    >>> result = train(dirty_df)
    >>> print(result.explain())
    """
    if not isinstance(schema, Schema):
        raise TypeError(
            f"@clean expects a Schema instance, got {type(schema).__name__}. "
            f"Usage: @clean(Schema(col=Col(...)))"
        )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            df, key = _get_df_arg(func, args, kwargs)
            cleaned_df, steps = apply_schema(df, schema, mode=mode)
            args, kwargs = _inject_df(args, kwargs, key, cleaned_df)
            return_value = func(*args, **kwargs)
            return CleanResult(
                df=cleaned_df,
                steps=steps,
                return_value=return_value,
                func_name=func.__name__,
            )

        wrapper._cleanframe_decorator = "clean"
        wrapper._cleanframe_schema = schema
        return wrapper

    return decorator


# ── @validate ────────────────────────────────────────────────────────────────

def validate(schema: Schema):
    """
    Decorator that validates the DataFrame and raises DataQualityError if dirty.
    Never modifies data. Use in production APIs / critical pipelines.

    Parameters
    ----------
    schema : Schema
        The validation contract.

    Raises
    ------
    DataQualityError
        If any null or out-of-range issues are found.

    Examples
    --------
    >>> @validate(schema)
    ... def predict(df):
    ...     return model.predict(df)
    >>> try:
    ...     predict(dirty_df)
    ... except DataQualityError as e:
    ...     print(e.issues)
    """
    if not isinstance(schema, Schema):
        raise TypeError(
            f"@validate expects a Schema instance, got {type(schema).__name__}."
        )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            df, _ = _get_df_arg(func, args, kwargs)
            issues = collect_issues(df, schema)
            if issues:
                raise DataQualityError(issues)
            return func(*args, **kwargs)

        wrapper._cleanframe_decorator = "validate"
        wrapper._cleanframe_schema = schema
        return wrapper

    return decorator


# ── @audit ────────────────────────────────────────────────────────────────────

def audit(schema: Schema, mode: str = "fix"):
    """
    Decorator that cleans the DataFrame AND prints a full change log.
    Ideal during development and debugging.

    Parameters
    ----------
    schema : Schema
        The cleaning contract.
    mode : str
        "fix" or "warn" (same as @clean).

    Examples
    --------
    >>> @audit(schema)
    ... def process(df):
    ...     pass
    >>> process(df)   # prints change log automatically
    """
    if not isinstance(schema, Schema):
        raise TypeError(
            f"@audit expects a Schema instance, got {type(schema).__name__}."
        )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            df, key = _get_df_arg(func, args, kwargs)
            cleaned_df, steps = apply_schema(df, schema, mode=mode)
            args, kwargs = _inject_df(args, kwargs, key, cleaned_df)
            return_value = func(*args, **kwargs)
            result = CleanResult(
                df=cleaned_df,
                steps=steps,
                return_value=return_value,
                func_name=func.__name__,
            )
            print(result.explain())
            return result

        wrapper._cleanframe_decorator = "audit"
        wrapper._cleanframe_schema = schema
        return wrapper

    return decorator


# ── @profile ─────────────────────────────────────────────────────────────────

def profile(func):
    """
    Decorator that prints a data quality report before the function runs.
    No schema needed. No cleaning. Just inspection.

    Examples
    --------
    >>> @profile
    ... def explore(df):
    ...     pass
    >>> explore(df)
    # cleanframe @profile  →  explore()
    # Shape : 1000 rows × 5 columns
    # Column     dtype    nulls  null%   note
    # ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        df, _ = _get_df_arg(func, args, kwargs)
        _print_profile(df, func.__name__)
        return func(*args, **kwargs)

    wrapper._cleanframe_decorator = "profile"
    return wrapper


def _print_profile(df: pd.DataFrame, func_name: str):
    """Print a formatted data quality report."""
    print(f"\ncleanframe @profile  →  {func_name}()")
    print(f"{'Shape':<12}: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"{'Memory':<12}: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print()

    header = f"  {'Column':<20} {'dtype':<12} {'nulls':>6}  {'null%':>6}  note"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for col in df.columns:
        series = df[col]
        nulls = int(series.isna().sum())
        null_pct = 100 * nulls / len(series) if len(series) > 0 else 0
        dtype = str(series.dtype)
        notes = []

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 2:
                try:
                    skew = clean.skew()
                    if abs(skew) > 1.0:
                        notes.append(f"skewed ({skew:+.1f})")
                except Exception:
                    pass

        if pd.api.types.is_object_dtype(series):
            n_unique = series.nunique()
            notes.append(f"{n_unique} unique")

        note_str = ", ".join(notes)
        flag = "⚠" if nulls > 0 else " "
        print(f"  {flag} {col:<20} {dtype:<12} {nulls:>6}  {null_pct:>5.1f}%  {note_str}")

    print()


# ── @pipeline ────────────────────────────────────────────────────────────────

def pipeline(*schemas: Schema, mode: str = "fix"):
    """
    Decorator that applies multiple Schemas in sequence.
    Each schema's output feeds into the next schema.

    Parameters
    ----------
    *schemas : Schema
        One or more Schema instances, applied left to right.
    mode : str
        "fix", "warn", or "raise" — applied to all schemas.

    Examples
    --------
    >>> raw_schema     = Schema(age=Col(null="median"), dept=Col(null="mode"))
    >>> feature_schema = Schema(age=Col(clip=(0, 120), dtype="float"))
    >>>
    >>> @pipeline(raw_schema, feature_schema)
    ... def train(df):
    ...     model.fit(df)
    """
    for i, s in enumerate(schemas):
        if not isinstance(s, Schema):
            raise TypeError(
                f"@pipeline expects Schema instances, got {type(s).__name__} "
                f"at position {i}."
            )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            df, key = _get_df_arg(func, args, kwargs)
            all_steps = []

            for schema in schemas:
                df, steps = apply_schema(df, schema, mode=mode)
                all_steps.extend(steps)

            args, kwargs = _inject_df(args, kwargs, key, df)
            return_value = func(*args, **kwargs)
            return CleanResult(
                df=df,
                steps=all_steps,
                return_value=return_value,
                func_name=func.__name__,
            )

        wrapper._cleanframe_decorator = "pipeline"
        wrapper._cleanframe_schemas = schemas
        return wrapper

    return decorator
