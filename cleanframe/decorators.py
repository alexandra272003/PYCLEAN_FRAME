"""
cleanframe.decorators
=====================
The 5 decorators — the main user-facing API.

@clean(schema)          — clean df before function runs
@validate(schema)       — raise if dirty, never fix
@audit(schema)          — clean + print full change log
@profile                — print df health report, no cleaning
@pipeline(s1, s2, ...)  — chain multiple schemas in order

All decorators:
  - preserve the original function's name and docstring (functools.wraps)
  - handle both positional (func(df)) and keyword (func(df=df)) calls
  - never mutate the original DataFrame
  - return the function's own return value (never swallow it)
"""

import functools
import inspect
import pandas as pd

from .engine import apply_schema, CleanResult
from .schema import Schema


def _get_df_arg(func, args, kwargs):
    """
    Find the DataFrame argument in args/kwargs.

    Looks for the FIRST pd.DataFrame parameter by position.
    Handles both positional and keyword calls.

    Returns
    -------
    (index_or_key, df, is_positional)
      index_or_key : int (position) or str (kwarg key)
      df           : the DataFrame
      is_positional: True if found in args
    """
    param_names = list(inspect.signature(func).parameters.keys())

    # check positional args first
    for i, (name, val) in enumerate(zip(param_names, args)):
        if isinstance(val, pd.DataFrame):
            return i, val, True

    # check keyword args
    for name, val in kwargs.items():
        if isinstance(val, pd.DataFrame):
            return name, val, False

    return None, None, False


def _replace_df_arg(args, kwargs, index_or_key, new_df, is_positional):
    """
    Replace the DataFrame in args or kwargs with cleaned version.
    Returns (new_args, new_kwargs).
    """
    args = list(args)
    if is_positional:
        args[index_or_key] = new_df
    else:
        kwargs[index_or_key] = new_df
    return args, kwargs


# ── @clean ────────────────────────────────────────────────────────────────────

def clean(schema: Schema, mode: str = "fix"):
    """
    Clean the DataFrame argument before the function body runs.

    Parameters
    ----------
    schema : Schema
        Declares which columns to clean and how.
    mode : "fix" | "warn"
        fix  — clean silently (default)
        warn — clean AND print log of every change

    Returns
    -------
    The decorated function returns a CleanResult.
    Access cleaned df via result.df, explain via result.explain(),
    code via result.to_code().

    Example
    -------
        schema = Schema(
            age    = Col(null="median", clip=(0, 120)),
            salary = Col(null="median", clip=(0, 999_999)),
            dept   = Col(null="Unknown"),
        )

        @clean(schema)
        def train_model(df):
            model.fit(df)

        result = train_model(dirty_df)
        print(result.explain())
        print(result.to_code())
    """
    if not isinstance(schema, Schema):
        raise TypeError(
            f"@clean expects a Schema as first argument, got {type(schema).__name__}.\n"
            f"Usage: @clean(schema)  where schema = Schema(...)"
        )
    if mode not in ("fix", "warn"):
        raise ValueError(
            f"@clean mode must be 'fix' or 'warn', got {mode!r}.\n"
            f"Use @validate(schema) to raise on dirty data."
        )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            idx, df, is_pos = _get_df_arg(func, args, kwargs)

            if df is None:
                # no DataFrame found — call function as-is
                return func(*args, **kwargs)

            result = apply_schema(df, schema, mode=mode)
            args, kwargs = _replace_df_arg(args, kwargs, idx, result.df, is_pos)

            # call original function with cleaned df
            func(*args, **kwargs)

            # always return CleanResult so user can call .explain() / .to_code()
            return result

        return wrapper
    return decorator


# ── @validate ─────────────────────────────────────────────────────────────────

def validate(schema: Schema):
    """
    Validate the DataFrame — raise DataQualityError if dirty.
    Never fixes data. Use in production APIs where bad data = hard stop.

    Parameters
    ----------
    schema : Schema

    Raises
    ------
    DataQualityError
        If any declared column has nulls or out-of-range values.

    Example
    -------
        @validate(schema)
        def predict(df):
            return model.predict(df)

        try:
            predict(dirty_df)
        except DataQualityError as e:
            print(e.issues)  # structured list of problems
    """
    if not isinstance(schema, Schema):
        raise TypeError(
            f"@validate expects a Schema, got {type(schema).__name__}."
        )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            idx, df, is_pos = _get_df_arg(func, args, kwargs)

            if df is not None:
                # raises DataQualityError if dirty — never fixes
                apply_schema(df, schema, mode="raise")

            # data is clean — call original and return its result
            return func(*args, **kwargs)

        return wrapper
    return decorator


# ── @audit ────────────────────────────────────────────────────────────────────

def audit(schema: Schema):
    """
    Clean the df AND print a full audit log of every change made.
    Use during development, debugging, or in compliance pipelines.

    Returns CleanResult — access .steps for structured log,
    or .explain() for human-readable output.

    Example
    -------
        @audit(schema)
        def process(df):
            pass

        result = process(df)
        # prints full change log automatically
        for step in result.steps:
            print(step["column"], step["operation"])
    """
    if not isinstance(schema, Schema):
        raise TypeError(
            f"@audit expects a Schema, got {type(schema).__name__}."
        )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            idx, df, is_pos = _get_df_arg(func, args, kwargs)

            if df is None:
                return func(*args, **kwargs)

            # mode="warn" cleans AND prints the log
            result = apply_schema(df, schema, mode="warn")
            args, kwargs = _replace_df_arg(args, kwargs, idx, result.df, is_pos)
            func(*args, **kwargs)
            return result

        return wrapper
    return decorator


# ── @profile ──────────────────────────────────────────────────────────────────

def profile(func):
    """
    Print a data quality report before running the function.
    No Schema needed. No cleaning. Just shows you what's in the df.

    Use as the first step when exploring a new dataset.

    Example
    -------
        @profile
        def explore(df):
            pass

        explore(df)
        # prints: shape, dtypes, null counts, skewness, top values
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _, df, _ = _get_df_arg(func, args, kwargs)

        if df is not None:
            _print_profile(df, func.__name__)

        return func(*args, **kwargs)
    return wrapper


def _print_profile(df: pd.DataFrame, func_name: str):
    """Print a clean data quality report for df."""
    print(f"\n{'─'*50}")
    print(f"cleanframe @profile  →  {func_name}()")
    print(f"{'─'*50}")
    print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()
    print(f"{'Column':<22} {'dtype':<12} {'nulls':>6} {'null%':>7}  note")
    print("─" * 60)

    for col in df.columns:
        series   = df[col]
        nulls    = int(series.isnull().sum())
        pct      = nulls / max(len(df), 1) * 100
        dtype_s  = str(series.dtype)
        note     = ""

        if pd.api.types.is_numeric_dtype(series):
            try:
                skew = series.dropna().skew()
                if abs(skew) > 1.0:
                    note = f"skewed ({skew:+.1f})"
            except Exception:
                pass
        elif pd.api.types.is_object_dtype(series):
            n_unique = series.nunique()
            note = f"{n_unique} unique"

        print(f"  {col:<20} {dtype_s:<12} {nulls:>6} {pct:>6.1f}%  {note}")

    print(f"{'─'*50}\n")


# ── @pipeline ─────────────────────────────────────────────────────────────────

def pipeline(*schemas: Schema):
    """
    Apply multiple schemas in sequence — each step cleans progressively.
    Use for multi-stage ML pipelines where raw cleaning and feature
    engineering are separate concerns.

    Parameters
    ----------
    *schemas : Schema
        Any number of Schema objects, applied left-to-right.

    Example
    -------
        raw_schema = Schema(
            age  = Col(null="median"),
            dept = Col(null="Unknown"),
        )
        feature_schema = Schema(
            age  = Col(clip=(0, 120), dtype="float"),
            dept = Col(dtype="category"),
        )

        @pipeline(raw_schema, feature_schema)
        def train(df):
            model.fit(df)

        result = train(dirty_df)
        print(result.explain())  # shows all steps from both schemas
    """
    for i, s in enumerate(schemas):
        if not isinstance(s, Schema):
            raise TypeError(
                f"@pipeline argument {i} must be a Schema, "
                f"got {type(s).__name__}."
            )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            idx, df, is_pos = _get_df_arg(func, args, kwargs)

            if df is None:
                return func(*args, **kwargs)

            all_steps = []
            current_df = df

            for i, schema in enumerate(schemas):
                result = apply_schema(current_df, schema, mode="fix")
                current_df = result.df
                # tag each step with which pipeline stage it came from
                for step in result.steps:
                    step["pipeline_stage"] = i + 1
                all_steps.extend(result.steps)

            args, kwargs = _replace_df_arg(args, kwargs, idx, current_df, is_pos)
            func(*args, **kwargs)

            # return combined result from all stages
            return CleanResult(current_df, all_steps, mode="fix")

        return wrapper
    return decorator
