"""
cleanframe.engine
=================
apply_schema() — applies a Schema to a DataFrame.

Design principles:
  - Never mutates the original df (always works on df.copy())
  - Steps are tracked as structured dicts, not strings
    (to_code() generates code from step data, not by parsing log strings)
  - mode controls behaviour:
      "fix"   — clean silently (default)
      "warn"  — clean AND print what changed
      "raise" — raise DataQualityError if any issue found, never fix
  - Each column operation is independent — one column failing
    never stops other columns from being processed
"""

import pandas as pd
from .schema import Schema, _NULL_STRATEGIES
from .utils import find_column, auto_strategy


class DataQualityError(Exception):
    """
    Raised by mode="raise" when dirty data is detected.
    Contains structured issue list, not just a string.

    Attributes
    ----------
    issues : list of dicts
        Each dict: {"column": str, "problem": str, "count": int}
    """

    def __init__(self, issues: list):
        self.issues = issues
        lines = [
            f"  [{i['column']}] {i['problem']} ({i['count']} value(s))"
            for i in issues
        ]
        super().__init__(
            f"Data quality issues found ({len(issues)} problem(s)):\n"
            + "\n".join(lines)
            + "\n\nUse mode='fix' to auto-remediate, or clean your data first."
        )


class CleanResult:
    """
    Returned by apply_schema() and by every decorator.

    Attributes
    ----------
    df     : pd.DataFrame   — the cleaned dataframe
    steps  : list of dicts  — structured record of every operation
    mode   : str            — "fix" / "warn" / "raise"

    Methods
    -------
    explain()  → human-readable report of what changed
    to_code()  → equivalent pandas code as a string
    summary()  → one-line summary
    """

    def __init__(self, df: pd.DataFrame, steps: list, mode: str):
        self.df = df
        self.steps = steps
        self.mode = mode

    # ── explain ───────────────────────────────────────────────────────────

    def explain(self) -> str:
        """Human-readable report of every operation performed."""
        if not self.steps:
            return "cleanframe: no changes made (no nulls or issues found)."

        lines = [
            f"cleanframe report (mode={self.mode!r})",
            "=" * 45,
        ]
        for i, step in enumerate(self.steps, 1):
            col     = step["column"]
            op      = step["operation"]
            detail  = step.get("detail", "")
            count   = step.get("count", "")
            count_s = f" ({count} value(s))" if count else ""
            lines.append(f"{i:2}. [{col}] {op}{count_s}")
            if detail:
                lines.append(f"      → {detail}")
        return "\n".join(lines)

    # ── to_code ───────────────────────────────────────────────────────────

    def to_code(self) -> str:
        """
        Returns equivalent pandas code as a string.
        Generated from structured step data — NOT by parsing log strings.
        Paste this into your notebook to reproduce without the library.
        """
        if not self.steps:
            return "# cleanframe: no changes were needed"

        lines = [
            "import pandas as pd",
            "",
            "# cleanframe-generated cleaning code",
            "# paste this to reproduce cleaning without the library",
            "",
        ]

        for step in self.steps:
            col = step["column"]
            op  = step["operation"]
            raw = step.get("raw_col", col)  # original col name before rename

            reason = step.get("detail", "")
            if reason:
                lines.append(f"# {reason}")

            if op == "fill_mean":
                lines.append(f"df[{raw!r}] = df[{raw!r}].fillna(df[{raw!r}].mean())")

            elif op == "fill_median":
                lines.append(f"df[{raw!r}] = df[{raw!r}].fillna(df[{raw!r}].median())")

            elif op == "fill_mode":
                lines.append(f"df[{raw!r}] = df[{raw!r}].fillna(df[{raw!r}].mode()[0])")

            elif op == "fill_scalar":
                val = step.get("value")
                lines.append(f"df[{raw!r}] = df[{raw!r}].fillna({val!r})")

            elif op == "fill_ffill":
                lines.append(f"df[{raw!r}] = df[{raw!r}].ffill()")

            elif op == "fill_bfill":
                lines.append(f"df[{raw!r}] = df[{raw!r}].bfill()")

            elif op == "drop_rows":
                lines.append(f"df = df.dropna(subset=[{raw!r}])")

            elif op == "clip":
                lo, hi = step["clip"]
                lines.append(f"df[{raw!r}] = df[{raw!r}].clip({lo!r}, {hi!r})")

            elif op == "dtype":
                dtype = step["dtype"]
                lines.append(f"df[{raw!r}] = df[{raw!r}].astype({dtype!r})")

            elif op == "rename":
                new_name = step["new_name"]
                lines.append(f"df = df.rename(columns={{{raw!r}: {new_name!r}}})")

            elif op == "skipped":
                lines.append(f"# [{col}] skipped: {step.get('detail', '')}")

            lines.append("")

        return "\n".join(lines)

    def summary(self) -> str:
        """One-line summary for quick logging."""
        n = len([s for s in self.steps if s["operation"] != "skipped"])
        cols = [s["column"] for s in self.steps if s["operation"] != "skipped"]
        return f"cleanframe: {n} operation(s) on {list(dict.fromkeys(cols))}"

    def __repr__(self):
        return (
            f"<CleanResult mode={self.mode!r} "
            f"steps={len(self.steps)} "
            f"shape={self.df.shape}>"
        )


# ── apply_schema ──────────────────────────────────────────────────────────────

def apply_schema(
    df: pd.DataFrame,
    schema: Schema,
    mode: str = "fix",
) -> CleanResult:
    """
    Apply a Schema to a DataFrame.

    Parameters
    ----------
    df     : pd.DataFrame
    schema : Schema
    mode   : "fix" | "warn" | "raise"
        fix   — clean silently, return CleanResult
        warn  — clean AND print log of every change
        raise — raise DataQualityError if ANY issue found, never clean

    Returns
    -------
    CleanResult

    Raises
    ------
    DataQualityError
        Only when mode="raise" and issues are found.
    TypeError
        If df is not a pd.DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"apply_schema expects a pd.DataFrame, got {type(df).__name__}.\n"
            f"Did you pass the wrong argument?"
        )

    if mode not in ("fix", "warn", "raise"):
        raise ValueError(
            f"mode must be 'fix', 'warn', or 'raise', got {mode!r}"
        )

    # work on a copy — never mutate the original
    result_df = df.copy()
    steps = []

    # --- mode="raise": scan first, raise before touching anything ---
    if mode == "raise":
        issues = []
        for col_key, rule in schema.cols.items():
            real_col = find_column(result_df.columns, col_key)

            if real_col is None:
                if rule.required:
                    issues.append({
                        "column": col_key,
                        "problem": "required column missing",
                        "count": 1,
                    })
                continue

            null_count = int(result_df[real_col].isnull().sum())
            if null_count > 0 and rule.null not in (None, "ignore"):
                issues.append({
                    "column": real_col,
                    "problem": f"has null values (null strategy: {rule.null!r})",
                    "count": null_count,
                })

            if rule.clip is not None:
                lo, hi = rule.clip
                if pd.api.types.is_numeric_dtype(result_df[real_col]):
                    out_of_range = int(
                        ((result_df[real_col] < lo) | (result_df[real_col] > hi)).sum()
                    )
                    if out_of_range > 0:
                        issues.append({
                            "column": real_col,
                            "problem": f"has {out_of_range} values outside clip range {rule.clip}",
                            "count": out_of_range,
                        })

        if issues:
            raise DataQualityError(issues)

        return CleanResult(result_df, [], mode)

    # --- mode="fix" or "warn": clean column by column ---
    for col_key, rule in schema.cols.items():

        real_col = find_column(result_df.columns, col_key)

        # column not found
        if real_col is None:
            if rule.required:
                raise KeyError(
                    f"Required column {col_key!r} not found in DataFrame.\n"
                    f"Available columns: {list(result_df.columns)}\n"
                    f"Check for typos or case mismatches."
                )
            steps.append({
                "column":    col_key,
                "operation": "skipped",
                "detail":    "column not in DataFrame (set required=True to raise instead)",
            })
            continue

        # ── null handling ─────────────────────────────────────────────────
        null_strategy = rule.null

        if null_strategy == "auto":
            null_strategy = auto_strategy(real_col, result_df[real_col])

        null_count = int(result_df[real_col].isnull().sum())

        if null_strategy is not None and null_strategy != "ignore" and null_count > 0:

            if null_strategy == "mean":
                if not pd.api.types.is_numeric_dtype(result_df[real_col]):
                    steps.append({
                        "column":    real_col,
                        "operation": "skipped",
                        "detail":    "null='mean' requires numeric column",
                        "raw_col":   real_col,
                    })
                else:
                    val = result_df[real_col].mean()
                    result_df[real_col] = result_df[real_col].fillna(val)
                    steps.append({
                        "column":    real_col,
                        "operation": "fill_mean",
                        "detail":    f"filled with mean ({val:.4g})",
                        "count":     null_count,
                        "raw_col":   real_col,
                    })

            elif null_strategy == "median":
                if not pd.api.types.is_numeric_dtype(result_df[real_col]):
                    steps.append({
                        "column":    real_col,
                        "operation": "skipped",
                        "detail":    "null='median' requires numeric column",
                        "raw_col":   real_col,
                    })
                else:
                    val = result_df[real_col].median()
                    result_df[real_col] = result_df[real_col].fillna(val)
                    steps.append({
                        "column":    real_col,
                        "operation": "fill_median",
                        "detail":    f"filled with median ({val:.4g})",
                        "count":     null_count,
                        "raw_col":   real_col,
                    })

            elif null_strategy == "mode":
                mode_vals = result_df[real_col].mode()
                if len(mode_vals) == 0:
                    steps.append({
                        "column":    real_col,
                        "operation": "skipped",
                        "detail":    "null='mode' but column is all null",
                        "raw_col":   real_col,
                    })
                else:
                    val = mode_vals[0]
                    result_df[real_col] = result_df[real_col].fillna(val)
                    steps.append({
                        "column":    real_col,
                        "operation": "fill_mode",
                        "detail":    f"filled with mode ({val!r})",
                        "count":     null_count,
                        "raw_col":   real_col,
                    })

            elif null_strategy == "drop":
                result_df = result_df.dropna(subset=[real_col]).reset_index(drop=True)
                steps.append({
                    "column":    real_col,
                    "operation": "drop_rows",
                    "detail":    f"dropped {null_count} rows with null values",
                    "count":     null_count,
                    "raw_col":   real_col,
                })

            elif null_strategy == "ffill":
                result_df[real_col] = result_df[real_col].ffill()
                steps.append({
                    "column":    real_col,
                    "operation": "fill_ffill",
                    "detail":    "forward-filled null values",
                    "count":     null_count,
                    "raw_col":   real_col,
                })

            elif null_strategy == "bfill":
                result_df[real_col] = result_df[real_col].bfill()
                steps.append({
                    "column":    real_col,
                    "operation": "fill_bfill",
                    "detail":    "backward-filled null values",
                    "count":     null_count,
                    "raw_col":   real_col,
                })

            else:
                # scalar fill — any value including 0, "Unknown", False
                result_df[real_col] = result_df[real_col].fillna(null_strategy)
                steps.append({
                    "column":    real_col,
                    "operation": "fill_scalar",
                    "detail":    f"filled with value {null_strategy!r}",
                    "count":     null_count,
                    "value":     null_strategy,
                    "raw_col":   real_col,
                })

        # ── clip ──────────────────────────────────────────────────────────
        if rule.clip is not None:
            lo, hi = rule.clip
            if not pd.api.types.is_numeric_dtype(result_df[real_col]):
                steps.append({
                    "column":    real_col,
                    "operation": "skipped",
                    "detail":    f"clip={rule.clip} requires numeric column",
                    "raw_col":   real_col,
                })
            else:
                out_count = int(
                    ((result_df[real_col] < lo) | (result_df[real_col] > hi)).sum()
                )
                result_df[real_col] = result_df[real_col].clip(lo, hi)
                steps.append({
                    "column":    real_col,
                    "operation": "clip",
                    "detail":    f"clipped to ({lo}, {hi}) — {out_count} values affected",
                    "count":     out_count,
                    "clip":      (lo, hi),
                    "raw_col":   real_col,
                })

        # ── dtype coerce ──────────────────────────────────────────────────
        if rule.dtype is not None:
            try:
                result_df[real_col] = result_df[real_col].astype(rule.dtype)
                steps.append({
                    "column":    real_col,
                    "operation": "dtype",
                    "detail":    f"coerced to dtype={rule.dtype!r}",
                    "dtype":     rule.dtype,
                    "raw_col":   real_col,
                })
            except (ValueError, TypeError) as e:
                steps.append({
                    "column":    real_col,
                    "operation": "skipped",
                    "detail":    f"dtype coerce to {rule.dtype!r} failed: {e}",
                    "raw_col":   real_col,
                })

        # ── rename ────────────────────────────────────────────────────────
        if rule.rename is not None:
            result_df = result_df.rename(columns={real_col: rule.rename})
            steps.append({
                "column":    real_col,
                "operation": "rename",
                "detail":    f"renamed to {rule.rename!r}",
                "new_name":  rule.rename,
                "raw_col":   real_col,
            })

    # warn mode — print everything that happened
    if mode == "warn":
        cr = CleanResult(result_df, steps, mode)
        print(cr.explain())

    return CleanResult(result_df, steps, mode)
