"""
cleanframe.schema
=================
Col() and Schema() — the user-facing building blocks.

These are pure config objects. No pandas, no cleaning logic.
They only store what the user declared.

Example
-------
    Schema(
        age        = Col(null="median",  clip=(0, 120)),
        salary_usd = Col(null="median",  clip=(0, 999_999)),
        department = Col(null="Unknown", dtype="category"),
        joined_on  = Col(null="ffill",   required=True),
    )
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple


_NULL_STRATEGIES = {
    "mean", "median", "mode",
    "drop", "ffill", "bfill",
    "auto", "ignore",
}

_VALID_DTYPES = {
    "int", "int32", "int64",
    "float", "float32", "float64",
    "str", "string", "bool",
    "category", "datetime",
}


@dataclass
class Col:
    """
    Cleaning contract for ONE column.

    Parameters
    ----------
    null : str or scalar
        How to handle nulls.
        Strategies : "mean" "median" "mode" "drop" "ffill" "bfill" "auto" "ignore"
        Scalar     : any value (0, -1, "Unknown", False) used as fill directly

    clip : (min, max) tuple
        Clip values to this range. Numeric columns only.

    dtype : str
        Coerce to this dtype after cleaning.

    rename : str
        Rename column after all cleaning is done.

    required : bool
        If True and column missing from df raises KeyError.
        Default False = silently skip missing columns.

    Examples
    --------
        Col(null="median", clip=(0, 120))
        Col(null="Unknown", dtype="category")
        Col(null=0)
        Col(null="auto")
        Col(null="median", required=True, rename="age_clean")
    """

    null: Any = None
    clip: Optional[Tuple[Any, Any]] = None
    dtype: Optional[str] = None
    rename: Optional[str] = None
    required: bool = False

    def __post_init__(self):
        if self.clip is not None:
            if (
                not isinstance(self.clip, (tuple, list))
                or len(self.clip) != 2
            ):
                raise ValueError(
                    f"Col.clip must be a (min, max) tuple, got {self.clip!r}.\n"
                    f"Example: clip=(0, 120)"
                )
            self.clip = tuple(self.clip)

        if self.dtype is not None and self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Col.dtype={self.dtype!r} not recognised.\n"
                f"Supported: {sorted(_VALID_DTYPES)}"
            )


class Schema:
    """
    Groups Col() declarations for a whole DataFrame.

    Only declare columns you want cleaned.
    All other columns pass through untouched.

    Example
    -------
        Schema(
            age    = Col(null="median", clip=(0, 120)),
            salary = Col(null="median", clip=(0, 999_999)),
            dept   = Col(null="Unknown", dtype="category"),
        )
    """

    def __init__(self, **columns: Col):
        for name, rule in columns.items():
            if not isinstance(rule, Col):
                raise TypeError(
                    f"Schema: '{name}' must be a Col() instance, "
                    f"got {type(rule).__name__}.\n"
                    f"Example: Schema(age=Col(null='median'))"
                )
        self.cols = columns

    def __repr__(self):
        parts = ", ".join(
            f"{k}=Col(null={v.null!r}, clip={v.clip!r})"
            for k, v in self.cols.items()
        )
        return f"Schema({parts})"

    def column_names(self):
        return list(self.cols.keys())
