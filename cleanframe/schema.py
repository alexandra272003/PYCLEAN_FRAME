"""
cleanframe/schema.py
Col() and Schema() — the building blocks of a cleaning contract.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

from .exceptions import CleanFrameConfigError

# Valid string null strategies
_NULL_STRATEGIES = frozenset({
    "mean", "median", "mode",
    "drop", "ffill", "bfill",
    "auto", "ignore",
})

# Valid dtype targets
_DTYPES = frozenset({"int", "float", "str", "bool", "category", "datetime"})


@dataclass
class Col:
    """
    Cleaning contract for a single column.

    Parameters
    ----------
    null : str or scalar, optional
        How to handle missing values.
        String strategies: "mean", "median", "mode", "drop",
                           "ffill", "bfill", "auto", "ignore"
        Scalar (int, float, str): fill with that exact value.
        Default: "ignore" (do nothing).

    clip : (min, max) tuple, optional
        Clip numeric values to this range. Skipped for non-numeric columns.

    dtype : str, optional
        Coerce column to this dtype after cleaning.
        Options: "int", "float", "str", "bool", "category", "datetime"

    rename : str, optional
        Rename the column after all other operations.

    required : bool
        If True, raise KeyError when this column is missing from the DataFrame.
        Default: False (silently skip missing columns).

    Examples
    --------
    >>> Col(null="median", clip=(0, 120), dtype="float")
    Col(null='median', clip=(0, 120), dtype='float')

    >>> Col(null="Unknown", required=True)
    Col(null='Unknown', required=True)
    """

    null: Union[str, int, float, None] = "ignore"
    clip: Optional[Tuple[float, float]] = None
    dtype: Optional[str] = None
    rename: Optional[str] = None
    required: bool = False

    def __post_init__(self):
        # Any string is valid: either a named strategy or a literal scalar fill value.
        # (e.g. Col(null="Unknown") fills nulls with the string "Unknown")

        # Validate clip
        if self.clip is not None:
            if not (isinstance(self.clip, (tuple, list)) and len(self.clip) == 2):
                raise CleanFrameConfigError("clip must be a (min, max) tuple.")
            if self.clip[0] >= self.clip[1]:
                raise CleanFrameConfigError(
                    f"clip min ({self.clip[0]}) must be less than clip max ({self.clip[1]})."
                )

        # Validate dtype
        if self.dtype is not None and self.dtype not in _DTYPES:
            raise CleanFrameConfigError(
                f"Unknown dtype '{self.dtype}'. Valid options: {sorted(_DTYPES)}"
            )

    def __repr__(self):
        parts = []
        if self.null != "ignore":
            parts.append(f"null={self.null!r}")
        if self.clip is not None:
            parts.append(f"clip={self.clip}")
        if self.dtype is not None:
            parts.append(f"dtype={self.dtype!r}")
        if self.rename is not None:
            parts.append(f"rename={self.rename!r}")
        if self.required:
            parts.append("required=True")
        return f"Col({', '.join(parts)})" if parts else "Col()"


class Schema:
    """
    Cleaning contract for a DataFrame — a collection of Col() rules.

    Parameters
    ----------
    **columns : Col
        Keyword arguments where the key is the column name and
        the value is a Col() instance.

    Examples
    --------
    >>> schema = Schema(
    ...     age    = Col(null="median", clip=(0, 120)),
    ...     salary = Col(null="median", clip=(0, 999_999)),
    ...     dept   = Col(null="Unknown", dtype="category"),
    ... )
    """

    def __init__(self, **columns):
        for name, col in columns.items():
            if not isinstance(col, Col):
                raise TypeError(
                    f"Schema expects Col() instances, got {type(col).__name__!r} for '{name}'. "
                    f"Use: Schema(age=Col(null='median')) not Schema(age='median')"
                )
        self.columns: dict[str, Col] = columns

    def __len__(self):
        return len(self.columns)

    def __repr__(self):
        cols = ", ".join(f"{k}={v!r}" for k, v in self.columns.items())
        return f"Schema({cols})"

    def __iter__(self):
        return iter(self.columns.items())
