"""
cleanframe
==========
Declarative data cleaning contracts for pandas.
Annotation-based. Zero API calls. Offline-first.

Quick start
-----------
>>> from cleanframe import clean, Col, Schema
>>>
>>> schema = Schema(
...     age    = Col(null="median",  clip=(0, 120)),
...     salary = Col(null="median",  clip=(0, 999_999)),
...     dept   = Col(null="Unknown", dtype="category"),
... )
>>>
>>> @clean(schema)
... def train_model(df):
...     model.fit(df)
...
>>> result = train_model(dirty_df)
>>> print(result.explain())
>>> print(result.to_code())
"""

from .schema import Col, Schema
from .decorators import clean, validate, audit, profile, pipeline
from .result import CleanResult
from .exceptions import DataQualityError, CleanFrameConfigError

__version__ = "1.0.0"
__author__ = "Alexandra Pratap Singh"
__license__ = "MIT"

__all__ = [
    # Schema building
    "Col",
    "Schema",
    # Decorators
    "clean",
    "validate",
    "audit",
    "profile",
    "pipeline",
    # Result
    "CleanResult",
    # Exceptions
    "DataQualityError",
    "CleanFrameConfigError",
]
