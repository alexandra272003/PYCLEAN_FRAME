"""
cleanframe
==========
Declarative data cleaning contracts for pandas DataFrames.

Quick start
-----------
    import pandas as pd
    from cleanframe import clean, validate, audit, profile, pipeline, Col, Schema

    schema = Schema(
        age    = Col(null="median",  clip=(0, 120)),
        salary = Col(null="median",  clip=(0, 999_999)),
        dept   = Col(null="Unknown", dtype="category"),
    )

    @clean(schema)
    def train_model(df):
        model.fit(df)

    result = train_model(dirty_df)
    print(result.explain())   # what changed
    print(result.to_code())   # equivalent pandas code
"""

from .schema import Col, Schema
from .decorators import clean, validate, audit, profile, pipeline
from .engine import DataQualityError, CleanResult

__version__ = "0.3.0"
__author__ = "Alexandra Pratap Singh"

__all__ = [
    "Col",
    "Schema",
    "clean",
    "validate",
    "audit",
    "profile",
    "pipeline",
    "DataQualityError",
    "CleanResult",
]
