"""
tests/test_cleanframe.py
Full test suite for cleanframe. Run with: python -m pytest tests/ -v
"""
import warnings
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleanframe import clean, validate, audit, profile, pipeline, Schema, Col, CleanResult
from cleanframe.exceptions import DataQualityError, CleanFrameConfigError
from cleanframe.utils import find_column, auto_strategy
from cleanframe.engine import apply_schema


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def basic_df():
    return pd.DataFrame({
        "age":    [25.0, None, 34.0, None, 200.0],
        "salary": [50000.0, 200000.0, None, 80000.0, -1000.0],
        "dept":   ["Eng", None, "HR", "Eng", None],
    })

@pytest.fixture
def basic_schema():
    return Schema(
        age    = Col(null="median",  clip=(0, 120)),
        salary = Col(null="median",  clip=(0, 999_999)),
        dept   = Col(null="Unknown", dtype="category"),
    )

@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "age":    [25.0, 30.0, 34.0],
        "salary": [50000.0, 80000.0, 70000.0],
        "dept":   ["Eng", "HR", "Eng"],
    })


# ─── Col() validation ─────────────────────────────────────────────────────────

class TestColValidation:
    def test_valid_strategies(self):
        for s in ["mean", "median", "mode", "drop", "ffill", "bfill", "auto", "ignore"]:
            assert Col(null=s).null == s

    def test_scalar_int_fill(self):
        assert Col(null=0).null == 0

    def test_scalar_str_fill(self):
        assert Col(null="Unknown").null == "Unknown"

    def test_scalar_float_fill(self):
        assert Col(null=-1.0).null == -1.0

    def test_invalid_strategy_raises(self):
        # Any string is valid as a scalar fill value — no strategy validation on strings
        col = Col(null="random_garbage")
        assert col.null == "random_garbage"

    def test_clip_validation(self):
        assert Col(clip=(0, 120)).clip == (0, 120)

    def test_clip_min_gte_max_raises(self):
        with pytest.raises(CleanFrameConfigError, match="clip min"):
            Col(clip=(120, 0))

    def test_clip_equal_raises(self):
        with pytest.raises(CleanFrameConfigError):
            Col(clip=(5, 5))

    def test_invalid_dtype_raises(self):
        with pytest.raises(CleanFrameConfigError, match="Unknown dtype"):
            Col(dtype="complex128")

    def test_valid_dtypes(self):
        for d in ["int", "float", "str", "bool", "category", "datetime"]:
            assert Col(dtype=d).dtype == d

    def test_repr_includes_null(self):
        assert "median" in repr(Col(null="median"))

    def test_repr_empty(self):
        assert repr(Col()) == "Col()"

    def test_required_default_false(self):
        assert Col().required is False

    def test_required_true(self):
        assert Col(required=True).required is True


# ─── Schema() validation ──────────────────────────────────────────────────────

class TestSchemaValidation:
    def test_basic_schema_creation(self):
        s = Schema(age=Col(null="median"), dept=Col(null="mode"))
        assert "age" in s.columns
        assert len(s) == 2

    def test_wrong_value_type_raises(self):
        with pytest.raises(TypeError):
            Schema(age="median")

    def test_dict_value_raises(self):
        with pytest.raises(TypeError):
            Schema(age={"null": "median"})

    def test_repr_contains_schema(self):
        assert "Schema" in repr(Schema(age=Col(null="mean")))

    def test_iterable(self):
        s = Schema(a=Col(), b=Col())
        keys = [k for k, _ in s]
        assert keys == ["a", "b"]


# ─── find_column() ────────────────────────────────────────────────────────────

class TestFindColumn:
    def test_exact_match(self):
        df = pd.DataFrame({"age": [1]})
        assert find_column(df, "age") == "age"

    def test_case_insensitive_upper(self):
        df = pd.DataFrame({"Age": [1]})
        assert find_column(df, "age") == "Age"

    def test_case_insensitive_allcaps(self):
        df = pd.DataFrame({"AGE": [1]})
        assert find_column(df, "age") == "AGE"

    def test_whitespace_stripped(self):
        df = pd.DataFrame({" age ": [1]})
        assert find_column(df, "age") == " age "

    def test_no_partial_match(self):
        # Critical: 'salary' must NOT match 'salary_usd'
        df = pd.DataFrame({"salary_usd": [1]})
        assert find_column(df, "salary") is None

    def test_missing_returns_none(self):
        df = pd.DataFrame({"dept": ["HR"]})
        assert find_column(df, "age") is None


# ─── auto_strategy() ──────────────────────────────────────────────────────────

class TestAutoStrategy:
    def test_datetime_dtype(self):
        s = pd.Series(pd.date_range("2023-01-01", periods=5))
        assert auto_strategy(s, "ts") == "ffill"

    def test_datetime_name_created_at(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert auto_strategy(s, "created_at") == "ffill"

    def test_datetime_name_date(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert auto_strategy(s, "order_date") == "ffill"

    def test_object_dtype_gives_mode(self):
        s = pd.Series(["a", "b", "c"])
        assert auto_strategy(s, "xyz") == "mode"

    def test_categorical_name_dept(self):
        s = pd.Series([1.0, 2.0])
        assert auto_strategy(s, "dept") == "mode"

    def test_skewed_numeric_gives_median(self):
        s = pd.Series([1.0] * 10 + [100000.0])
        assert auto_strategy(s, "income") == "median"

    def test_symmetric_numeric_gives_mean(self):
        s = pd.Series([10.0, 11.0, 12.0, 10.5, 11.5, 10.8, 11.2])
        result = auto_strategy(s, "value")
        assert result in ("mean", "median")

    def test_tiny_series_safe_default(self):
        s = pd.Series([1.0, 2.0])
        result = auto_strategy(s, "x")
        assert result in ("mean", "median")


# ─── apply_schema() engine ────────────────────────────────────────────────────

class TestEngine:
    def test_fill_mean(self, basic_df):
        schema = Schema(age=Col(null="mean"))
        cleaned, _ = apply_schema(basic_df, schema)
        assert cleaned["age"].isna().sum() == 0

    def test_fill_median(self, basic_df):
        schema = Schema(age=Col(null="median"))
        cleaned, _ = apply_schema(basic_df, schema)
        assert cleaned["age"].isna().sum() == 0

    def test_fill_mode(self, basic_df):
        schema = Schema(dept=Col(null="mode"))
        cleaned, _ = apply_schema(basic_df, schema)
        assert cleaned["dept"].isna().sum() == 0
        assert cleaned["dept"].iloc[1] == "Eng"  # mode of Eng, HR, Eng

    def test_fill_scalar_string(self, basic_df):
        schema = Schema(dept=Col(null="Unknown"))
        cleaned, _ = apply_schema(basic_df, schema)
        assert (cleaned["dept"] == "Unknown").sum() == 2

    def test_fill_scalar_zero(self, basic_df):
        schema = Schema(salary=Col(null=0))
        cleaned, _ = apply_schema(basic_df, schema)
        assert cleaned["salary"].isna().sum() == 0

    def test_fill_ffill(self):
        df = pd.DataFrame({"val": [1.0, None, None, 4.0]})
        cleaned, _ = apply_schema(df, Schema(val=Col(null="ffill")))
        assert cleaned["val"].tolist() == [1.0, 1.0, 1.0, 4.0]

    def test_fill_bfill(self):
        df = pd.DataFrame({"val": [None, None, 3.0, 4.0]})
        cleaned, _ = apply_schema(df, Schema(val=Col(null="bfill")))
        assert cleaned["val"].tolist() == [3.0, 3.0, 3.0, 4.0]

    def test_fill_drop(self):
        df = pd.DataFrame({"age": [25.0, None, 30.0]})
        cleaned, _ = apply_schema(df, Schema(age=Col(null="drop")))
        assert len(cleaned) == 2
        assert cleaned["age"].isna().sum() == 0

    def test_clip_upper(self, basic_df):
        schema = Schema(age=Col(clip=(0, 120)))
        cleaned, _ = apply_schema(basic_df, schema)
        assert cleaned["age"].max() <= 120

    def test_clip_lower(self, basic_df):
        schema = Schema(salary=Col(clip=(0, 999_999)))
        cleaned, _ = apply_schema(basic_df, schema)
        assert cleaned["salary"].min() >= 0

    def test_clip_records_step(self, basic_df):
        _, steps = apply_schema(basic_df, Schema(salary=Col(clip=(0, 100000))))
        clip_steps = [s for s in steps if s["action"] == "clip"]
        assert len(clip_steps) == 1
        assert clip_steps[0]["before"] > 0

    def test_clip_non_numeric_skipped(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        cleaned, _ = apply_schema(df, Schema(name=Col(clip=(0, 100))))
        assert list(cleaned["name"]) == ["Alice", "Bob"]

    def test_dtype_category(self, basic_df):
        cleaned, _ = apply_schema(basic_df, Schema(dept=Col(dtype="category")))
        assert str(cleaned["dept"].dtype) == "category"

    def test_dtype_float(self):
        df = pd.DataFrame({"age": ["25", "30", "35"]})
        cleaned, _ = apply_schema(df, Schema(age=Col(dtype="float")))
        assert cleaned["age"].dtype == "float64"

    def test_dtype_int(self):
        df = pd.DataFrame({"count": [1.0, 2.0, 3.0]})
        cleaned, _ = apply_schema(df, Schema(count=Col(dtype="int")))
        assert str(cleaned["count"].dtype) == "Int64"

    def test_rename(self, basic_df):
        cleaned, _ = apply_schema(basic_df, Schema(dept=Col(rename="department")))
        assert "department" in cleaned.columns
        assert "dept" not in cleaned.columns

    def test_case_insensitive_column(self):
        df = pd.DataFrame({"Age": [25.0, None, 30.0]})
        cleaned, _ = apply_schema(df, Schema(age=Col(null="median")))
        assert cleaned["Age"].isna().sum() == 0

    def test_missing_optional_column_skipped(self):
        df = pd.DataFrame({"age": [25.0]})
        schema = Schema(age=Col(null="median"), ghost=Col(null="mean"))
        cleaned, steps = apply_schema(df, schema)  # should not raise
        skips = [s for s in steps if s["action"] == "skip"]
        assert len(skips) == 1

    def test_required_column_raises(self):
        df = pd.DataFrame({"age": [25.0]})
        schema = Schema(salary=Col(null="median", required=True))
        with pytest.raises(KeyError, match="Required column"):
            apply_schema(df, schema)

    def test_mode_raise_with_dirty_data(self):
        df = pd.DataFrame({"age": [25.0, None]})
        with pytest.raises(DataQualityError):
            apply_schema(df, Schema(age=Col(null="median")), mode="raise")

    def test_mode_raise_with_clean_data(self, clean_df, basic_schema):
        # Clean data should pass mode=raise
        cleaned, steps = apply_schema(clean_df, basic_schema, mode="raise")
        assert cleaned is not None

    def test_original_df_not_mutated(self, basic_df):
        original_nulls = basic_df["age"].isna().sum()
        apply_schema(basic_df, Schema(age=Col(null="median")))
        assert basic_df["age"].isna().sum() == original_nulls

    def test_auto_strategy_applied(self, basic_df):
        schema = Schema(age=Col(null="auto"), dept=Col(null="auto"))
        cleaned, _ = apply_schema(basic_df, schema)
        assert cleaned["age"].isna().sum() == 0
        assert cleaned["dept"].isna().sum() == 0

    def test_ignore_strategy_leaves_nulls(self, basic_df):
        original_nulls = basic_df["age"].isna().sum()
        cleaned, _ = apply_schema(basic_df, Schema(age=Col(null="ignore")))
        assert cleaned["age"].isna().sum() == original_nulls

    def test_combined_null_clip_dtype(self):
        df = pd.DataFrame({"age": [25.0, None, 200.0]})
        cleaned, _ = apply_schema(
            df,
            Schema(age=Col(null="median", clip=(0, 120), dtype="float"))
        )
        assert cleaned["age"].isna().sum() == 0
        assert cleaned["age"].max() <= 120
        assert cleaned["age"].dtype == "float64"


# ─── @clean decorator ─────────────────────────────────────────────────────────

class TestCleanDecorator:
    def test_returns_clean_result(self, basic_df, basic_schema):
        @clean(basic_schema)
        def process(df): pass
        assert isinstance(process(basic_df), CleanResult)

    def test_df_is_cleaned(self, basic_df, basic_schema):
        @clean(basic_schema)
        def process(df): return df
        result = process(basic_df)
        assert result.df["age"].isna().sum() == 0
        assert result.df["salary"].min() >= 0

    def test_function_receives_clean_df(self, basic_df, basic_schema):
        seen = {}
        @clean(basic_schema)
        def process(df):
            seen["nulls"] = df["age"].isna().sum()
        process(basic_df)
        assert seen["nulls"] == 0

    def test_kwargs_df(self, basic_df, basic_schema):
        @clean(basic_schema)
        def process(df): pass
        assert isinstance(process(df=basic_df), CleanResult)

    def test_preserves_function_name(self, basic_schema):
        @clean(basic_schema)
        def my_special_function(df): pass
        assert my_special_function.__name__ == "my_special_function"

    def test_preserves_docstring(self, basic_schema):
        @clean(basic_schema)
        def process(df):
            """My docstring."""
            pass
        assert process.__doc__ == "My docstring."

    def test_no_dataframe_raises(self, basic_schema):
        @clean(basic_schema)
        def process(x, y): pass
        with pytest.raises(TypeError, match="could not find a pd.DataFrame"):
            process(1, 2)

    def test_wrong_schema_type_raises(self):
        with pytest.raises(TypeError):
            @clean({"age": "median"})
            def f(df): pass

    def test_mode_warn_emits_warning(self, basic_df, basic_schema):
        @clean(basic_schema, mode="warn")
        def process(df): pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            process(basic_df)
            assert len(w) > 0

    def test_mode_raise_raises_on_dirty(self, basic_df, basic_schema):
        @clean(basic_schema, mode="raise")
        def process(df): pass
        with pytest.raises(DataQualityError):
            process(basic_df)

    def test_return_value_captured(self, basic_df, basic_schema):
        @clean(basic_schema)
        def process(df): return 42
        assert process(basic_df).return_value == 42

    def test_metadata_attached(self, basic_schema):
        @clean(basic_schema)
        def process(df): pass
        assert process._cleanframe_decorator == "clean"
        assert process._cleanframe_schema is basic_schema


# ─── @validate decorator ──────────────────────────────────────────────────────

class TestValidateDecorator:
    def test_raises_on_dirty(self, basic_df, basic_schema):
        @validate(basic_schema)
        def process(df): pass
        with pytest.raises(DataQualityError) as exc_info:
            process(basic_df)
        assert len(exc_info.value.issues) > 0

    def test_passes_on_clean_data(self, clean_df, basic_schema):
        @validate(basic_schema)
        def process(df): return "ok"
        assert process(clean_df) == "ok"

    def test_does_not_modify_df(self, basic_df, basic_schema):
        original_nulls = int(basic_df["age"].isna().sum())
        @validate(basic_schema)
        def process(df): pass
        try:
            process(basic_df)
        except DataQualityError:
            pass
        assert int(basic_df["age"].isna().sum()) == original_nulls

    def test_wrong_schema_raises(self):
        with pytest.raises(TypeError):
            @validate("not a schema")
            def f(df): pass


# ─── @audit decorator ────────────────────────────────────────────────────────

class TestAuditDecorator:
    def test_returns_clean_result(self, basic_df, basic_schema, capsys):
        @audit(basic_schema)
        def process(df): pass
        assert isinstance(process(basic_df), CleanResult)

    def test_prints_cleaning_report(self, basic_df, basic_schema, capsys):
        @audit(basic_schema)
        def process(df): pass
        process(basic_df)
        out = capsys.readouterr().out
        assert "cleanframe" in out
        assert "Cleaning report" in out


# ─── @profile decorator ──────────────────────────────────────────────────────

class TestProfileDecorator:
    def test_prints_report(self, basic_df, capsys):
        @profile
        def explore(df): pass
        explore(basic_df)
        out = capsys.readouterr().out
        assert "cleanframe" in out
        assert "Shape" in out

    def test_does_not_modify_df(self, basic_df):
        original_nulls = int(basic_df["age"].isna().sum())
        @profile
        def explore(df): pass
        explore(basic_df)
        assert int(basic_df["age"].isna().sum()) == original_nulls

    def test_no_schema_needed(self, basic_df):
        @profile
        def explore(df): return 99
        result = explore(basic_df)
        assert result == 99  # @profile returns function's return value directly


# ─── @pipeline decorator ─────────────────────────────────────────────────────

class TestPipelineDecorator:
    def test_applies_multiple_schemas(self, basic_df):
        s1 = Schema(age=Col(null="median"), dept=Col(null="mode"))
        s2 = Schema(salary=Col(null="mean", clip=(0, 200000)))

        @pipeline(s1, s2)
        def process(df): pass

        result = process(basic_df)
        assert result.df["age"].isna().sum() == 0
        assert result.df["salary"].isna().sum() == 0
        assert result.df["salary"].min() >= 0

    def test_steps_merged_from_all_schemas(self, basic_df):
        s1 = Schema(age=Col(null="median"))
        s2 = Schema(salary=Col(null="mean"))

        @pipeline(s1, s2)
        def process(df): pass

        result = process(basic_df)
        active = [s for s in result.steps if s["action"] == "fill_null"]
        assert len(active) >= 2

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            @pipeline({"age": "median"})
            def f(df): pass


# ─── CleanResult ──────────────────────────────────────────────────────────────

class TestCleanResult:
    def test_explain_returns_string(self, basic_df, basic_schema):
        @clean(basic_schema)
        def process(df): pass
        result = process(basic_df)
        assert isinstance(result.explain(), str)
        assert len(result.explain()) > 0

    def test_explain_no_changes(self):
        df = pd.DataFrame({"age": [25.0, 30.0]})
        @clean(Schema(age=Col(null="median")))
        def process(df): pass
        assert "No changes" in process(df).explain()

    def test_to_code_has_import(self, basic_df, basic_schema):
        @clean(basic_schema)
        def process(df): pass
        code = process(basic_df).to_code()
        assert "import pandas as pd" in code
        assert "df.copy()" in code

    def test_to_code_executable(self, basic_df, basic_schema):
        @clean(basic_schema)
        def process(df): pass
        code = process(basic_df).to_code()
        exec_code = code.replace(
            "df = df.copy()",
            f"df = pd.DataFrame({basic_df.to_dict()!r}).copy()"
        )
        exec(exec_code, {"pd": pd})  # should not raise

    def test_summary_returns_string(self, basic_df, basic_schema):
        @clean(basic_schema)
        def process(df): pass
        s = process(basic_df).summary()
        assert isinstance(s, str)
        assert "cleanframe" in s

    def test_repr_has_cleanresult(self, basic_df, basic_schema):
        @clean(basic_schema)
        def process(df): pass
        assert "CleanResult" in repr(process(basic_df))


# ─── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame({"age": pd.Series([], dtype=float)})
        @clean(Schema(age=Col(null="median")))
        def process(df): pass
        result = process(df)
        assert len(result.df) == 0

    def test_all_nulls_column(self):
        df = pd.DataFrame({"age": [None, None, None]})
        @clean(Schema(age=Col(null="median")))
        def process(df): pass
        result = process(df)  # NaN median → fillna(NaN) → still NaN, but no crash
        assert isinstance(result, CleanResult)

    def test_no_nulls_no_fill_step(self):
        df = pd.DataFrame({"age": [25.0, 30.0, 35.0]})
        @clean(Schema(age=Col(null="median")))
        def process(df): pass
        result = process(df)
        fills = [s for s in result.steps if s["action"] == "fill_null"]
        assert len(fills) == 0

    def test_stacking_profile_and_clean(self, basic_df, basic_schema):
        @profile
        @clean(basic_schema)
        def process(df): pass
        result = process(basic_df)
        assert isinstance(result, CleanResult)

    def test_large_dataframe_performance(self):
        import time
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "age":    rng.choice([20.0, 30.0, None], size=100_000),
            "salary": rng.choice([50000.0, 80000.0, -1.0], size=100_000),
        })
        schema = Schema(
            age    = Col(null="median", clip=(0, 120)),
            salary = Col(null="mean",   clip=(0, 200_000)),
        )
        @clean(schema)
        def process(df): pass
        start = time.time()
        result = process(df)
        assert time.time() - start < 5.0
        assert result.df["age"].isna().sum() == 0

    def test_second_arg_dataframe(self, basic_df, basic_schema):
        """DataFrame passed as second argument should still be found."""
        @clean(basic_schema)
        def process(name: str, df: pd.DataFrame): pass
        result = process("test", basic_df)
        assert isinstance(result, CleanResult)
