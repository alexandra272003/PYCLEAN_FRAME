"""
tests.py — comprehensive test suite for cleanframe
Run: python tests.py
"""

import sys
sys.path.insert(0, '/home/claude/cleanframe')

import pandas as pd
import numpy as np

from cleanframe import clean, validate, audit, profile, pipeline, Col, Schema, DataQualityError, CleanResult

passed = 0
failed = 0

def test(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f"  →  {detail}" if detail else ""))
        failed += 1

def section(title):
    print(f"\n── {title} {'─'*(45-len(title))}")

# ── fixtures ──────────────────────────────────────────────────────────────────

def make_df():
    return pd.DataFrame({
        "Age":        [20.0, None,   25.0,  None, 200.0],
        "salary_usd": [50000, 200000, -1000, 80000, None],
        "Dept":       ["HR", None,   "IT",  "HR", None],
        "score":      [85.0, 90.0,   None,  78.0, 92.0],
    })

schema = Schema(
    age        = Col(null="median",  clip=(0, 120)),
    salary_usd = Col(null="median",  clip=(0, 100_000)),
    dept       = Col(null="mode"),
    score      = Col(null="mean"),
)

# ── Col() validation ──────────────────────────────────────────────────────────
section("Col() validation")

try:
    Col(clip=(0,))
    test("clip=(0,) raises ValueError", False)
except ValueError:
    test("clip=(0,) raises ValueError", True)

try:
    Col(dtype="blob")
    test("dtype='blob' raises ValueError", False)
except ValueError:
    test("dtype='blob' raises ValueError", True)

test("Col() with no args is valid", Col() is not None)
test("Col(null=0) scalar is valid", Col(null=0).null == 0)
test("Col(null='Unknown') string scalar valid", Col(null="Unknown").null == "Unknown")

# ── Schema() validation ───────────────────────────────────────────────────────
section("Schema() validation")

try:
    Schema(age="median")
    test("Schema with non-Col raises TypeError", False)
except TypeError:
    test("Schema with non-Col raises TypeError", True)

test("Schema stores cols dict", len(schema.cols) == 4)
test("Schema repr works", "Schema(" in repr(schema))

# ── Column matching ───────────────────────────────────────────────────────────
section("Column matching (case-insensitive, no partial)")

from cleanframe.utils import find_column

test("exact match",              find_column(["Age","Salary"], "Age") == "Age")
test("case-insensitive match",   find_column(["Age","Salary"], "age") == "Age")
test("strip whitespace match",   find_column(["Age"], " age ") == "Age")
test("no partial match",         find_column(["salary_usd"], "salary") is None,
     "salary must NOT match salary_usd")
test("not found returns None",   find_column(["Age"], "weight") is None)

# ── engine apply_schema ───────────────────────────────────────────────────────
section("engine.apply_schema()")

from cleanframe.engine import apply_schema

df = make_df()
result = apply_schema(df, schema)

test("returns CleanResult",             hasattr(result, "df"))
test("cleaned df has no nulls in age",  result.df["Age"].isnull().sum() == 0)
test("clip applied — no age > 120",     result.df["Age"].max() <= 120)
test("clip applied — no salary < 0",    result.df["salary_usd"].min() >= 0)
test("clip applied — no salary > 100k", result.df["salary_usd"].max() <= 100_000)
test("dept nulls filled with mode",     result.df["Dept"].isnull().sum() == 0)
test("original df NOT mutated",         df["Age"].isnull().sum() == 2,
     f"original had nulls but now has {df['Age'].isnull().sum()}")

# ── all null strategies ───────────────────────────────────────────────────────
section("all null strategies")

base = pd.DataFrame({"v": [1.0, None, 3.0, None, 100.0]})

def check_strategy(strategy, expected_non_null=5):
    s = Schema(v=Col(null=strategy))
    r = apply_schema(base, s)
    return r.df["v"].notnull().sum() == expected_non_null

test("null='mean'",   check_strategy("mean"))
test("null='median'", check_strategy("median"))
test("null='mode'",   check_strategy("mode"))
test("null=0 scalar", check_strategy(0))
test("null='ignore'", check_strategy("ignore", expected_non_null=3))

df_ts = pd.DataFrame({"v": [1.0, None, None, 4.0]})
r_ff = apply_schema(df_ts, Schema(v=Col(null="ffill")))
test("null='ffill'", r_ff.df["v"].tolist() == [1.0, 1.0, 1.0, 4.0])

r_bf = apply_schema(df_ts, Schema(v=Col(null="bfill")))
test("null='bfill'", r_bf.df["v"].tolist() == [1.0, 4.0, 4.0, 4.0])

df_drop = pd.DataFrame({"v": [1.0, None, 3.0]})
r_drop = apply_schema(df_drop, Schema(v=Col(null="drop")))
test("null='drop'", len(r_drop.df) == 2)

# ── auto strategy ─────────────────────────────────────────────────────────────
section("null='auto' strategy detection")

from cleanframe.utils import auto_strategy

test("auto: object dtype → mode",
     auto_strategy("dept", pd.Series(["A", None, "A", "B"])) == "mode")
test("auto: date name → ffill",
     auto_strategy("created_date", pd.Series([1.0, 2.0])) == "ffill")
test("auto: low skew numeric → mean",
     auto_strategy("score", pd.Series([1.0,2.0,3.0,4.0,5.0])) == "mean")
test("auto: high skew numeric → median",
     auto_strategy("salary", pd.Series([1.0,1.0,1.0,1.0,100.0])) == "median")

# ── dtype coerce ──────────────────────────────────────────────────────────────
section("dtype coerce")

df_dtype = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
r_int = apply_schema(df_dtype, Schema(v=Col(dtype="int")))
test("dtype coerce float→int", str(r_int.df["v"].dtype).startswith("int"))

df_cat = pd.DataFrame({"dept": ["HR","IT","HR"]})
r_cat = apply_schema(df_cat, Schema(dept=Col(dtype="category")))
test("dtype coerce str→category", str(r_cat.df["dept"].dtype) == "category")

# ── rename ────────────────────────────────────────────────────────────────────
section("rename")

df_ren = pd.DataFrame({"age": [25.0, None]})
r_ren = apply_schema(df_ren, Schema(age=Col(null="median", rename="age_years")))
test("rename: old col gone",    "age" not in r_ren.df.columns)
test("rename: new col present", "age_years" in r_ren.df.columns)

# ── required columns ──────────────────────────────────────────────────────────
section("required=True")

df_req = pd.DataFrame({"salary": [50000]})
try:
    apply_schema(df_req, Schema(age=Col(null="median", required=True)))
    test("required missing col raises KeyError", False)
except KeyError:
    test("required missing col raises KeyError", True)

# not required — silently skip
r_skip = apply_schema(df_req, Schema(age=Col(null="median", required=False)))
test("non-required missing col skipped", "age" not in r_skip.df.columns)

# ── mode="raise" ──────────────────────────────────────────────────────────────
section("mode='raise'")

df_dirty = pd.DataFrame({"age": [20.0, None, 30.0]})
try:
    apply_schema(df_dirty, Schema(age=Col(null="median")), mode="raise")
    test("mode=raise raises DataQualityError on nulls", False)
except DataQualityError as e:
    test("mode=raise raises DataQualityError on nulls", True)
    test("DataQualityError has issues list", len(e.issues) > 0)

# clean df passes raise mode
df_clean_val = pd.DataFrame({"age": [20.0, 25.0, 30.0]})
r_clean = apply_schema(df_clean_val, Schema(age=Col(null="median")), mode="raise")
test("mode=raise passes on clean data", r_clean.df["age"].isnull().sum() == 0)

# ── explain() ─────────────────────────────────────────────────────────────────
section("CleanResult.explain()")

df_e = make_df()
r_e = apply_schema(df_e, schema)
explanation = r_e.explain()
test("explain() returns string",       isinstance(explanation, str))
test("explain() mentions age",         "Age" in explanation)
test("explain() mentions mode=fix",    "fix" in explanation)

# ── to_code() ─────────────────────────────────────────────────────────────────
section("CleanResult.to_code()")

code = r_e.to_code()
test("to_code() returns string",       isinstance(code, str))
test("to_code() has import pandas",    "import pandas" in code)
test("to_code() has fillna",           "fillna" in code)
test("to_code() has clip",             "clip" in code)
test("to_code() has actual col name",  "Age" in code or "salary_usd" in code)

# ── @clean decorator ──────────────────────────────────────────────────────────
section("@clean decorator")

@clean(schema)
def process(df):
    pass

df_c = make_df()
result_c = process(df_c)
test("@clean returns CleanResult",            isinstance(result_c, CleanResult))
test("@clean cleans age nulls",               result_c.df["Age"].isnull().sum() == 0)
test("@clean original not mutated",           df_c["Age"].isnull().sum() == 2)
test("@clean result.explain() works",         "fix" in result_c.explain())
test("@clean result.to_code() works",         "fillna" in result_c.to_code())

# keyword arg call
result_kw = process(df=df_c)
test("@clean works with keyword df=",  isinstance(result_kw, CleanResult))

# ── @validate decorator ───────────────────────────────────────────────────────
section("@validate decorator")

@validate(schema)
def strict_fn(df):
    return "ok"

df_v_dirty = make_df()
df_v_clean = pd.DataFrame({
    "Age": [20.0, 25.0], "salary_usd": [50000.0, 60000.0],
    "Dept": ["HR","IT"], "score": [85.0, 90.0],
})

try:
    strict_fn(df_v_dirty)
    test("@validate raises on dirty df", False)
except DataQualityError:
    test("@validate raises on dirty df", True)

result_v = strict_fn(df_v_clean)
test("@validate passes on clean df and returns fn result", result_v == "ok")

# ── @profile decorator ────────────────────────────────────────────────────────
section("@profile decorator")

@profile
def inspect_data(df):
    return "profiled"

df_p = make_df()
ret = inspect_data(df_p)
test("@profile returns function result",    ret == "profiled")
test("@profile does not clean df",          df_p["Age"].isnull().sum() == 2)

# ── @pipeline decorator ───────────────────────────────────────────────────────
section("@pipeline decorator")

schema1 = Schema(age=Col(null="median"), dept=Col(null="mode"))
schema2 = Schema(age=Col(clip=(0, 100)), salary_usd=Col(null="median"))

@pipeline(schema1, schema2)
def process_pipeline(df):
    pass

df_pl = make_df()
result_pl = process_pipeline(df_pl)
test("@pipeline returns CleanResult",       isinstance(result_pl, CleanResult))
test("@pipeline fills nulls (stage 1)",     result_pl.df["Age"].isnull().sum() == 0)
test("@pipeline clips (stage 2)",           result_pl.df["Age"].max() <= 100)
test("@pipeline steps tagged with stage",   any("pipeline_stage" in s for s in result_pl.steps))

# ── bad inputs ────────────────────────────────────────────────────────────────
section("bad inputs / edge cases")

try:
    @clean("not a schema")
    def bad_fn(df): pass
    test("@clean with non-Schema raises TypeError", False)
except TypeError:
    test("@clean with non-Schema raises TypeError", True)

# empty df
df_empty = pd.DataFrame({"age": pd.Series([], dtype=float)})
r_empty = apply_schema(df_empty, Schema(age=Col(null="median")))
test("empty df does not crash",  isinstance(r_empty.df, pd.DataFrame))

# all-null column
df_all_null = pd.DataFrame({"age": [None, None, None]})
r_all_null = apply_schema(df_all_null, Schema(age=Col(null="median")))
test("all-null column — median is NaN — handled gracefully",
     isinstance(r_all_null.df, pd.DataFrame))

# wrong type passed to apply_schema
try:
    apply_schema([1,2,3], schema)
    test("non-DataFrame raises TypeError", False)
except TypeError:
    test("non-DataFrame raises TypeError", True)

# ── final score ───────────────────────────────────────────────────────────────
total = passed + failed
print(f"\n{'═'*50}")
print(f"  {passed}/{total} tests passed", "✓" if failed == 0 else f"  ({failed} failed)")
print(f"{'═'*50}\n")
