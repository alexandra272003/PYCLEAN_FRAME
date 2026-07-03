# pycleanframe

**Declarative data cleaning contracts for pandas.**
Define your cleaning rules once, attach them to a function with a decorator, and every call is guaranteed to receive clean, validated data — with a full audit trail of exactly what changed, how, and why.

No API calls. No network dependency. No hidden global state. Just a schema, a decorator, and a DataFrame.

```
pip install pycleanframe
```

[![PyPI](https://img.shields.io/pypi/v/pycleanframe.svg)](https://pypi.org/project/pycleanframe/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pycleanframe.svg)](https://pypi.org/project/pycleanframe/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

---

## Table of Contents

- [Why pycleanframe](#why-pycleanframe)
  - [The problem](#the-problem)
  - [The pycleanframe approach](#the-pycleanframe-approach)
  - [Before / After](#before--after)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
  - [Execution lifecycle](#execution-lifecycle)
  - [Immutability guarantee](#immutability-guarantee)
- [Core Concepts](#core-concepts)
  - [Schema](#schema)
  - [Col() — Column Rules](#col--column-rules)
  - [Null Strategies](#null-strategies)
  - [Dtype Options](#dtype-options)
  - [Clipping](#clipping)
- [Decorators](#decorators)
  - [@clean](#clean-schema--clean-silently)
  - [@validate](#validate-schema--validate-without-modifying)
  - [@audit](#audit-schema--clean-and-print-a-change-log)
  - [@profile](#profile--inspect-without-any-schema)
  - [@pipeline](#pipeline-schema1-schema2--chain-multiple-schemas)
  - [Choosing the right decorator](#choosing-the-right-decorator)
- [CleanResult API](#cleanresult)
  - [Attributes and methods](#attributes-and-methods)
  - [Reading result.steps](#reading-resultsteps)
- [Execution Modes](#modes)
- [Schema Features](#schema-features)
- [Error Handling](#error-handling)
- [Testing With pycleanframe](#testing-with-pycleanframe)
- [Performance Notes](#performance-notes)
- [Comparison to Other Tools](#comparison-to-other-tools)
- [Full Example](#full-example)
- [Design Principles](#design-principles)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Requirements](#requirements)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why pycleanframe

### The problem

Every data science team eventually converges on the same failure mode. Production data arrives with nulls, outliers, and inconsistent types. The team writes cleaning code — `fillna()`, `clip()`, `astype()` — and that code lives wherever it was first needed: a notebook cell, a script, a function halfway through a pipeline. As the project grows, so does the number of places that logic is duplicated, each slightly different from the last.

Then the schema changes. A new required column is added. A null-handling strategy needs to change from `"mean"` to `"median"` because someone noticed the mean was being dragged by outliers. Now someone has to grep the codebase for every place age gets filled, salary gets clipped, or dept gets encoded — and update each one by hand, hoping nothing was missed.

Worse, when something breaks three stages downstream, no one can answer a simple question with confidence: **what exactly happened to this data before it reached the model?** Was a column filled with the mean or the median? Was an outlier clipped, or does it still contain the raw value that's currently breaking a `sqrt()` call in the loss function? Without an audit trail, debugging becomes archaeology.

### The pycleanframe approach

pycleanframe treats data cleaning as an explicit **contract** rather than an implicit convention. You describe the desired end state of each column in a `Schema`, once. You attach that schema to a function using a decorator. From that point forward, every call to that function is guaranteed to receive data that satisfies the contract — and you get back a structured record of every transformation that was applied to make that guarantee true.

This shifts data cleaning from "something we all remember to do" to "something that is structurally impossible to skip."

### Before / After

**Before — cleaning logic scattered and undocumented:**

```python
def train_model(df):
    df = df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["age"] = df["age"].clip(0, 120)
    df["salary"] = df["salary"].fillna(df["salary"].median())
    df["salary"] = df["salary"].clip(0, 999_999)
    df["dept"] = df["dept"].fillna("Unknown")
    df["dept"] = df["dept"].astype("category")
    model.fit(df)
```

Every function that touches this DataFrame has to remember to repeat this block, in the right order, with the right values — and nothing records that it happened.

**After — the contract lives in one place, and is enforced automatically:**

```python
schema = Schema(
    age    = Col(null="median",  clip=(0, 120)),
    salary = Col(null="median",  clip=(0, 999_999)),
    dept   = Col(null="Unknown", dtype="category"),
)

@clean(schema)
def train_model(df):
    model.fit(df)
```

The cleaning logic is defined once, is reusable across every function that needs it, and every call now produces a `CleanResult` documenting precisely what was done.

---

## Installation

```bash
pip install pycleanframe
```

For a specific version:

```bash
pip install pycleanframe==<version>
```

pycleanframe does not force an upgrade of pandas or numpy already pinned in a project — it only requires the minimum versions listed under [Requirements](#requirements).

---

## Quick Start

```python
from cleanframe import clean, Col, Schema

# 1. Define your cleaning rules once
schema = Schema(
    age    = Col(null="median",  clip=(0, 120)),
    salary = Col(null="median",  clip=(0, 999_999)),
    dept   = Col(null="Unknown", dtype="category"),
)

# 2. Attach the schema to the function that consumes the data
@clean(schema)
def train_model(df):
    model.fit(df)

# 3. Call it with dirty data — cleaning happens automatically, before your function body runs
result = train_model(dirty_df)

# 4. Inspect exactly what changed, column by column
print(result.explain())

# 5. Get the equivalent raw pandas code — useful for review, documentation, or handoff
print(result.to_code())
```

Five lines of setup replace what would otherwise be a growing, duplicated block of `fillna`/`clip`/`astype` calls scattered across every function that touches this data.

---

## How It Works

### Execution lifecycle

When you call a function decorated with `@clean`, `@audit`, or `@pipeline`, pycleanframe intercepts the call before your function body executes and runs the following sequence:

1. **Copy** — the incoming DataFrame is copied. The original object passed by the caller is never touched.
2. **Scan** — the copy is inspected column-by-column against the `Schema`: nulls are counted, values are checked against any `clip` range, and current dtype is compared against the target dtype.
3. **Resolve** — for each issue found, the corresponding rule from `Col()` is applied: a null-fill strategy, a clip operation, a dtype coercion, or a rename.
4. **Log** — every individual change — not just a per-column summary — is appended to an internal list as a structured step: which column, what kind of change, what value was used, how many rows were affected.
5. **Dispatch** — your original function is called with the now-clean DataFrame as its argument, and executes exactly as written — it has no awareness that cleaning occurred.
6. **Wrap** — the cleaned DataFrame, your function's return value, and the full list of logged steps are bundled together into a single `CleanResult` object, which is what gets returned to the caller.

Because this sequence runs identically every time the function is called, the same schema produces the same guarantees in a notebook, a test suite, and a production service — there is no environment-specific behavior to keep in sync.

### Immutability guarantee

pycleanframe never mutates the DataFrame you pass in. This matters in two common scenarios:

- **Reused DataFrames.** If you pass the same `df` into two differently-schema'd functions, cleaning one will not silently affect the other.
- **Debugging.** If a cleaned result looks wrong, you can always compare it directly against the untouched original, since the original is guaranteed to still be exactly what it was before the call.

---

## Core Concepts

### Schema

A `Schema` is a plain container mapping column names to `Col()` rules. It carries no cleaning logic itself — it is a declaration of intent, which the decorators interpret and execute.

```python
schema = Schema(
    age    = Col(null="median", clip=(0, 120)),
    salary = Col(null="median", clip=(0, 999_999)),
    dept   = Col(null="Unknown", dtype="category"),
)
```

Because a `Schema` is just a Python object, it can be constructed programmatically, stored in a config module, imported and reused across multiple projects, or built up dynamically from a data dictionary — there is nothing decorator-specific baked into its construction.

### `Col()` — Column Rules

`Col()` defines the cleaning contract for a single column. Every parameter is optional — specify only the behavior relevant to that column, and anything unspecified is left untouched.

```python
Col(
    null="median",     # strategy for filling missing values
    clip=(0, 120),      # clamp numeric values to a [min, max] range
    dtype="float",       # coerce to this dtype, applied after null-filling and clipping
    rename="age_yr",      # rename the column once all other operations are complete
    required=True,          # raise KeyError at call time if the column is absent
)
```

**Order of operations within a single column:** null-filling is applied first (so that clipping and dtype coercion operate on a complete column), then clipping, then dtype coercion, then renaming last. This ordering is fixed and is what makes `result.to_code()` deterministic and reproducible.

### Null Strategies

| Strategy    | Behavior                                                                 | Best suited for |
|-------------|----------------------------------------------------------------------------|------------------|
| `"mean"`    | Fill nulls with the column mean                                            | Roughly symmetric numeric distributions with few outliers |
| `"median"`  | Fill nulls with the column median                                           | Numeric distributions with outliers or skew |
| `"mode"`    | Fill nulls with the most frequent value                                      | Categorical or low-cardinality columns |
| `"drop"`    | Drop rows containing a null in this column                                    | Columns where imputation isn't semantically valid (e.g. an ID) |
| `"ffill"`   | Forward-fill — propagate the last valid value downward                         | Ordered/time-series data where the last known value is the best estimate |
| `"bfill"`   | Backward-fill — propagate the next valid value upward                           | Ordered/time-series data where a later value backfills a gap |
| `"auto"`    | Automatically infer a strategy from the column's dtype and observed distribution | Exploratory work, prototyping, first pass over a new dataset |
| `"ignore"`  | Take no action (default)                                                          | Columns that are allowed to contain nulls downstream |
| *scalar*    | Fill with a literal value, e.g. `null=0` or `null="Unknown"`                        | Columns where a specific placeholder is meaningful |

`"auto"` is intended for early exploration, not as a substitute for a deliberate choice in a schema you plan to keep long-term — explicit strategies are easier to reason about in `result.explain()` and easier to review in a pull request.

### Dtype Options

| dtype        | Behavior                                            | Notes |
|--------------|--------------------------------------------------------|-------|
| `"int"`      | Convert to pandas nullable integer (`Int64`)             | Use this instead of `"int64"` if nulls may still be present after filling |
| `"float"`    | Convert to `float64`                                        | |
| `"str"`      | Convert to string                                              | Useful for normalizing mixed-type object columns |
| `"bool"`     | Convert to boolean                                                | |
| `"category"` | Convert to `Categorical`                                            | Reduces memory footprint significantly on low-cardinality columns |
| `"datetime"` | Parse using `pd.to_datetime`                                          | Malformed values that can't be parsed are logged, not silently dropped |

### Clipping

`clip=(min, max)` applies pandas' `clip()` semantics: any value below `min` is raised to `min`, and any value above `max` is lowered to `max`. Either bound can be `None` to leave that side unconstrained, e.g. `clip=(0, None)` enforces a floor with no ceiling.

---

## Decorators

pycleanframe exposes five decorators, each suited to a different point in your workflow — from strict production validation, to permissive auto-fixing, to pure exploratory inspection.

### `@clean(schema)` — Clean silently

Cleans the DataFrame before your function runs and returns a `CleanResult`. This is the default, general-purpose entry point, appropriate anywhere you want cleaning enforced without any console output.

```python
from cleanframe import clean, Col, Schema

schema = Schema(
    age  = Col(null="median", clip=(0, 120)),
    dept = Col(null="Unknown", dtype="category"),
)

@clean(schema)
def train(df):
    model.fit(df)
    return "done"

result = train(dirty_df)
print(result.df)            # cleaned DataFrame
print(result.return_value)  # "done"
print(result.explain())     # full change report
```

### `@validate(schema)` — Validate without modifying

Checks the DataFrame against the schema and raises `DataQualityError` if any violation is found. **The data is never modified under any circumstance.** This is the appropriate decorator at trust boundaries — an API endpoint, a scheduled ETL job's input, a function accepting data from an external partner — where malformed input should be rejected outright rather than silently repaired.

```python
from cleanframe import validate, Col, Schema
from cleanframe.exceptions import DataQualityError

schema = Schema(
    age    = Col(null="median", clip=(0, 120)),
    salary = Col(null="median", clip=(0, 999_999)),
)

@validate(schema)
def predict(df):
    return model.predict(df)

try:
    result = predict(dirty_df)
except DataQualityError as e:
    print(e.issues)  # precise, structured list of every violation found
```

`e.issues` gives a machine-readable breakdown of what failed — column name, violation type, and count — suitable for logging or returning as a structured error response from an API.

### `@audit(schema)` — Clean and print a change log

Functionally identical to `@clean`, but automatically prints a formatted report of every change to stdout the moment the function is called. Ideal for development and debugging sessions, or for scripts run manually where you want visibility without adding a separate `print(result.explain())` call.

```python
from cleanframe import audit, Col, Schema

schema = Schema(
    age  = Col(null="median", clip=(0, 120)),
    dept = Col(null="mode"),
)

@audit(schema)
def process(df):
    pass

process(dirty_df)
# cleanframe → process() — Cleaning report
# =======================================================
#   ✓ [age] Filled 3 null(s) using median → value: 34.0
#   ✓ [age] Clipped 1 value(s) to range [0, 120]
#   ✓ [dept] Filled 2 null(s) using mode → value: 'Eng'
```

### `@profile` — Inspect without any schema

Requires no schema at all. Attach `@profile` to any function to print a data quality report before it runs — useful for quickly understanding an unfamiliar dataset before writing cleaning rules for it, or as a lightweight sanity check inserted temporarily into an existing pipeline.

```python
from cleanframe import profile

@profile
def explore(df):
    pass

explore(df)
# cleanframe @profile  →  explore()
# Shape       : 1000 rows × 5 columns
# Memory      : 42.3 KB
#
#    Column               dtype        nulls   null%   note
#    --------------------------------------------------------
#  ⚠ age                 float64          23    2.3%   skewed (+2.1)
#  ⚠ dept                object           10    1.0%   4 unique
#    salary              float64           0    0.0%
```

The `note` column surfaces distribution warnings — skew, low cardinality, and similar signals — to help you decide which null strategy and dtype are appropriate before you commit to a `Schema`.

### `@pipeline(schema1, schema2, ...)` — Chain multiple schemas

Applies several schemas in sequence, with each stage's output feeding directly into the next. This is the recommended pattern for separating **raw data-quality cleaning** from **feature engineering**, keeping each schema small, focused, and independently reviewable.

```python
from cleanframe import pipeline, Col, Schema

# Stage 1: correct raw data-quality issues
raw_schema = Schema(
    age  = Col(null="median"),
    dept = Col(null="mode"),
)

# Stage 2: engineer features for the model
feature_schema = Schema(
    age  = Col(clip=(0, 120), dtype="float"),
    dept = Col(dtype="category"),
)

@pipeline(raw_schema, feature_schema)
def train(df):
    model.fit(df)

result = train(dirty_df)
print(result.explain())  # aggregated report across every stage, in order
```

Splitting stages this way means a change to feature engineering never risks silently altering how raw nulls are handled, and vice versa — each schema can be tested and reasoned about independently.

### Choosing the right decorator

| Situation | Recommended decorator |
|---|---|
| Standard internal function that should always receive clean data | `@clean` |
| API or service boundary where bad input should be rejected, not fixed | `@validate` |
| Actively debugging or developing, want visibility without extra print statements | `@audit` |
| Exploring a new/unfamiliar dataset, no schema written yet | `@profile` |
| Raw cleaning and feature engineering need to stay logically separate | `@pipeline` |

---

## CleanResult

Every decorator except `@profile` and `@validate` returns a `CleanResult` — a single object bundling the cleaned data, your function's output, and the complete audit trail together, so nothing about the transformation is lost once your function returns.

### Attributes and methods

```python
result = train(dirty_df)

result.df            # the cleaned DataFrame
result.return_value  # whatever your function returned
result.steps         # raw list of every change, as structured dicts
result.explain()     # human-readable report (string)
result.to_code()     # equivalent, copy-pasteable pandas code
result.summary()     # one-line summary string
```

**`result.explain()`**

```
cleanframe → train() — Cleaning report
=======================================================
  ✓ [age] Filled 3 null(s) using median → value: 34.0
  ✓ [age] Clipped 1 value(s) to range [0, 120]
  ✓ [salary] Filled 1 null(s) using median → value: 65000.0
  ✓ [dept] Filled 2 null(s) using scalar('Unknown')
  ✓ [dept] Dtype coerced: object → category

  DataFrame shape after cleaning: (1000, 3)
```

**`result.to_code()`**

```python
import pandas as pd
import numpy as np

# Generated by cleanframe — equivalent pandas code
df = df.copy()

# Fill nulls in 'age' (median)
df['age'] = df['age'].fillna(34.0)

# Clip outliers in 'age'
df['age'] = df['age'].clip(lower=0, upper=120)

# Fill nulls in 'dept' (scalar('Unknown'))
df['dept'] = df['dept'].fillna('Unknown')

# Coerce dtype of 'dept'
df['dept'] = df['dept'].astype('category')
```

### Reading `result.steps`

`result.steps` is the raw, structured form of everything shown in `result.explain()` — a list of dictionaries, one per change, in the exact order they were applied. This is the form to use if you want to programmatically assert on cleaning behavior (see [Testing With pycleanframe](#testing-with-pycleanframe)), log it to a monitoring system, or feed it into your own custom report format.

Each step captures, at minimum: the column affected, the type of operation performed, the strategy or value used, and the number of rows affected — everything `explain()` renders as text is present here in structured form.

---

## Modes

Every cleaning decorator accepts a `mode` parameter that controls how violations are handled at runtime:

```python
@clean(schema, mode="fix")    # default — clean silently, no output
@clean(schema, mode="warn")   # clean, and emit a UserWarning for each change
@clean(schema, mode="raise")  # raise DataQualityError instead of cleaning
```

This allows the exact same schema to serve different purposes across environments without any duplicated logic:

- **`"fix"`** in a notebook or an internal batch job, where auto-repair is the desired behavior.
- **`"warn"`** in staging, where you want to observe how often and where cleaning is actually kicking in before trusting it silently in production.
- **`"raise"`** in a strict production path, where any deviation from the expected contract should halt processing rather than be silently patched over.

---

## Schema Features

**Case-insensitive column matching** — schemas match regardless of case or incidental leading/trailing whitespace in column names, so schemas remain stable even when upstream naming is inconsistent:

```python
# Matches 'Age', 'AGE', or ' age ' in the source DataFrame
schema = Schema(age=Col(null="median"))
```

**Optional vs. required columns** — control whether a missing column is skipped silently or treated as a hard failure at call time:

```python
schema = Schema(
    age    = Col(null="median"),                 # optional — skipped if missing
    salary = Col(null="median", required=True),  # required — raises KeyError if missing
)
```

**Column renaming** — renames are applied after all other cleaning operations, so every other rule in a `Col()` is always written against the *original* column name, not the renamed one:

```python
schema = Schema(
    dept = Col(null="mode", rename="department"),
)
```

---

## Error Handling

pycleanframe defines its own exception hierarchy so calling code can distinguish cleaning-related failures from ordinary pandas errors:

```python
from cleanframe.exceptions import DataQualityError
```

`DataQualityError` is raised in two situations:

1. By `@validate`, whenever any column fails to satisfy its schema.
2. By any cleaning decorator running in `mode="raise"`, at the first violation encountered.

The exception's `.issues` attribute is a structured, iterable list of the specific violations found — not just a formatted string — so it can be logged, serialized, or turned into an API error response without string-parsing an error message.

A missing `required=True` column raises a standard `KeyError`, keeping that failure mode consistent with ordinary Python/pandas behavior rather than introducing a second custom exception type for the same underlying concept.

---

## Testing With pycleanframe

Because every cleaning decorator returns a `CleanResult` with a structured `.steps` list, cleaning behavior is straightforward to assert on directly in a test suite, rather than only being verified by eye via `explain()`:

```python
def test_age_nulls_filled_with_median():
    result = train(dirty_df)
    age_steps = [s for s in result.steps if s["column"] == "age"]
    assert any(s["operation"] == "fill_null" for s in age_steps)
    assert result.df["age"].isnull().sum() == 0

def test_salary_is_clipped():
    result = train(dirty_df)
    assert result.df["salary"].between(0, 999_999).all()
```

This means a schema change that unintentionally alters cleaning behavior — for example, someone changing a null strategy from `"median"` to `"mean"` — is caught by the test suite the same way any other regression would be, rather than only being discovered by inspecting output manually.

---

## Performance Notes

- pycleanframe copies the input DataFrame once per decorated call. For very large DataFrames processed at high frequency, this copy is the dominant cost — the same cost you would pay writing `df = df.copy()` manually, which is standard practice before in-place cleaning regardless of whether pycleanframe is used.
- Converting a column to `"category"` dtype reduces memory for low-cardinality string columns, but is unnecessary and can add overhead on high-cardinality columns — reserve it for columns with a genuinely small number of distinct values.
- `@pipeline` applies schemas sequentially, each producing its own copy internally; for very large datasets with many pipeline stages, consider consolidating related rules into a single `Schema` where the logical separation between stages isn't essential.

---

## Comparison to Other Tools

| | pycleanframe | Manual pandas | Validation-only libraries (e.g. pandera, Great Expectations) |
|---|---|---|---|
| Fixes data automatically | Yes | Manual, per-call | No — validation only |
| Rejects invalid data | Yes, via `@validate` / `mode="raise"` | Manual | Yes |
| Full audit trail of changes | Yes, structured + human-readable | No, unless hand-written | N/A |
| Generates equivalent raw code | Yes, via `to_code()` | N/A | No |
| Attaches directly to functions | Yes, via decorators | No | Typically schema-only, applied separately |
| Network/service dependency | None | None | Varies by tool |

pycleanframe is not a replacement for a full data-validation framework in contexts that need expectation suites, data docs, or integration with orchestration tools — it is intentionally narrower and lighter-weight, focused specifically on the "clean this DataFrame the same way, every time, at this function boundary" problem.

---

## Full Example

```python
import pandas as pd
from cleanframe import clean, Col, Schema

# Dirty input data
df = pd.DataFrame({
    "age":    [25, None, 34, None, 200],
    "salary": [50000, 200000, None, 80000, -1000],
    "dept":   ["Eng", None, "HR", "Eng", None],
})

# Define the contract
schema = Schema(
    age    = Col(null="median",  clip=(0, 120)),
    salary = Col(null="median",  clip=(0, 999_999)),
    dept   = Col(null="Unknown", dtype="category"),
)

@clean(schema)
def process(df):
    return df.shape

result = process(df)

print(result.explain())
# cleanframe → process() — Cleaning report
# =======================================================
#   ✓ [age] Filled 2 null(s) using median → value: 29.5
#   ✓ [age] Clipped 1 value(s) to range [0, 120]
#   ✓ [salary] Filled 1 null(s) using median → value: 65000.0
#   ✓ [salary] Clipped 1 value(s) to range [0, 999999]
#   ✓ [dept] Filled 2 null(s) using scalar('Unknown')
#   ✓ [dept] Dtype coerced: object → category

print(result.to_code())     # copy-paste pandas equivalent
print(result.summary())     # one-line summary
print(result.return_value)  # (5, 3)
```

---

## Design Principles

- **Declarative over imperative.** You describe the *desired end state* of a column — not the sequence of pandas calls needed to get there. This makes schemas readable by anyone on the team, not just whoever wrote the original cleaning code.
- **Explicit over implicit.** Every cleaning action is logged individually, with the exact value and row count involved; nothing happens silently that can't be inspected afterward via `explain()` or `.steps`.
- **Non-mutating by default.** Functions decorated with pycleanframe never modify the caller's original DataFrame, eliminating an entire category of bugs caused by unexpected in-place mutation.
- **Composable.** Schemas are plain objects — they can be built programmatically, extended, unit-tested, and reused across multiple functions and pipeline stages without any decorator-specific machinery required to construct them.
- **Fail loud when asked to.** `mode="raise"` and `@validate` exist specifically so that "fail fast on bad data" is a first-class, one-parameter choice, not something bolted on separately.
- **No hidden dependencies.** pycleanframe runs entirely offline on top of pandas and numpy. It makes no network calls and depends on no external service of any kind — a schema behaves identically whether run on a laptop with no internet connection or inside a fully sandboxed CI job.

---

## Frequently Asked Questions

**Does pycleanframe mutate my original DataFrame?**
No. Every decorated function operates on an internal copy; the DataFrame you pass in is left exactly as it was.

**What happens if a column in the schema doesn't exist in the DataFrame?**
By default it is silently skipped. Set `required=True` on that column's `Col()` to raise a `KeyError` instead.

**Can I use the same schema across multiple functions?**
Yes — a `Schema` is a plain, reusable object. It's common to define shared schemas in a dedicated module and import them wherever needed.

**How is this different from just writing a `clean_df(df)` helper function myself?**
A hand-written helper function has to be manually called, manually kept in sync everywhere it's needed, and doesn't produce a structured audit trail by default. pycleanframe enforces the cleaning step structurally (it's impossible to call the decorated function without it running) and gives you `explain()`/`to_code()`/`.steps` for free.

**Does pycleanframe work with `dask` or `polars` DataFrames?**
No — pycleanframe is built specifically around pandas' API and semantics (`fillna`, `clip`, `astype`, `pd.to_datetime`).

**What Python versions are supported?**
Python 3.8 and above — see [Requirements](#requirements).

---

## Requirements

- Python >= 3.8
- pandas >= 1.3
- numpy >= 1.21

---

## Roadmap

Planned areas of exploration for future releases include: additional null-fill strategies (interpolation-based), optional integration hooks for logging cleaning reports to external monitoring systems, and expanded dtype coercion options for structured/nested pandas columns. Contributions and issue discussion on any of these directions are welcome.

---

## Contributing

Issues and pull requests are welcome. Before opening a PR, please:

1. Open an issue describing the change if it's non-trivial, so design direction can be agreed on before implementation begins.
2. Include tests for any new behavior — the existing test suite is the source of truth for expected behavior, and new functionality should be covered the same way.
3. Run the full test suite locally before submitting, and ensure existing tests continue to pass.
4. Keep pull requests focused on a single change where possible, to make review straightforward.

---

## License

MIT
