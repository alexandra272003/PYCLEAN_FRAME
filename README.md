# pycleanframe

**Declarative data cleaning contracts for pandas. Annotation-based. Zero API calls. Offline-first.**

Stop writing repetitive pandas cleaning code. Define your rules once in a `Schema`, decorate your function, and get a clean DataFrame automatically — with a full report of every change made.

---

## Install

```bash
pip install pycleanframe
```

---

## Quick Start

```python
from cleanframe import clean, Col, Schema

# 1. Define your cleaning rules
schema = Schema(
    age    = Col(null="median",  clip=(0, 120)),
    salary = Col(null="median",  clip=(0, 999_999)),
    dept   = Col(null="Unknown", dtype="category"),
)

# 2. Decorate your function
@clean(schema)
def train_model(df):
    model.fit(df)

# 3. Call it with dirty data — it cleans automatically
result = train_model(dirty_df)

# 4. See what changed
print(result.explain())

# 5. Get the equivalent pandas code
print(result.to_code())
```

---

## How It Works

When you call a decorated function:

1. cleanframe scans the DataFrame for issues (nulls, out-of-range values, wrong dtypes)
2. It fixes them according to your `Schema` rules
3. Your function receives the **clean** DataFrame
4. You get back a `CleanResult` with the cleaned data + a full change log

Your original DataFrame is **never mutated** — cleanframe always works on a copy.

---

## Col() — Column Rules

`Col()` defines the cleaning rules for a single column.

```python
Col(
    null="median",   # How to fill missing values (see strategies below)
    clip=(0, 120),   # Clip numeric values to [min, max] range
    dtype="float",   # Coerce column to this dtype after cleaning
    rename="age_yr", # Rename the column after all other operations
    required=True,   # Raise KeyError if this column is missing from the DataFrame
)
```

### Null Strategies

| Strategy | What it does |
|----------|-------------|
| `"mean"` | Fill nulls with the column mean |
| `"median"` | Fill nulls with the column median (robust to outliers) |
| `"mode"` | Fill nulls with the most frequent value |
| `"drop"` | Drop rows that have a null in this column |
| `"ffill"` | Forward fill — copy the last valid value downward |
| `"bfill"` | Backward fill — copy the next valid value upward |
| `"auto"` | Automatically pick the best strategy based on dtype and column name |
| `"ignore"` | Do nothing (default) |
| any scalar | Fill with that exact value, e.g. `null=0` or `null="Unknown"` |

### Dtype Options

| dtype | What it does |
|-------|-------------|
| `"int"` | Convert to nullable integer (Int64) |
| `"float"` | Convert to float64 |
| `"str"` | Convert to string |
| `"bool"` | Convert to boolean |
| `"category"` | Convert to pandas Categorical (saves memory) |
| `"datetime"` | Parse as datetime using `pd.to_datetime` |

---

## Decorators

### `@clean(schema)` — Clean silently

Cleans the DataFrame before your function runs. Returns a `CleanResult`.

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
print(result.df)           # cleaned DataFrame
print(result.return_value) # "done"
print(result.explain())    # full change report
```

---

### `@validate(schema)` — Validate without modifying

Checks the DataFrame and raises `DataQualityError` if any issues are found. **Never modifies the data.** Use this in production APIs where you want to reject bad data instead of fixing it.

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
    print(e.issues)  # list of exactly what's wrong
```

---

### `@audit(schema)` — Clean and print a change log

Same as `@clean` but automatically prints a detailed report of every change. Great for debugging and development.

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
# Automatically prints:
# cleanframe → process() — Cleaning report
# =======================================================
#   ✓ [age] Filled 3 null(s) using median → value: 34.0
#   ✓ [age] Clipped 1 value(s) to range [0, 120]
#   ✓ [dept] Filled 2 null(s) using mode → value: 'Eng'
```

---

### `@profile` — Inspect without any schema

Prints a data quality report before your function runs. No schema needed — just attach it to any function to see what's in the DataFrame.

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

---

### `@pipeline(schema1, schema2, ...)` — Chain multiple schemas

Applies multiple schemas in sequence. Each schema's output feeds into the next. Use this to separate raw cleaning from feature engineering.

```python
from cleanframe import pipeline, Col, Schema

# Step 1: fix raw data quality issues
raw_schema = Schema(
    age  = Col(null="median"),
    dept = Col(null="mode"),
)

# Step 2: engineer features
feature_schema = Schema(
    age  = Col(clip=(0, 120), dtype="float"),
    dept = Col(dtype="category"),
)

@pipeline(raw_schema, feature_schema)
def train(df):
    model.fit(df)

result = train(dirty_df)
print(result.explain())  # shows changes from ALL schemas
```

---

## CleanResult

Every decorator (except `@profile` and `@validate`) returns a `CleanResult` object.

```python
result = train(dirty_df)

result.df            # the cleaned DataFrame
result.return_value  # whatever your function returned
result.steps         # raw list of every change (dicts)
result.explain()     # human-readable report (string)
result.to_code()     # equivalent pandas code you can copy-paste
result.summary()     # one-line summary string
```

### `result.explain()` example output

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

### `result.to_code()` example output

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

---

## Modes

All cleaning decorators support a `mode` parameter:

```python
@clean(schema, mode="fix")    # default — clean silently, no output
@clean(schema, mode="warn")   # clean and emit a UserWarning for each change
@clean(schema, mode="raise")  # raise DataQualityError instead of cleaning
```

---

## Schema Features

### Case-insensitive column matching

```python
# Works even if your DataFrame has 'Age', 'AGE', or ' age '
schema = Schema(age=Col(null="median"))
```

### Optional vs required columns

```python
schema = Schema(
    age    = Col(null="median"),                     # optional — silently skipped if missing
    salary = Col(null="median", required=True),      # required — raises KeyError if missing
)
```

### Rename columns

```python
schema = Schema(
    dept = Col(null="mode", rename="department"),    # renamed after cleaning
)
```

---

## Full Example

```python
import pandas as pd
from cleanframe import clean, Col, Schema

# Dirty data
df = pd.DataFrame({
    "age":    [25, None, 34, None, 200],
    "salary": [50000, 200000, None, 80000, -1000],
    "dept":   ["Eng", None, "HR", "Eng", None],
})

# Define rules
schema = Schema(
    age    = Col(null="median",  clip=(0, 120)),
    salary = Col(null="median",  clip=(0, 999_999)),
    dept   = Col(null="Unknown", dtype="category"),
)

# Decorate and call
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

## Requirements

- Python >= 3.8
- pandas >= 1.3
- numpy >= 1.21

---

## License

MIT