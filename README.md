# 🚀 CleanFrame

Declarative data cleaning framework for Pandas using decorators.

CleanFrame ensures your data is **always clean before it reaches your function** — automatically and consistently.

---

## ⚡ Why?

Data cleaning in real-world systems is:
- manual
- repetitive
- easy to forget

```python
df.fillna(...)
df.clip(...)
train(df)
````

👉 This leads to bugs and inconsistent pipelines.

---

## ✅ Solution

Attach cleaning rules directly to your function:

```python
@clean(schema)
def train(df):
    ...
```

Now:

* cleaning is automatic
* cannot be skipped
* always consistent

---

## 📦 Features

* `@clean` → automatic cleaning
* `@validate` → strict validation (raises error)
* `@audit` → logs all transformations
* `@profile` → data overview (nulls, dtypes, stats)
* `@pipeline` → multi-step cleaning

---

## 🧩 Example

### 📄 Input CSV (`university.csv`)

```csv
Age,salary_usd,Dept
20,50000,HR
,200000,
25,-1000,IT
```

---

### 🧠 Code

```python
import pandas as pd
from cleanframe import Schema, Col, clean

# load data
df = pd.read_csv("university.csv")

# define rules
schema = Schema(
    age=Col(null="median"),
    salary=Col(clip=(0, 100000)),
    dept=Col(null="mode")
)

# apply cleaning automatically
@clean(schema)
def train(df):
    print(df)

result = train(df)

print("\nEXPLAIN:")
print(result.explain())

print("\nCODE:")
print(result.to_code())
```

---

## ✅ Output

### Cleaned Data

```
    age  salary_usd dept
0  20.0       50000   HR
1  22.5      100000   HR
2  25.0           0   IT
```

---

### Explain

```
age null->median (1)
salary_usd clipped (0,100000)
dept null->mode (1)
```

---

### Generated Code

```python
df['age'] = df['age'].fillna(df['age'].median())
df['salary_usd'] = df['salary_usd'].clip(0,100000)
df['dept'] = df['dept'].fillna(df['dept'].mode()[0])
```

---

## 🧠 How It Works

```
Schema (rules)
      ↓
@clean decorator intercepts function
      ↓
Engine applies rules
      ↓
Clean data passed to function
```

---

## ⚖️ Comparison

| Tool         | Behavior          |
| ------------ | ----------------- |
| Pandas       | manual cleaning   |
| Scikit-learn | pipeline-based    |
| CleanFrame   | enforced cleaning |

---

## 🔐 Privacy

* No external API calls
* Fully offline
* No data leaves your system

---

## 🎯 Use Cases

* ML pipelines
* backend APIs
* data preprocessing
* team projects

---

## ⚙️ Installation

```bash
pip install -e .
```

---

## 🛠 Tech

* Python
* Pandas
* Decorators
* Schema-based design

---

## 🚀 Future Work

* auto mode (`null="auto"`)
* data quality scoring
* visualization

---

## 👤 Author

Alexandra Pratap Singh

---

## ⭐

If you like this project, give it a star.
