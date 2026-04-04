import pandas as pd
from cleanframe import Schema, Col, clean, validate, audit, profile, pipeline

df = pd.DataFrame({
    "Age": [20, None, 25],
    "salary_usd": [50000, 200000, -1000],
    "Dept": ["HR", None, "IT"]
})

schema = Schema(
    age=Col(null="median"),
    salary=Col(clip=(0, 100000)),
    dept=Col(null="mode")
)

@clean(schema)
def run(df):
    print("\nCLEANED:")
    print(df)

result = run(df)

print("\nEXPLAIN:")
print(result.explain())

print("\nCODE:")
print(result.to_code())