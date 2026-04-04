import pandas as pd
from .utils import normalize_columns, find_column


class CleanResult:
    def __init__(self, df, logs):
        self.df = df
        self.logs = logs

    def explain(self):
        return "\n".join(self.logs)

    def to_code(self):
        code = []

        for log in self.logs:
            parts = log.split()
            col = parts[0]

            if "median" in log:
                code.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())")

            elif "mean" in log:
                code.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mean())")

            elif "mode" in log:
                code.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])")

            elif "value" in log:
                val = log.split("(")[-1].strip(")")
                code.append(f"df['{col}'] = df['{col}'].fillna({val})")

            elif "clipped" in log:
                rng = log.split("(")[-1].strip(")")
                code.append(f"df['{col}'] = df['{col}'].clip({rng})")

        return "\n".join(code)


def apply_schema(df: pd.DataFrame, schema, mode="fix"):
    df = df.copy()
    df = normalize_columns(df)

    logs = []

    for col, rules in schema.cols.items():

        real_col = find_column(df.columns, col)

        if not real_col:
            logs.append(f"{col} not found")
            continue

        series = df[real_col]

        # NULL handling
        null_count = series.isnull().sum()

        if rules.null is not None and null_count > 0:

            if mode == "raise":
                raise ValueError(f"{real_col} has null values")

            if rules.null == "mean":
                value = series.mean()
                logs.append(f"{real_col} null->mean ({null_count})")

            elif rules.null == "median":
                value = series.median()
                logs.append(f"{real_col} null->median ({null_count})")

            elif rules.null == "mode":
                value = series.mode()[0]
                logs.append(f"{real_col} null->mode ({null_count})")

            else:
                value = rules.null
                logs.append(f"{real_col} null->value ({value})")

            if mode != "warn":
                df[real_col] = series.fillna(value)

        # CLIP
        if rules.clip:
            low, high = rules.clip
            logs.append(f"{real_col} clipped ({low},{high})")

            if mode != "warn":
                df[real_col] = df[real_col].clip(low, high)

    return CleanResult(df, logs)