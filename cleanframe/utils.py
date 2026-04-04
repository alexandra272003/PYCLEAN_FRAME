def normalize_columns(df):
    """
    Normalize column names:
    - lowercase
    - remove spaces
    """
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def find_column(df_cols, target):
    """
    Find best matching column name.

    Exact match → first priority  
    Partial match → fallback
    """
    target = target.lower()

    # exact match
    for col in df_cols:
        if target == col:
            return col

    # partial match
    for col in df_cols:
        if target in col:
            return col

    return None