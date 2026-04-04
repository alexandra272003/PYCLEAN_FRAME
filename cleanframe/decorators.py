from functools import wraps
from .engine import apply_schema


# CLEAN
def clean(schema, mode="fix"):
    def decorator(func):

        @wraps(func)
        def wrapper(df, *args, **kwargs):
            result = apply_schema(df, schema, mode)
            func(result.df, *args, **kwargs)
            return result

        return wrapper
    return decorator


# VALIDATE
def validate(schema):
    def decorator(func):

        @wraps(func)
        def wrapper(df, *args, **kwargs):
            result = apply_schema(df, schema, mode="raise")
            return func(result.df, *args, **kwargs)

        return wrapper
    return decorator


# AUDIT
def audit(schema):
    def decorator(func):

        @wraps(func)
        def wrapper(df, *args, **kwargs):
            result = apply_schema(df, schema, mode="fix")

            print("\n--- AUDIT LOG ---")
            for log in result.logs:
                print(log)

            return func(result.df, *args, **kwargs)

        return wrapper
    return decorator


# PROFILE
def profile(func):

    @wraps(func)
    def wrapper(df, *args, **kwargs):

        print("\n--- DATA PROFILE ---")
        print("Columns:", list(df.columns))

        print("\nDtypes:")
        print(df.dtypes)

        print("\nNull counts:")
        print(df.isnull().sum())

        print("\nStats:")
        print(df.describe(include="all"))

        return func(df, *args, **kwargs)

    return wrapper


# PIPELINE
def pipeline(*schemas):
    def decorator(func):

        @wraps(func)
        def wrapper(df, *args, **kwargs):

            current_df = df

            for i, schema in enumerate(schemas):

                result = apply_schema(current_df, schema, mode="fix")
                current_df = result.df

                print(f"\n--- PIPELINE STEP {i+1} ---")
                for log in result.logs:
                    print(log)

            return func(current_df, *args, **kwargs)

        return wrapper
    return decorator