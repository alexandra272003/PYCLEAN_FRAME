"""
cleanframe/exceptions.py
Custom exceptions for cleanframe.
"""


class DataQualityError(Exception):
    """
    Raised when data fails validation (mode='raise' or @validate decorator).

    Attributes
    ----------
    issues : list[dict]
        Structured list of problems found.
        Each dict has keys: column, problem, detail.
    """

    def __init__(self, issues: list):
        self.issues = issues
        lines = [f"  [{i['column']}] {i['problem']}: {i['detail']}" for i in issues]
        super().__init__(
            f"Data quality check failed — {len(issues)} issue(s) found:\n" + "\n".join(lines)
        )


class CleanFrameConfigError(ValueError):
    """Raised when Col() or Schema() are configured incorrectly."""
    pass
