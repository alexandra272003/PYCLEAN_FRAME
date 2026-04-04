from dataclasses import dataclass

@dataclass
class Col:
    null: str = None      # "mean", "median", "mode", or a fixed value
    clip: tuple = None    # (min, max)


class Schema:
    def __init__(self, **cols):
        """
        Example:
        Schema(
            age=Col(null="median"),
            salary=Col(clip=(0, 100000))
        )
        """
        self.cols = cols

    def __repr__(self):
        return f"Schema({self.cols})"