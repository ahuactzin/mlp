import pandas as pd
from dataclasses import dataclass, field

@dataclass
class DFStepRecord:
    step_name: str
    n_rows: int
    n_cols: int
    columns: list

class DataFrameEvolutionTracker:
    def __init__(self, df: pd.DataFrame):
        """Initialize the tracker with the first dataframe snapshot."""
        self.records: list[DFStepRecord] = []
        self.register_step("Initialization", df)

    def register_step(self, step_name: str, df: pd.DataFrame):
        """Register the dataframe state at a specific step."""
        record = DFStepRecord(
            step_name=step_name,
            n_rows=df.shape[0],
            n_cols=df.shape[1],
            columns=list(df.columns)
        )
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the tracking history as a pandas DataFrame."""
        return pd.DataFrame({
            "Step": [r.step_name for r in self.records],
            "Rows": [r.n_rows for r in self.records],
            "Columns": [r.n_cols for r in self.records],
            "Column Names": [r.columns for r in self.records],
        })

    def __repr__(self):
        return self.to_dataframe().__repr__()
