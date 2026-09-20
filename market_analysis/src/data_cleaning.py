"""Data loading and basic cleaning."""
from pathlib import Path
import pandas as pd


def load_and_clean_data(path: str | Path) -> pd.DataFrame:
    """Load the CSV, remove exact duplicates, parse dates, and sort by asset/date."""
    df = pd.read_csv(path)
    df = df.drop_duplicates().copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["asset", "date"]).reset_index(drop=True)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages."""
    return pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
    })
