"""Exploratory summaries used in the original notebook."""
import pandas as pd


def asset_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["sector", "asset"])
        .agg(
            observations=("date", "size"),
            valid_close=("close", "count"),
            mean_close=("close", "mean"),
            median_close=("close", "median"),
            close_std=("close", "std"),
            min_close=("close", "min"),
            max_close=("close", "max"),
            avg_volume=("volume", "mean"),
            median_volume=("volume", "median"),
            volume_std=("volume", "std"),
            avg_sentiment=("sentiment", "mean"),
        )
        .reset_index()
    )


def add_price_outlier_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag close-price observations outside each asset's 1.5-IQR bounds."""
    out = df.copy()
    out["q1"] = out.groupby("asset")["close"].transform(lambda x: x.quantile(0.25))
    out["q3"] = out.groupby("asset")["close"].transform(lambda x: x.quantile(0.75))
    out["iqr"] = out["q3"] - out["q1"]
    out["lower_bound"] = out["q1"] - 1.5 * out["iqr"]
    out["upper_bound"] = out["q3"] + 1.5 * out["iqr"]
    out["is_price_outlier"] = (out["close"] < out["lower_bound"]) | (out["close"] > out["upper_bound"])
    return out
