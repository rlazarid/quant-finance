"""Return, rolling-risk, lag, momentum, and volume feature engineering."""
import numpy as np
import pandas as pd
from .config import ROLLING_WINDOW, TRADING_DAYS


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["asset", "date"]).reset_index(drop=True).copy()
    out["prev_close"] = out.groupby("asset")["close"].shift(1)
    out["return_1d"] = out["close"] / out["prev_close"] - 1
    return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rolling = (
        out.groupby("asset")["return_1d"]
        .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
        .agg(["mean", "std", "min", "max"])
        .reset_index(level=0, drop=True)
        .rename(columns={
            "mean": "rolling_mean_20d",
            "std": "rolling_vol_20d",
            "min": "rolling_min_20d",
            "max": "rolling_max_20d",
        })
    )
    out = out.join(rolling)
    out["rolling_vol_20d_ann"] = out["rolling_vol_20d"] * np.sqrt(TRADING_DAYS)
    return out


def add_predictive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lag in [1, 2, 3, 5, 10]:
        out[f"return_1d_lag_{lag}"] = out.groupby("asset")["return_1d"].shift(lag)

    out["momentum_5d"] = out.groupby("asset")["close"].pct_change(5, fill_method=None)
    out["momentum_20d"] = out.groupby("asset")["close"].pct_change(20, fill_method=None)
    out["mean_return_5d"] = out.groupby("asset")["return_1d"].transform(lambda x: x.rolling(5).mean())
    out["volatility_5d"] = out.groupby("asset")["return_1d"].transform(lambda x: x.rolling(5).std())

    out["volume_mean_20d"] = out.groupby("asset")["volume"].transform(lambda x: x.rolling(20).mean())
    out["relative_volume"] = out["volume"] / out["volume_mean_20d"]
    out["volume_std_20d"] = out.groupby("asset")["volume"].transform(lambda x: x.rolling(20).std())
    out["volume_zscore_20d"] = (out["volume"] - out["volume_mean_20d"]) / out["volume_std_20d"]
    out["cross_sectional_z"] = out.groupby("date")["return_1d"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create next-day return and binary up/down target within each asset."""
    out = df.copy()
    out["target_return_1d"] = out.groupby("asset")["return_1d"].shift(-1)
    out["target_up"] = (
        out["target_return_1d"].gt(0)
        .where(out["target_return_1d"].notna())
        .astype("Int64")
    )
    return out
