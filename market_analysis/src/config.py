"""Project-wide constants."""

TRADING_DAYS = 252
ROLLING_WINDOW = 20
BETA_WINDOW = 60

FEATURES = [
    "return_1d",
    "return_1d_lag_1",
    "return_1d_lag_2",
    "return_1d_lag_3",
    "momentum_5d",
    "momentum_20d",
    "mean_return_5d",
    "volatility_5d",
    "rolling_vol_20d_ann",
    "rolling_beta_60d",
    "relative_volume",
    "volume_zscore_20d",
    "market_return",
    "sentiment",
]

TARGET = "target_up"
