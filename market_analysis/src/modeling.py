"""Model-table construction, chronological splitting, and logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import FEATURES, TARGET


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["date", "asset", "sector"] + FEATURES + [TARGET]]
        .dropna()
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )


def chronological_split(model_df: pd.DataFrame, train_fraction: float = 0.80):
    """Split by unique dates so future observations never enter the training set."""
    dates = np.sort(model_df["date"].unique())
    split_idx = int(train_fraction * len(dates))
    split_date = dates[split_idx]
    train_df = model_df[model_df["date"] < split_date].copy()
    test_df = model_df[model_df["date"] >= split_date].copy()
    return train_df, test_df, split_date


def make_xy(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)
    return model
