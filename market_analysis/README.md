# Market Direction Analysis

A compact quantitative research project built from an exploratory market-data notebook. The pipeline cleans a multi-asset time series, engineers return/risk/market features, creates a next-day direction target, performs a chronological train/test split, and evaluates a logistic-regression baseline.

## Structure

```text
adia_market_analysis/
├── main.py
├── requirements.txt
├── data/
├── outputs/
└── src/
    ├── config.py
    ├── data_cleaning.py
    ├── eda.py
    ├── features.py
    ├── market_model.py
    ├── modeling.py
    ├── evaluation.py
    └── visualization.py
```

## Features

The feature set mirrors the original analysis: daily and lagged returns, 5- and 20-day momentum, rolling return statistics and annualized volatility, 60-day rolling market beta, relative volume and a 20-day volume z-score, market return, and sentiment.

The target is whether the asset's next daily return is positive. The train/test split is chronological rather than random to avoid training on future observations.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py data/data_analysis_practice.csv
```

Generated summaries, predictions, coefficients, and plots are written to `outputs/`.

## Notes

This repository is intended as a transparent research/practice workflow rather than a production trading system. The logistic regression is a simple interpretable baseline; performance should be assessed out of sample and with appropriate attention to transaction costs, stability, and leakage before any trading interpretation.
