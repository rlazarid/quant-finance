# Quantitative Finance

A collection of research projects exploring **machine learning, statistical modeling, and computational methods in quantitative finance**.

The focus is on rigorous empirical evaluation: time-aware validation, careful treatment of data leakage and bias, comparison against simple statistical baselines, and evaluation using financially meaningful metrics.

## Projects

### 1. Cross-Sectional Equity Return Prediction

Machine-learning models for predicting the **relative performance of equities** using publicly available market data.

Rather than predicting the direction of an individual stock, the problem is formulated cross-sectionally: at each point in time, stocks are ranked by their predicted forward excess returns and evaluated through both predictive and portfolio-level metrics.

**Methods**

* Cross-sectional feature engineering
* Momentum, volatility, liquidity, trend, and market-risk signals
* Ridge regression and gradient-boosted trees
* Expanding-window walk-forward validation
* Long-short portfolio construction
* Transaction-cost and turnover analysis

**Evaluation**

* Spearman Information Coefficient (IC)
* IC stability through time
* Annualized return and volatility
* Sharpe ratio
* Maximum drawdown
* Portfolio turnover

The project also explicitly considers common sources of bias in financial ML, including look-ahead bias, overlapping prediction horizons, and survivorship bias.

→ [`cross-sectional-equities/`](cross-sectional-equities/)

