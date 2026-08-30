import yfinance as yf
import pandas as pd
from pathlib import Path

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META','NVDA']  # List of stock tickers

data = yf.download(tickers, start='2020-01-01', end='2024-01-01', auto_adjust=True, group_by='ticker')
print(data.head())
