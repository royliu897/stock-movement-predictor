from __future__ import annotations

import os
from datetime import datetime

import pandas as pd


def fetch_daily_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not api_key or not api_secret:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_API_SECRET before using the Alpaca fetcher.")

    import alpaca_trade_api as tradeapi

    client = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")
    bars = client.get_bars(symbol, tradeapi.rest.TimeFrame.Day, start=start, end=end).df
    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["symbol"] = symbol
    return bars


def fetch_example_window(symbol: str = "SPY") -> pd.DataFrame:
    return fetch_daily_bars(symbol=symbol, start="2022-01-01", end=datetime.utcnow().strftime("%Y-%m-%d"))

