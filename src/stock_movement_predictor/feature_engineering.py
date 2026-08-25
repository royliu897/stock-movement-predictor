from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


def build_features(frame: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    df = frame.copy()
    grouped = df.groupby("symbol", group_keys=False)

    df["MA_10"] = grouped["close"].transform(lambda s: s.rolling(window=10, min_periods=10).mean())
    df["MA_50"] = grouped["close"].transform(lambda s: s.rolling(window=50, min_periods=50).mean())
    df["MA_200"] = grouped["close"].transform(lambda s: s.rolling(window=200, min_periods=200).mean())
    df["RSI_14"] = grouped["close"].transform(_compute_rsi)
    df["MACDh_12_26_9"] = grouped["close"].transform(_compute_macd_hist)

    close_diff = grouped["close"].diff()
    open_diff = grouped["open"].diff()
    df["rvi_numerator"] = close_diff.add(open_diff).rolling(window=10, min_periods=10).mean()
    df["rvi_denominator"] = (df["high"] - df["low"]).rolling(window=10, min_periods=10).mean()
    df["rvi"] = 100.0 * df["rvi_numerator"] / df["rvi_denominator"].replace(0.0, np.nan)
    df["mean_close"] = grouped["close"].transform(lambda s: s.rolling(window=5, min_periods=5).mean())
    df["std_close"] = grouped["close"].transform(lambda s: s.rolling(window=5, min_periods=5).std())
    df["future_close"] = grouped["close"].shift(-horizon)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

    cleaned = df.dropna(subset=[
        "MA_10",
        "MA_50",
        "MA_200",
        "RSI_14",
        "MACDh_12_26_9",
        "rvi",
        "mean_close",
        "std_close",
        "future_close",
    ]).reset_index(drop=True)
    return cleaned


FEATURE_COLUMNS = [
    "high",
    "low",
    "trade_count",
    "open",
    "volume",
    "vwap",
    "MA_10",
    "MA_50",
    "MA_200",
    "RSI_14",
    "MACDh_12_26_9",
    "rvi_numerator",
    "rvi_denominator",
    "rvi",
    "mean_close",
    "std_close",
]

