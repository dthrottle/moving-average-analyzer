from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator, SMAIndicator


@dataclass
class AnalyzerConfig:
    ticker: str
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    slope_window: int = 10
    correction_threshold: float = 0.10
    bear_threshold: float = 0.20
    years_back: int = 20
    output_csv: Optional[str] = None


def _parse_date(date_str: Optional[str]) -> Optional[pd.Timestamp]:
    if date_str is None:
        return None
    return pd.to_datetime(date_str)


def default_start_end(years_back: int = 20) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years_back)
    return start, end


def download_data(
    ticker: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    years_back: int,
) -> pd.DataFrame:
    if start is None or end is None:
        inferred_start, inferred_end = default_start_end(years_back=years_back)
        start = start or inferred_start
        end = end or inferred_end

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, actions=True)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker} between {start.date()} and {end.date()}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=lambda c: c.title())
    columns_to_keep = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns]
    df = df.ffill()
    df = df.dropna(how="any")
    df.index.name = "Date"
    return df


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    closes = df["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    closes = pd.Series(closes, index=df.index, name="Close").astype(float)

    df["SMA50"] = SMAIndicator(close=closes, window=50, fillna=False).sma_indicator()
    df["SMA100"] = SMAIndicator(close=closes, window=100, fillna=False).sma_indicator()
    df["SMA200"] = SMAIndicator(close=closes, window=200, fillna=False).sma_indicator()

    df["EMA50"] = EMAIndicator(close=closes, window=50, fillna=False).ema_indicator()
    df["EMA200"] = EMAIndicator(close=closes, window=200, fillna=False).ema_indicator()
    return df


def compute_indicators(df: pd.DataFrame, slope_window: int) -> pd.DataFrame:
    df["GoldenCross"] = (df["SMA50"] > df["SMA200"]).astype(int)
    df["SMA50_to_SMA200"] = df["SMA50"] / df["SMA200"]
    df["EMA50_to_EMA200"] = df["EMA50"] / df["EMA200"]
    df["Dist_SMA200"] = (df["Close"] - df["SMA200"]) / df["SMA200"]
    df["Dist_EMA200"] = (df["Close"] - df["EMA200"]) / df["EMA200"]

    df["Slope_SMA50"] = (df["SMA50"] - df["SMA50"].shift(slope_window)) / slope_window
    df["Slope_EMA50"] = (df["EMA50"] - df["EMA50"].shift(slope_window)) / slope_window
    df["Slope_SMA200"] = (df["SMA200"] - df["SMA200"].shift(slope_window)) / slope_window
    df["Slope_EMA200"] = (df["EMA200"] - df["EMA200"].shift(slope_window)) / slope_window

    return df


def compute_labels(
    df: pd.DataFrame,
    correction_threshold: float,
    bear_threshold: float,
) -> pd.DataFrame:
    rolling_max = df["Close"].cummax()
    drawdown = (df["Close"] / rolling_max) - 1.0
    df["Drawdown"] = drawdown

    df["InCorrection"] = (drawdown <= -correction_threshold).astype(int)
    df["InBearMarket"] = (drawdown <= -bear_threshold).astype(int)
    df["Label"] = ((df["InCorrection"] == 1) | (df["InBearMarket"] == 1)).astype(int)

    return df


def prepare_dataset(config: AnalyzerConfig) -> pd.DataFrame:
    df = download_data(config.ticker, config.start, config.end, config.years_back)
    df = compute_moving_averages(df)
    df = compute_indicators(df, slope_window=config.slope_window)
    df = compute_labels(
        df,
        correction_threshold=config.correction_threshold,
        bear_threshold=config.bear_threshold,
    )

    df = df.dropna()
    df = df.reset_index()
    selected_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA50",
        "SMA100",
        "SMA200",
        "EMA50",
        "EMA200",
        "GoldenCross",
        "SMA50_to_SMA200",
        "EMA50_to_EMA200",
        "Dist_SMA200",
        "Dist_EMA200",
    "Slope_SMA50",
    "Slope_EMA50",
        "Slope_SMA200",
        "Slope_EMA200",
        "Drawdown",
        "InCorrection",
        "InBearMarket",
        "Label",
    ]

    return df[selected_columns]


def analyze_ticker(config: AnalyzerConfig) -> pd.DataFrame:
    return prepare_dataset(config)


def run_cli(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Moving Average Analyzer")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g., SPY")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--slope-window",
        type=int,
        default=10,
        help="Window (trading days) for slope calculations",
    )
    parser.add_argument(
        "--correction-threshold",
        type=float,
        default=0.10,
        help="Peak-to-trough decline threshold for correction (e.g., 0.10 for 10%)",
    )
    parser.add_argument(
        "--bear-threshold",
        type=float,
        default=0.20,
        help="Peak-to-trough decline threshold for bear market (e.g., 0.20 for 20%)",
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=20,
        help="Number of years of history to download when start/end are omitted",
    )
    parser.add_argument(
        "--output",
        dest="output_csv",
        help="Output CSV file path. Defaults to <ticker>_features.csv",
    )

    args = parser.parse_args(argv)

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    config = AnalyzerConfig(
        ticker=args.ticker.upper(),
        start=start,
        end=end,
        slope_window=args.slope_window,
        correction_threshold=args.correction_threshold,
        bear_threshold=args.bear_threshold,
        years_back=args.years_back,
        output_csv=args.output_csv,
    )

    dataset = analyze_ticker(config)

    output_path = config.output_csv or f"{config.ticker}_features.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Saved features for {config.ticker} to {output_path}")


if __name__ == "__main__":
    run_cli()
