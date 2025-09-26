# Moving Average Analyzer

Python toolbox for generating moving-average-based features and peak-to-trough drawdown labels for a single ticker. Data is sourced from Yahoo Finance via `yfinance`, indicators come from `pandas` and the `ta` library, and results are exported as CSV for downstream ML workflows.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Fetch up to 20 years of daily data and save engineered features to a CSV:

```bash
python -m moving_average_analyzer --ticker SPY --output spy_features.csv
```

Key options:

| Flag | Description | Default |
| --- | --- | --- |
| `--ticker` | Ticker symbol to download | **required** |
| `--start` / `--end` | Explicit date range (`YYYY-MM-DD`) | inferred from `--years-back` |
| `--years-back` | History length when dates are omitted | `20` |
| `--slope-window` | Lookback (trading days) for moving-average slope | `10` |
| `--correction-threshold` | Drawdown trigger for correction label | `0.10` (10%) |
| `--bear-threshold` | Drawdown trigger for bear-market label | `0.20` (20%) |

The exported CSV includes:

* Price data (`Open`, `High`, `Low`, `Close`, `Volume`).
* Simple and exponential moving averages (50 / 100 / 200 day).
* Ratios, distance-from-average, and slope diagnostics (50-day and 200-day SMA/EMA).
* Drawdown metrics with binary `InCorrection`, `InBearMarket`, and overall `Label` columns.

## Development

Run the unit tests to validate indicator calculations and labeling logic:

```bash
python -m pytest
```

Feel free to extend `moving_average_analyzer/analyzer.py` with additional technical indicators or alternative labeling schemes.