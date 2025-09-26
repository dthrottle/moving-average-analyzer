# 📄 Specification: Moving Averages Analyzer for Market Corrections

## 1. Objective

Build a program to compute moving average–based indicators for a single ticker (e.g., SPY, AAPL). The outputs will serve as features for an ML model designed to classify or predict **potential market corrections or bear markets**.

---

## 2. Data Inputs

* **Ticker symbol**: string (e.g., `"SPY"`)
* **Historical price data**: daily OHLCV (Open, High, Low, Close, Volume).

  * Source: Yahoo Finance, Alpha Vantage, Quandl, or custom database.
  * Minimum lookback: 20 years if possible (to capture multiple cycles).

---

## 3. Preprocessing

* **Data cleaning**: fill missing values (forward-fill), handle splits/dividends if needed.
* **Feature windowing**: ensure rolling calculations align with trading days.

---

## 4. Moving Averages Computations

* **Simple Moving Averages (SMA)**:

  * 50-day SMA
  * 100-day SMA
  * 200-day SMA

* **Exponential Moving Averages (EMA)**:

  * 50-day EMA
  * 200-day EMA

* **Derived Indicators**:

  * **Golden Cross / Death Cross**: (50 > 200 vs. 50 < 200)
  * **SMA/EMA ratio**: (SMA50 / SMA200), (EMA50 / EMA200)
  * **Distance from moving average**: `(Price - MA) / MA`
  * **Slope of moving average**: difference over last N days

---

## 5. Correction / Bear Market Signal Logic

* **Correction**:

  * Peak-to-trough decline ≥ 10% from recent high
* **Bear Market**:

  * Peak-to-trough decline ≥ 20% from recent high

These will be **labels** used for ML training, aligned with dates of decline detection.

---

## 6. Output Format

Output features as a **CSV file** for each ticker:

| Date       | Close | SMA50 | SMA200 | EMA50 | EMA200 | GoldenCross | Dist_SMA200 | Slope_SMA200 | Label |
| ---------- | ----- | ----- | ------ | ----- | ------ | ----------- | ----------- | ------------ | ----- |
| 2020-01-01 | 325.0 | 310.2 | 290.5  | 312.5 | 295.0  | 1           | 0.12        | 0.15         | 0     |
| 2020-03-23 | 222.0 | 280.0 | 295.0  | 270.0 | 290.0  | 0           | -0.25       | -0.20        | 1     |

* **Label**:

  * 0 = no correction/bear
  * 1 = in correction/bear market

---

## 7. ML Training Data Considerations

* Align features with **forward-looking windows** (e.g., did a correction happen in next 30/60 trading days?).
* Possible tasks:

  * **Binary classification** (Correction coming vs. not coming).
  * **Multiclass classification** (No correction, Correction, Bear market).
  * **Regression** (predict drawdown magnitude).

---

## 8. Extensibility

* Add more technical indicators (RSI, MACD, Bollinger Bands).
* Apply across multiple tickers for more robust ML training.
* Store results in a database (SQLite/Postgres) instead of CSV.

---

✅ This spec gives you the foundation for:

1. A feature engineering pipeline (moving averages & derived indicators).
2. Labels for supervised ML training.
3. Exportable structured dataset.


