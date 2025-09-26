import math

import pandas as pd

from moving_average_analyzer import analyzer


def test_compute_labels_flags_correction_and_bear():
    closes = [100, 110, 108, 90, 88, 70]
    df = pd.DataFrame({"Close": closes})

    result = analyzer.compute_labels(df.copy(), correction_threshold=0.10, bear_threshold=0.20)

    assert result.loc[3, "InCorrection"] == 1
    assert result.loc[3, "InBearMarket"] == 0
    assert result.loc[3, "Label"] == 1
    assert result.loc[5, "InBearMarket"] == 1
    assert result.loc[5, "Label"] == 1


def test_compute_indicators_adds_expected_columns():
    df = pd.DataFrame(
        {
            "Close": [100, 102, 104, 106, 108, 110],
            "SMA50": [100, 100, 100, 100, 100, 100],
            "SMA200": [95, 95, 95, 95, 95, 95],
            "EMA50": [100, 101, 102, 103, 104, 105],
            "EMA200": [98, 98, 98, 98, 98, 98],
        }
    )

    enriched = analyzer.compute_indicators(df.copy(), slope_window=2)

    assert "GoldenCross" in enriched.columns
    assert (enriched["GoldenCross"] == 1).all()
    assert "SMA50_to_SMA200" in enriched.columns
    assert "Dist_SMA200" in enriched.columns
    assert "Slope_SMA50" in enriched.columns
    assert "Slope_EMA50" in enriched.columns
    assert "Slope_SMA200" in enriched.columns

    assert math.isclose(enriched["Slope_SMA50"].iloc[-1], 0.0)
    expected_ema_slope = (enriched["EMA50"].iloc[-1] - enriched["EMA50"].iloc[-3]) / 2
    assert math.isclose(enriched["Slope_EMA50"].iloc[-1], expected_ema_slope)


