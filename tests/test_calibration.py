"""Tests for the calibration table (evaluate/calibration.py)."""

import pandas as pd
import pytest

from f1predictor.evaluate.calibration import calibration_table, wilson_interval


def test_buckets_average_prediction_and_actual_rate():
    # Three 10% predictions (one happened) and two 60% predictions (both happened).
    prob = pd.Series([0.1, 0.1, 0.1, 0.6, 0.6])
    happened = pd.Series([1, 0, 0, 1, 1])
    table = calibration_table(prob, happened, [0.0, 0.5, 1.0]).set_index("bucket")
    assert table.loc["0%-50%", "n"] == 3
    assert table.loc["0%-50%", "predicted"] == pytest.approx(0.1)
    assert table.loc["0%-50%", "actual"] == pytest.approx(1 / 3)
    assert table.loc["50%-100%", "actual"] == 1.0


def test_a_probability_of_exactly_one_lands_in_the_top_bucket():
    table = calibration_table(pd.Series([1.0, 0.0]), pd.Series([1, 0]), [0.0, 0.5, 1.0])
    assert table["n"].tolist() == [1, 1]


def test_empty_buckets_are_left_out():
    table = calibration_table(pd.Series([0.01, 0.02]), pd.Series([0, 0]), [0.0, 0.5, 1.0])
    assert table["bucket"].tolist() == ["0%-50%"]


def test_wilson_interval_known_values():
    # 0 of 10: the range still allows a rate up to ~28%: no hits in 10 tries
    # isn't proof the rate is 0.
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-3)
    # 50 of 100: centred on 50%, roughly +/- 10 points.
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.404, abs=1e-3) and high == pytest.approx(0.596, abs=1e-3)
