import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import CATEGORICAL, FORBIDDEN, NUMERIC, network_rate_30d, prior_stats  # noqa: E402


def test_no_forbidden_columns_are_features():
    assert not FORBIDDEN & set(NUMERIC + CATEGORICAL)


def test_history_ignores_orders_not_yet_delivered():
    # Order B is purchased after order A was purchased, but before A arrived,
    # so B must not be able to see whether A was late.
    purchase = ["2018-01-01", "2018-01-05", "2018-01-20"]
    delivered = ["2018-01-10", "2018-01-12", "2018-01-25"]
    counts, totals = prior_stats(["s", "s", "s"], purchase, delivered, [1, 0, 0])
    assert counts.tolist() == [0, 0, 2]
    assert totals.tolist() == [0, 0, 1]


def test_history_is_separate_for_each_group():
    purchase = ["2018-01-01", "2018-02-01"]
    delivered = ["2018-01-10", "2018-02-10"]
    counts, _ = prior_stats(["a", "b"], purchase, delivered, [1, 1])
    assert counts.tolist() == [0, 0]


def test_changing_a_future_label_does_not_change_past_features():
    purchase = pd.to_datetime(["2018-01-01", "2018-01-15", "2018-03-01"])
    delivered = pd.to_datetime(["2018-01-05", "2018-01-20", "2018-03-05"])
    first = network_rate_30d(purchase, delivered, [1, 0, 0])
    second = network_rate_30d(purchase, delivered, [1, 0, 1])
    assert np.array_equal(first, second, equal_nan=True)


def test_network_rate_uses_only_the_last_30_days():
    purchase = pd.to_datetime(["2018-01-01", "2018-03-20"])
    delivered = pd.to_datetime(["2018-01-02", "2018-03-25"])
    rates = network_rate_30d(purchase, delivered, [1, 0])
    assert np.isnan(rates).all()
