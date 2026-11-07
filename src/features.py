"""Turn the SQL extract into model features without leaking the future."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "olist.db"
SQL_PATH = ROOT / "sql" / "features.sql"

CATEGORICAL = ["customer_state", "seller_state", "category", "payment_type"]
NUMERIC = [
    "est_days", "n_items", "n_sellers", "total_price", "total_freight",
    "total_weight_g", "total_volume_cm3", "payment_installments",
    "distance_km", "same_state", "purchase_dow", "purchase_hour",
    "seller_prior_orders", "seller_late_rate", "route_prior_orders",
    "route_late_rate", "route_mean_days", "slack_days",
    "network_late_rate_30d",
]
# Columns only known after the order is placed. A test checks these stay out.
FORBIDDEN = {"delivered_ts", "actual_days", "is_late", "order_id", "seller_id"}

SMOOTHING = 20
DEFAULT_LATE_RATE = 0.08  # used only before any order has been delivered


def prior_stats(group, purchase_ts, delivered_ts, values):
    """For each order, count and sum `values` over orders in the same group
    that were DELIVERED strictly before this order was PURCHASED.
    """
    purchase = pd.to_datetime(purchase_ts).to_numpy()
    delivered = pd.to_datetime(delivered_ts).to_numpy()
    group = np.asarray(group)
    values = np.asarray(values, dtype=float)
    counts = np.zeros(len(group))
    totals = np.zeros(len(group))
    for key in np.unique(group):
        members = np.flatnonzero(group == key)
        order = members[np.argsort(delivered[members])]
        cumulative = np.concatenate([[0.0], np.cumsum(values[order])])
        seen = np.searchsorted(delivered[order], purchase[members], side="left")
        counts[members] = seen
        totals[members] = cumulative[seen]
    return counts, totals


def network_rate_30d(purchase_ts, delivered_ts, late):
    """Late rate across all orders delivered in the 30 days before purchase."""
    purchase = pd.to_datetime(purchase_ts).to_numpy()
    delivered = pd.to_datetime(delivered_ts).to_numpy()
    order = np.argsort(delivered)
    delivered_sorted = delivered[order]
    cumulative = np.concatenate([[0.0], np.cumsum(np.asarray(late, dtype=float)[order])])
    end = np.searchsorted(delivered_sorted, purchase, side="left")
    start = np.searchsorted(delivered_sorted, purchase - np.timedelta64(30, "D"), side="left")
    n = end - start
    return np.where(n > 0, (cumulative[end] - cumulative[start]) / np.maximum(n, 1), np.nan)


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    purchase = pd.to_datetime(df["purchase_ts"])
    df["purchase_dow"] = purchase.dt.dayofweek
    df["purchase_hour"] = purchase.dt.hour
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["distance_km"] = haversine_km(df["customer_lat"], df["customer_lng"],
                                     df["seller_lat"], df["seller_lng"])

    network = network_rate_30d(df["purchase_ts"], df["delivered_ts"], df["is_late"])
    df["network_late_rate_30d"] = network
    baseline = np.nan_to_num(network, nan=DEFAULT_LATE_RATE)

    n, late = prior_stats(df["seller_id"], df["purchase_ts"], df["delivered_ts"], df["is_late"])
    df["seller_prior_orders"] = n
    df["seller_late_rate"] = (late + SMOOTHING * baseline) / (n + SMOOTHING)

    route = df["seller_state"] + ">" + df["customer_state"]
    n, late = prior_stats(route, df["purchase_ts"], df["delivered_ts"], df["is_late"])
    df["route_prior_orders"] = n
    df["route_late_rate"] = (late + SMOOTHING * baseline) / (n + SMOOTHING)

    n, days = prior_stats(route, df["purchase_ts"], df["delivered_ts"], df["actual_days"])
    df["route_mean_days"] = np.where(n > 0, days / np.maximum(n, 1), np.nan)
    # Days of margin the promised date leaves over this route's usual time.
    df["slack_days"] = df["est_days"] - df["route_mean_days"]
    return df


def load_dataset() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        raw = pd.read_sql_query(SQL_PATH.read_text(encoding="utf-8"), con)
    return add_features(raw)
