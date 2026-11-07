"""Load the raw Olist CSVs into a local SQLite database."""
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "olist.zip"
DB_PATH = ROOT / "data" / "olist.db"

TABLES = {
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_customers_dataset.csv": "customers",
    "olist_sellers_dataset.csv": "sellers",
    "olist_products_dataset.csv": "products",
    "olist_geolocation_dataset.csv": "geolocation",
    "product_category_name_translation.csv": "category_translation",
}

INDEXES = [
    "CREATE INDEX idx_items_order ON order_items(order_id)",
    "CREATE INDEX idx_items_seller ON order_items(seller_id)",
    "CREATE INDEX idx_pay_order ON order_payments(order_id)",
    "CREATE INDEX idx_orders_customer ON orders(customer_id)",
    "CREATE INDEX idx_geo_zip ON geolocation(geolocation_zip_code_prefix)",
]


def main() -> None:
    DB_PATH.unlink(missing_ok=True)
    with zipfile.ZipFile(RAW_ZIP) as archive, sqlite3.connect(DB_PATH) as con:
        for filename, table in TABLES.items():
            with archive.open(filename) as handle:
                frame = pd.read_csv(handle)
            frame.to_sql(table, con, index=False)
            print(f"{table:22s} {len(frame):>9,d} rows")
        for statement in INDEXES:
            con.execute(statement)


if __name__ == "__main__":
    main()
