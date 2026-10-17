"""Builds the mock orders SQLite DB from data/seed/orders.json.

Drop-and-recreate seed script, not an incremental migration - appropriate for
mock dev data with no real history to preserve. Re-run anytime after editing
the seed JSON.

Schema is normalized (orders / order_items / shipment_timeline as separate
tables) rather than JSON-in-a-column, since the whole point of moving off flat
JSON is to let SQL do structured lookups instead of a Python linear scan over
nested lists.

Every table uses a surrogate integer `id` as its real primary key, and
relations between tables point at that `id` (order_items.order_id,
shipment_timeline.order_id), not at the text order_number. `order_number`
stays on `orders` as a separate UNIQUE column - it's still the natural/business
key callers (get_order_status) look up by, but it is no longer what joins are
built on.

Every table also carries `status`, `created_at`, `updated_at` for a uniform
shape. Note `shipment_timeline.timestamp` (when the real-world shipping event
happened) and `shipment_timeline.created_at`/`updated_at` (when this DB row was
inserted/touched) are different concepts that happen to sit next to each
other - don't conflate them. `order_items.status` has no real meaning yet
(nothing reads it) - it's included purely for schema consistency, defaulted to
'active'.

`updated_at` is refreshed automatically by an AFTER UPDATE trigger per table,
so application code never has to remember to set it by hand.

Scope note: the seed JSON's `return_request` field (present on one order) is
not migrated - nothing currently reads it, so there's no table for it yet.
"""

import json
import sqlite3
from pathlib import Path

SEED_PATH = Path(__file__).parent.parent / "data" / "seed" / "orders.json"
DB_PATH = Path(__file__).parent.parent / "data" / "orders" / "orders.db"

SCHEMA = """
CREATE TABLE orders (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    order_number     TEXT NOT NULL UNIQUE,
    user_email       TEXT NOT NULL,
    status           TEXT NOT NULL,
    total_amount     REAL NOT NULL,
    shipping_city    TEXT,
    shipping_pincode TEXT,
    shipping_line1   TEXT,
    tracking_number  TEXT,
    carrier          TEXT,
    shipment_status  TEXT,
    created_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    sku        TEXT NOT NULL,
    quantity   INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    status     TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE shipment_timeline (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    status     TEXT NOT NULL,
    location   TEXT,
    timestamp  TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_shipment_timeline_order_id ON shipment_timeline(order_id);

CREATE TRIGGER trg_orders_updated_at AFTER UPDATE ON orders
BEGIN
    UPDATE orders SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER trg_order_items_updated_at AFTER UPDATE ON order_items
BEGIN
    UPDATE order_items SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER trg_shipment_timeline_updated_at AFTER UPDATE ON shipment_timeline
BEGIN
    UPDATE shipment_timeline SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
"""


def main():
    print(f"Loading seed orders from {SEED_PATH}")
    orders = json.loads(SEED_PATH.read_text())
    print(f"Loaded {len(orders)} orders")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        print(f"Removing existing DB at {DB_PATH}")
        DB_PATH.unlink()

    print(f"Creating fresh DB at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    for order in orders:
        shipping = order["shipping_address"]
        shipment = order["shipment"]
        print(f"Inserting order {order['order_number']!r} ({order['status']})")

        cursor = conn.execute(
            """
            INSERT INTO orders (
                order_number, user_email, status, total_amount,
                shipping_city, shipping_pincode, shipping_line1,
                tracking_number, carrier, shipment_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order["order_number"],
                order["user_email"],
                order["status"],
                order["total_amount"],
                shipping.get("city"),
                shipping.get("pincode"),
                shipping.get("line1"),
                shipment.get("tracking_number"),
                shipment.get("carrier"),
                shipment.get("status"),
            ),
        )
        order_id = cursor.lastrowid
        print(f"  -> orders.id={order_id}")

        for item in order["items"]:
            conn.execute(
                "INSERT INTO order_items (order_id, sku, quantity, unit_price) VALUES (?, ?, ?, ?)",
                (order_id, item["sku"], item["quantity"], item["unit_price"]),
            )

        for event in shipment.get("timeline", []):
            conn.execute(
                "INSERT INTO shipment_timeline (order_id, status, location, timestamp) VALUES (?, ?, ?, ?)",
                (order_id, event["status"], event.get("location"), event["timestamp"]),
            )

        if "return_request" in order:
            print(f"  (skipping return_request on {order['order_number']!r} - not migrated, see module docstring)")

    conn.commit()

    order_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    item_count = conn.execute("SELECT COUNT(*) FROM order_items").fetchone()[0]
    event_count = conn.execute("SELECT COUNT(*) FROM shipment_timeline").fetchone()[0]
    conn.close()

    print(f"Done: {order_count} orders, {item_count} order_items, {event_count} shipment_timeline rows")


if __name__ == "__main__":
    main()
