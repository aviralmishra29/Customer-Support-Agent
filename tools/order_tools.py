"""Mock order-lookup tool, backed by the SQLite DB built by ingestion/seed_orders_db.py
(run that script first to generate data/orders/orders.db from data/seed/orders.json)."""

import json
import sqlite3
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

DB_PATH = Path(__file__).parent.parent / "data" / "orders" / "orders.db"

# All real order numbers look like BOAT-2026-001001 (BOAT-<4-digit year>-<6-digit sequence>).
# Enforcing this shape at the schema level means a malformed/hallucinated order number is
# rejected as a validation error before it ever reaches a SQL query, instead of silently
# returning an indistinguishable "no order found" - same not-trust-the-LLM's-output-shape
# principle as the graph-level router's structured output.
ORDER_NUMBER_PATTERN = r"^BOAT-\d{4}-\d{6}$"


class OrderLookupArgs(BaseModel):
    order_number: str = Field(
        pattern=ORDER_NUMBER_PATTERN,
        description="The exact order number, formatted like 'BOAT-2026-001001'.",
    )


def _fetch_order(order_number: str) -> dict | None:
    """order_number is the natural/business key callers use; internally every
    join runs off orders.id, the surrogate PK - see ingestion/seed_orders_db.py."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        order_row = conn.execute(
            "SELECT * FROM orders WHERE order_number = ?", (order_number,)
        ).fetchone()
        if order_row is None:
            return None

        order_id = order_row["id"]

        items = conn.execute(
            "SELECT sku, quantity, unit_price FROM order_items WHERE order_id = ?",
            (order_id,),
        ).fetchall()

        timeline = conn.execute(
            "SELECT status, location, timestamp FROM shipment_timeline "
            "WHERE order_id = ? ORDER BY timestamp ASC",
            (order_id,),
        ).fetchall()

        return {
            "user_email": order_row["user_email"],
            "status": order_row["status"],
            "items": [dict(item) for item in items],
            "tracking": {
                "tracking_number": order_row["tracking_number"],
                "carrier": order_row["carrier"],
                "status": order_row["shipment_status"],
                "timeline": [dict(event) for event in timeline],
            },
        }
    finally:
        conn.close()


@tool(args_schema=OrderLookupArgs)
def get_order_status(order_number: str) -> str:
    """Look up a customer's order status by its exact order number (e.g. 'BOAT-2026-001001').

    Returns the order's current status, items, and shipment tracking details.
    Only call this when the customer has provided a specific order number.
    """
    print(f"get_order_status called with order_number={order_number!r}")

    result = _fetch_order(order_number)
    if result is None:
        print("get_order_status: no matching order found")
        return f"No order found with order number '{order_number}'."

    print(f"get_order_status found: {result}")
    return json.dumps(result)
