"""Mock order-lookup tool for tool-calling tests."""

import json
from pathlib import Path

from langchain_core.tools import tool

ORDERS_PATH = Path(__file__).parent.parent / "data" / "seed" / "orders.json"


@tool
def get_order_status(order_number: str) -> str:
    """Look up a customer's order status by its exact order number (e.g. 'BOAT-2026-001001').

    Returns the order's current status, items, and shipment tracking details.
    Only call this when the customer has provided a specific order number.
    """
    orders = json.loads(ORDERS_PATH.read_text())
    print(f"get_order_status called with order_number={order_number!r}")

    for order in orders:
        if order["order_number"] == order_number:
            result = {
                "status": order["status"],
                "items": order["items"],
                "tracking": order["shipment"],
            }
            print(f"get_order_status found: {result}")
            return json.dumps(result)

    print("get_order_status: no matching order found")
    return f"No order found with order number '{order_number}'."
