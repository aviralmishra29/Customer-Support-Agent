"""Conditional-edge routing logic. Reads State, returns the next node's name."""

from graph.state import State

ROUTE_TO_NODE = {
    "PRODUCT_RAG": "rag_node",
    "ORDER": "order_node",
    "ESCALATION": "escalation_node",
    "AUTH": "auth_node",
    "FALLBACK": "fallback_node",
}


def route_after_router(state: State) -> str:
    route = state["route"]
    node_name = ROUTE_TO_NODE.get(route, "fallback_node")
    print(f"route_after_router: route={route!r} -> node={node_name!r}")
    return node_name
