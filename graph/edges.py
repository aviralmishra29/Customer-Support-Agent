"""Conditional-edge routing logic. Reads State, returns the next node's name."""

from graph.nodes import is_authenticated
from graph.state import State

ROUTE_TO_NODE = {
    "PRODUCT_RAG": "rag_node",
    "ORDER": "order_node",
    "ESCALATION": "escalation_node",
    "AUTH": "auth_node",
    "FALLBACK": "fallback_node",
}

# Routes that require a verified identity before their node is allowed to run.
# Enforced here, in code, not inside the node and never as something the LLM decides.
AUTH_REQUIRED_ROUTES = {"ORDER"}


def route_from_start(state: State) -> str:
    """A stateful sub-flow (auth_node's email -> OTP exchange) must resume directly,
    not be reclassified by the router each turn - the router has no notion of
    conversation stage, only of the current message's content in isolation."""
    if state.get("auth_stage"):
        print(f"route_from_start: mid-auth-flow (auth_stage={state['auth_stage']!r}), bypassing router -> auth_node")
        return "auth_node"
    return "router_node"


def route_after_router(state: State) -> str:
    route = state["route"]

    if route in AUTH_REQUIRED_ROUTES and not is_authenticated(state):
        print(f"route_after_router: route={route!r} requires auth, not authenticated -> auth_node")
        return "auth_node"

    node_name = ROUTE_TO_NODE.get(route, "fallback_node")
    print(f"route_after_router: route={route!r} -> node={node_name!r}")
    return node_name
