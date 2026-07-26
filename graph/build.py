"""Assembles and compiles the support-agent StateGraph."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph.edges import route_after_router, route_from_start
from graph.nodes import (
    auth_node,
    escalation_node,
    fallback_node,
    order_node,
    rag_node,
    router_node,
)
from graph.state import State


def build_graph():
    builder = StateGraph(State)

    builder.add_node("router_node", router_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("order_node", order_node)
    builder.add_node("escalation_node", escalation_node)
    builder.add_node("auth_node", auth_node)
    builder.add_node("fallback_node", fallback_node)

    builder.add_conditional_edges(
        START,
        route_from_start,
        {
            "router_node": "router_node",
            "auth_node": "auth_node",
        },
    )
    builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "rag_node": "rag_node",
            "order_node": "order_node",
            "escalation_node": "escalation_node",
            "auth_node": "auth_node",
            "fallback_node": "fallback_node",
        },
    )
    builder.add_edge("rag_node", END)
    builder.add_edge("order_node", END)
    builder.add_edge("escalation_node", END)
    builder.add_edge("auth_node", END)
    builder.add_edge("fallback_node", END)

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph
