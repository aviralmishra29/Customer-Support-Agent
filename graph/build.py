"""Assembles and compiles the support-agent StateGraph."""

import sqlite3
from pathlib import Path

import httpx
from chromadb.errors import ChromaError
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from graph.edges import route_after_router, route_from_start
from graph.nodes import (
    _answer_update,
    auth_node,
    escalation_node,
    fallback_node,
    order_node,
    rag_node,
    router_node,
)
from graph.state import State

# MemorySaver (in-RAM only) replaced with SqliteSaver - a conversation's checkpointed state
# (messages, auth status, everything) now survives a process restart. Decided alongside the
# local-only deploy target: this needs single-machine durability, not multi-replica sharing,
# so SqliteSaver is the right fit (PostgresSaver would be for a distributed deployment this
# project explicitly isn't doing). check_same_thread=False matches langgraph's own
# from_conn_string() usage - needed since FastAPI will later call this from multiple requests.
CHECKPOINT_DB_PATH = Path(__file__).parent.parent / "data" / "checkpoints" / "checkpoints.db"

# Infra failures a node can hit (Ollama unreachable/timed out, Chroma unreachable, SQLite
# locked/missing) that are expected and anticipated - caught specifically so the real
# exception is logged distinctly from a genuine, unexpected bug. Verified against the project's
# own data (tests/test_infra_failure_baseline.py), not guessed: ChatOllama raises httpx errors,
# Chroma raises chromadb.errors.ChromaError subclasses, SQLite raises sqlite3.Error subclasses
# for both a missing file AND real lock contention.
KNOWN_INFRA_ERRORS = (httpx.ConnectError, httpx.TimeoutException, ChromaError, sqlite3.Error)

INFRA_FALLBACK_ANSWER = "I'm having trouble processing that right now - please try again in a moment."


def _safe_node(node_fn, name):
    """Wraps a node so an infra failure (or any other unexpected exception) degrades to a safe
    user-facing answer instead of crashing the whole graph.invoke() call. One wrapper applied
    uniformly at registration time, rather than try/except scattered through every node's
    internals - same reasoning as classify_query/router using one shared fix instead of N
    one-off patches."""
    def wrapped(state):
        try:
            return node_fn(state)
        except KNOWN_INFRA_ERRORS as e:
            print(f"[{name}] infra failure: {type(e).__module__}.{type(e).__name__}: {e}")
            return _answer_update(INFRA_FALLBACK_ANSWER)
        except Exception as e:
            # Last-resort catch-all: anything genuinely unexpected still degrades safely
            # rather than crashing - but this is logged loudly since it's NOT an anticipated
            # case the way the 4 types above are, and likely signals a real bug to fix.
            print(f"[{name}] UNEXPECTED failure (not a known infra error): {type(e).__module__}.{type(e).__name__}: {e}")
            return _answer_update(INFRA_FALLBACK_ANSWER)
    return wrapped


def build_graph():
    builder = StateGraph(State)

    builder.add_node("router_node", _safe_node(router_node, "router_node"))
    builder.add_node("rag_node", _safe_node(rag_node, "rag_node"))
    builder.add_node("order_node", _safe_node(order_node, "order_node"))
    builder.add_node("escalation_node", _safe_node(escalation_node, "escalation_node"))
    builder.add_node("auth_node", _safe_node(auth_node, "auth_node"))
    builder.add_node("fallback_node", _safe_node(fallback_node, "fallback_node"))

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

    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)
    return graph
