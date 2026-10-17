"""Verifies Phoenix tracing actually captures the graph's real structure - not just that
ChatOllama calls show up as spans, but that graph node boundaries (router_node, rag_node, etc.)
are visible too. Requires a local Phoenix server already running (see observability.py).

This is purely to generate ONE real traced request and manually inspect the result in the
Phoenix UI (http://localhost:6006) - not an automated assertion, since "does this look right
in the trace tree" needs a human looking at the actual UI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from observability import setup_tracing

setup_tracing()

from graph.build import build_graph

if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "test-observability-1"}}

    print("\n=== Sending one real product question through the traced graph ===")
    result = graph.invoke(
        {"messages": [("human", "Which earbuds have good battery life?")]},
        config=config,
    )
    print(f"route: {result['route']}")
    print(f"answer: {result['answer']}")
    print("\nNow open http://localhost:6006 in a browser and check the trace for this request.")
