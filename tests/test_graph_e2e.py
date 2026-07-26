"""End-to-end test of the compiled support-agent graph, including multi-turn memory
via a fixed thread_id across two calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.build import build_graph

if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "test-thread-1"}}

    print("=== Turn 1: product question ===")
    result = graph.invoke(
        {"messages": [("human", "Which earbuds have good battery life?")]},
        config=config,
    )
    print(f"route: {result['route']}")
    print(f"answer: {result['answer']}")

    print("\n=== Turn 2: order question, same thread ===")
    result = graph.invoke(
        {"messages": [("human", "What's the status of my order BOAT-2026-001001?")]},
        config=config,
    )
    print(f"route: {result['route']}")
    print(f"answer: {result['answer']}")

    print("\n=== Turn 3: check accumulated message history in this thread ===")
    state = graph.get_state(config)
    print(f"Total messages in thread: {len(state.values['messages'])}")
    for msg in state.values["messages"]:
        print(f"  [{msg.type}] {msg.content[:80]}")
