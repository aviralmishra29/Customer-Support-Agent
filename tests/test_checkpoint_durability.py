"""Verifies SqliteSaver actually persists across process restarts - not just that a .db file
gets created, but that a FRESH Python process (simulating a server restart) can read back a
conversation's full state without re-running anything.

Run with --write first (starts a conversation, writes real state), then run with --read in a
SEPARATE process invocation (simulating the app restarting) to confirm the state is still
there. A single script run can't prove this by itself - MemorySaver would also "work" within
one process; the whole point is proving it survives BETWEEN processes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.build import build_graph

THREAD_ID = "test-durability-thread"

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode not in ("--write", "--read"):
        print("Usage: python test_checkpoint_durability.py --write | --read")
        sys.exit(1)

    graph = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    if mode == "--write":
        print("=== WRITE: starting a real conversation in this process ===")
        result = graph.invoke(
            {"messages": [("human", "Which earbuds have good battery life?")]},
            config=config,
        )
        print(f"route: {result['route']}")
        print(f"answer: {result['answer']}")
        state = graph.get_state(config)
        print(f"\nState written. Message count: {len(state.values['messages'])}")
        print("Now run this script again with --read (a genuinely separate process invocation).")

    elif mode == "--read":
        print("=== READ: fresh process, reading back state with NO new invoke() call ===")
        state = graph.get_state(config)
        if not state.values:
            print("FAIL: no state found - durability did not work")
            sys.exit(1)
        print(f"State found. Message count: {len(state.values['messages'])}")
        for msg in state.values["messages"]:
            print(f"  [{msg.type}] {msg.content[:80]}")
        print("\nPASS: state survived into a fresh process without re-invoking the graph.")
