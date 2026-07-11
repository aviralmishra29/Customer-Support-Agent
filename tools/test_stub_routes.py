"""Test the ESCALATION and AUTH stub paths through the real compiled graph."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.build import build_graph

TEST_CASES = [
    {
        "label": "real-escalation",
        "message": "My boAt earphone's battery caught fire while charging, I need help immediately!",
        "thread_id": "test-escalation-1",
    },
    {
        "label": "auth-request",
        "message": "I forgot my password, can you help me reset it and log back into my account?",
        "thread_id": "test-auth-1",
    },
]

if __name__ == "__main__":
    graph = build_graph()

    for case in TEST_CASES:
        print(f"\n=== Test case: {case['label']} ===")
        print(f"Message: {case['message']}")
        config = {"configurable": {"thread_id": case["thread_id"]}}
        result = graph.invoke({"messages": [("human", case["message"])]}, config=config)
        print(f"route: {result['route']}")
        print(f"answer: {result['answer']}")

        state = graph.get_state(config)
        print(f"messages in thread: {len(state.values['messages'])}")
        for msg in state.values["messages"]:
            print(f"  [{msg.type}] {msg.content[:80]}")
