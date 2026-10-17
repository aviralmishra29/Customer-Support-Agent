"""Verifies the router fix for the ORDER-vs-PRODUCT_RAG ambiguity found by evals/run_evals.py:
a general return-policy question (no order referenced) was misrouted to ORDER. Checks both
the fixed case AND that real ORDER questions still correctly route to ORDER - a fix here must
not just move the ambiguity the other direction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.build import build_graph

CASES = [
    ("policy-no-order-ref", "How long do I have to return a defective product?", "PRODUCT_RAG"),
    ("policy-generic-return-window", "What's your return policy?", "PRODUCT_RAG"),
    ("real-order-status", "Where is my order #12345, it was supposed to arrive yesterday?", "ORDER"),
    ("real-order-return", "I want to return my order BOAT-2026-001001, how do I do that?", "ORDER"),
]

if __name__ == "__main__":
    graph = build_graph()
    all_passed = True

    for label, question, expected in CASES:
        config = {"configurable": {"thread_id": f"test-order-vs-policy-{label}"}}
        result = graph.invoke({"messages": [("human", question)]}, config=config)
        route = result["route"]
        passed = route == expected
        all_passed = all_passed and passed
        print(f"{'PASS' if passed else 'FAIL'}  {label}: expected={expected} got={route}  question={question!r}")

    print(f"\n{'ALL PASSED' if all_passed else 'SOME FAILED'}")
