"""Verifies: when classify_query returns "unsure", rag_node declines early (a clarifying
question) instead of running retrieve_candidates/rerank/generate_answer - both for correctness
(don't guess on low-confidence input) and for cost (skips the most expensive part of the
pipeline, the exact thing that caused a 192s confused-reasoning loop in rerank during
adversarial testing).

Picks messages specifically crafted to still pass the router (so they reach rag_node at all)
but be genuinely ambiguous once there - this is deliberately a different case shape than the
prompt-injection/off-topic tests, which mostly get caught one layer earlier now."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.build import build_graph
import lc_query

AMBIGUOUS_CASES = [
    "I need something",
    "Is it good?",
    "What about the thing I asked before but different",
]

if __name__ == "__main__":
    print("=== Direct classify_query check on ambiguous inputs ===")
    for q in AMBIGUOUS_CASES:
        category = lc_query.classify_query(q)
        print(f"{q!r} -> {category!r}")

    print("\n=== Full graph: timing check on whichever case actually lands on 'unsure' ===")
    graph = build_graph()
    for i, q in enumerate(AMBIGUOUS_CASES):
        config = {"configurable": {"thread_id": f"test-unsure-{i}"}}
        start = time.perf_counter()
        result = graph.invoke({"messages": [("human", q)]}, config=config)
        elapsed = time.perf_counter() - start
        print(f"\nquestion: {q!r}")
        print(f"route: {result['route']}  wall_time: {elapsed:.2f}s")
        print(f"answer: {result['answer']}")
