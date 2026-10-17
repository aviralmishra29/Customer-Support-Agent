"""Verifies generate_answer's new numeric groundedness check:
1. On real product questions, legitimate answers should pass through unchanged (no false
   positives from incidental/stylistic numbers like "here are 3 options").
2. A directly-crafted fabricated answer (a number not in the context) should correctly trigger
   PRODUCT_UNGROUNDED_FALLBACK.

Real product questions are run through the actual pipeline to see if the check ever misfires
on normal, legitimate output - this is the main risk flagged before building it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import lc_query

REAL_QUESTIONS = [
    "Which earbuds have good battery life?",
    "Tell me about the boAt Rockerz 450",
    "What smartwatches do you have with AMOLED displays?",
    "Recommend a waterproof speaker",
]

if __name__ == "__main__":
    vs = lc_query.get_vectorstore()

    print("=== Part 1: real questions through the full pipeline - watching for false positives ===")
    for q in REAL_QUESTIONS:
        print(f"\n--- {q!r} ---")
        category = lc_query.classify_query(q)
        candidates = lc_query.retrieve_candidates(vs, q, category)
        top = lc_query.rerank(q, candidates)
        answer = lc_query.generate_answer(q, top)
        print(f"FINAL ANSWER: {answer!r}")
        print(f"fallback triggered: {answer == lc_query.PRODUCT_UNGROUNDED_FALLBACK}")

    print("\n=== Part 2: directly testing the check logic with a fabricated number ===")
    fake_context_chunks = [type("C", (), {"page_content": "boAt Rockerz 450 has 40h battery life, priced at Rs. 1999."})()]
    context_numbers = lc_query._extract_numbers(fake_context_chunks[0].page_content)
    fake_answer_numbers = lc_query._extract_numbers("The Rockerz 450 actually has 99h of battery life!")
    fabricated = fake_answer_numbers - context_numbers
    print(f"context numbers: {context_numbers}")
    print(f"fake answer numbers: {fake_answer_numbers}")
    print(f"fabricated (should be non-empty, e.g. contains '99'): {fabricated}")
