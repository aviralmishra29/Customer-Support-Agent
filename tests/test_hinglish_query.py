"""Test the full LangChain pipeline against Hindi/Hinglish queries, using real corpus content."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from lc_query import classify_query, generate_answer, get_vectorstore, rerank, retrieve_candidates

TEST_QUESTIONS = [
    "Mera order kaise track karun?",  # matches an indexed Hinglish FAQ closely
    "Mera order abhi tak nahi aaya, kaise pata karu?",  # similar intent, fresh phrasing, not indexed verbatim
]

if __name__ == "__main__":
    vectorstore = get_vectorstore()

    for question in TEST_QUESTIONS:
        print(f"\n{'=' * 60}")
        print(f"Query: {question}")

        category = classify_query(question)
        candidates = retrieve_candidates(vectorstore, question, category)

        print("Top candidates:")
        for i, c in enumerate(candidates):
            print(f"  {i + 1}. {c.page_content[:90]}")

        top_results = rerank(question, candidates)
        answer = generate_answer(question, top_results)

        print(f"\nFinal answer: {answer}")
