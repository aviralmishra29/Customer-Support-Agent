"""
Isolated timing test: same prompts/model as test_router_structured_output.py,
but a plain ChatOllama.invoke() with no .with_structured_output() binding,
to check whether structured output (tool-call binding under the hood) is
what's inflating latency, or if think:false calls are just this slow now.
"""

import time

from langchain_ollama import ChatOllama

SYSTEM_PROMPT = """You are a routing classifier for a customer support system.
Reply with exactly one word: PRODUCT_RAG, ORDER, ESCALATION, AUTH, or FALLBACK.
"""

MESSAGES = [
    "These boAt Rockerz earbuds are so good, so goated that they really feel like fire. Please suggest me few more like this.",
    "Where is my order #12345, it was supposed to arrive yesterday?",
    "My boAt earphone's battery caught fire while charging, I need help immediately!",
    "What's the weather like today?",
]


def main():
    print("Loading plain ChatOllama(qwen3:1.7b, think=False), no structured output...")
    llm = ChatOllama(model="qwen3:1.7b", think=False)

    durations = []
    for msg in MESSAGES:
        print(f"\n--- Message: {msg} ---")
        start = time.perf_counter()
        result = llm.invoke([("system", SYSTEM_PROMPT), ("human", msg)])
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"Raw content: {result.content!r}")
        print(f"Time taken: {elapsed:.2f}s")

    total = sum(durations)
    avg = total / len(durations)
    print(f"\n=== Timing summary: {len(durations)} calls, total {total:.2f}s, avg {avg:.2f}s/call ===")


if __name__ == "__main__":
    main()
