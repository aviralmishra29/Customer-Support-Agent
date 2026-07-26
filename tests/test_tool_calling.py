"""Test whether ChatOllama.bind_tools() correctly decides tool-vs-direct-answer across 3 cases."""

import sys
from pathlib import Path

from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.order_tools import get_order_status

CHAT_MODEL = "qwen3:1.7b"

TEST_CASES = [
    "What's the status of my order BOAT-2026-001001?",
    "Which earbuds have good battery life?",
    "I placed an order yesterday, when will it arrive?",
]

if __name__ == "__main__":
    llm = ChatOllama(model=CHAT_MODEL, reasoning=True)
    llm_with_tools = llm.bind_tools([get_order_status])
    print(f"Bound tools to {CHAT_MODEL}: [get_order_status]")

    for i, question in enumerate(TEST_CASES, start=1):
        print(f"\n=== Case {i}: {question!r} ===")
        response = llm_with_tools.invoke(question)
        print(f"reasoning: {response.additional_kwargs.get('reasoning_content', '')!r}")
        print(f"content: {response.content!r}")
        print(f"tool_calls: {response.tool_calls}")
