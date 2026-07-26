"""
Test whether qwen3:1.7b + LangChain's .with_structured_output() reliably
returns a schema-conformant route decision, before we build the real
graph-level router node on top of this mechanism.
"""

import time
from enum import Enum

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class Route(str, Enum):
    PRODUCT_RAG = "PRODUCT_RAG"
    ORDER = "ORDER"
    ESCALATION = "ESCALATION"
    AUTH = "AUTH"
    FALLBACK = "FALLBACK"


class RouterDecision(BaseModel):
    route: Route = Field(
        description="Which subsystem should handle this user message."
    )


SYSTEM_PROMPT = """You are a routing classifier for a customer support system.
Given a user message, decide which subsystem should handle it:

- PRODUCT_RAG: questions about products, features, comparisons, recommendations/suggestions
  ("suggest me more like this", "what's a good alternative to X"), or general FAQs/policies.
  This includes messages that praise OR complain about a product using figurative/casual
  language (e.g. calling something "fire", "insane", "trash", or sarcastic complaints about
  quality/performance) as long as the underlying intent is about the product itself, not a
  real safety incident.
  Example: "Wow my new earbuds died in literally one day, great job boAt, truly explosive
  quality." -> PRODUCT_RAG (a sarcastic complaint about battery/quality - not a real hazard,
  and not "unclear intent" either - it's clearly about this product's performance).
- ORDER: questions about an existing order's status, tracking, delivery, or order-specific issues.
- ESCALATION: genuine safety concerns - fire, smoke, injury, explosion, or similar hazards actually
  happening or reported by the user. Do NOT trigger this for figurative language (e.g. "these
  earbuds are fire" as a compliment) - only real safety incidents.
- AUTH: requests to log in, log out, reset password, or verify identity.
- FALLBACK: anything that doesn't clearly fit the above (small talk, unrelated topics, unclear intent).

You may also be told whether a safety-related keyword was detected in the message as a hint -
use it as a signal, not a verdict; read the full message and decide for yourself.
"""

TEST_CASES = [
    {
        "label": "earbuds-figurative-fire (adversarial)",
        "message": (
            "These boAt Rockerz earbuds are so good, so goated that they really feel "
            "like fire. Please suggest me few more like this."
        ),
        "safety_keyword_hit": True,
    },
    {
        "label": "obvious-order-status",
        "message": "Where is my order #12345, it was supposed to arrive yesterday?",
        "safety_keyword_hit": False,
    },
    {
        "label": "real-escalation",
        "message": "My boAt earphone's battery caught fire while charging, I need help immediately!",
        "safety_keyword_hit": True,
    },
    {
        "label": "fallback-offtopic",
        "message": "What's the weather like today?",
        "safety_keyword_hit": False,
    },
    {
        "label": "auth-request",
        "message": "I forgot my password, can you help me reset it and log back into my account?",
        "safety_keyword_hit": False,
    },
    {
        "label": "blended-intent (order + product)",
        "message": (
            "My order #778 hasn't arrived yet, and also is the Airdopes 141 good for "
            "gym use or should I look at something else?"
        ),
        "safety_keyword_hit": False,
    },
    {
        "label": "hinglish-product-question",
        "message": "Yeh headphone ka battery kitni der chalta hai, gym ke liye sahi rahega?",
        "safety_keyword_hit": False,
    },
    {
        "label": "hedged-safety-complaint (ambiguous, no keyword hit)",
        "message": "My earphones got really hot yesterday while charging, is that normal or should I be worried?",
        "safety_keyword_hit": False,
    },
    {
        "label": "sarcastic-complaint (figurative, negative sentiment)",
        "message": "Wow my new earbuds died in literally one day, great job boAt, truly explosive quality.",
        "safety_keyword_hit": True,
    },
]


def main():
    print("Loading ChatOllama(qwen3:1.7b) with structured output bound to RouterDecision...")
    llm = ChatOllama(model="qwen3:1.7b", think=False)
    structured_llm = llm.with_structured_output(RouterDecision)

    durations = []

    for case in TEST_CASES:
        print(f"\n--- Test case: {case['label']} ---")
        print(f"Message: {case['message']}")
        print(f"safety_keyword_hit hint: {case['safety_keyword_hit']}")

        user_prompt = (
            f"safety_keyword_hit: {case['safety_keyword_hit']}\n"
            f"User message: {case['message']}"
        )

        start = time.perf_counter()
        try:
            result = structured_llm.invoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", user_prompt),
                ]
            )
            elapsed = time.perf_counter() - start
            durations.append(elapsed)
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")
            print(f"route value: {result.route} (enum member: {result.route.name})")
            print(f"Time taken: {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.perf_counter() - start
            durations.append(elapsed)
            print(f"FAILED with exception: {type(e).__name__}: {e}")
            print(f"Time taken (before failure): {elapsed:.2f}s")

    total = sum(durations)
    avg = total / len(durations) if durations else 0
    print(f"\n=== Timing summary: {len(durations)} calls, total {total:.2f}s, avg {avg:.2f}s/call ===")


if __name__ == "__main__":
    main()
