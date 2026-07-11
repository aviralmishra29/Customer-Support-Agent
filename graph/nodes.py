"""Node functions for the support-agent graph.

Each node takes the current State and returns a dict of the fields it updates -
never mutates State in place, per LangGraph convention.
"""

import json
import re
import sys
from enum import Enum
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
import lc_query
from tools.order_tools import get_order_status

from graph.state import State

CHAT_MODEL = "qwen3:1.7b"

SAFETY_KEYWORD_PATTERN = re.compile(
    r"\b(fire|smoke|explod|burn|spark|injur|shock|melt)\w*\b", re.IGNORECASE
)


class Route(str, Enum):
    PRODUCT_RAG = "PRODUCT_RAG"
    ORDER = "ORDER"
    ESCALATION = "ESCALATION"
    AUTH = "AUTH"
    FALLBACK = "FALLBACK"


class RouterDecision(BaseModel):
    route: Route = Field(description="Which subsystem should handle this user message.")


ROUTER_SYSTEM_PROMPT = """You are a routing classifier for a customer support system.
Given a user message, decide which subsystem should handle it:

- PRODUCT_RAG: questions about products, features, comparisons, recommendations/suggestions
  ("suggest me more like this", "what's a good alternative to X"), or general FAQs/policies.
  This includes messages that praise OR complain about a product using figurative/casual
  language (e.g. calling something "fire", "insane", "trash", or sarcastic complaints about
  quality/performance) as long as the underlying intent is about the product itself, not a
  real safety incident.
- ORDER: questions about an existing order's status, tracking, delivery, or order-specific issues.
- ESCALATION: genuine safety concerns - fire, smoke, injury, explosion, or similar hazards actually
  happening or reported by the user. Do NOT trigger this for figurative language (e.g. "these
  earbuds are fire" as a compliment, or "explosive quality" as sarcasm) - only real safety incidents.
- AUTH: requests to log in, log out, reset password, or verify identity.
- FALLBACK: anything that doesn't clearly fit the above (small talk, unrelated topics, unclear intent).

You may also be told whether a safety-related keyword was detected in the message as a hint -
use it as a signal, not a verdict; read the full message and decide for yourself.
"""


def _last_human_text(state: State) -> str:
    for msg in reversed(state["messages"]):
        msg_type = getattr(msg, "type", None)
        if msg_type == "human":
            return msg.content
    return ""


def _answer_update(answer: str, **extra) -> dict:
    """Every node's answer must land in both `answer` and `messages`
    (as an AIMessage) so the next turn's router/RAG calls can see prior replies."""
    return {"answer": answer, "messages": [AIMessage(content=answer)], **extra}


ORDER_PHRASING_PROMPT = ChatPromptTemplate.from_template("""A customer asked: "{question}"

Here is their order's real status, already looked up - do not invent or change any of these facts,
just phrase them into a short, friendly reply:

Order status: {status}
Items: {items}
Carrier: {carrier}
Tracking number: {tracking_number}
Latest tracking update: {latest_update}

Reply:""")


def router_node(state: State) -> dict:
    question = _last_human_text(state)
    safety_keyword_hit = bool(SAFETY_KEYWORD_PATTERN.search(question))
    print(f"router_node: question={question!r} safety_keyword_hit={safety_keyword_hit}")

    llm = ChatOllama(model=CHAT_MODEL, reasoning=False)
    structured_llm = llm.with_structured_output(RouterDecision)

    user_prompt = f"safety_keyword_hit: {safety_keyword_hit}\nUser message: {question}"
    decision = structured_llm.invoke(
        [("system", ROUTER_SYSTEM_PROMPT), ("human", user_prompt)]
    )
    print(f"router_node decided: {decision.route.value}")

    return {"route": decision.route.value}


def rag_node(state: State) -> dict:
    question = _last_human_text(state)
    print(f"rag_node: question={question!r}")

    vectorstore = lc_query.get_vectorstore()
    category = lc_query.classify_query(question)
    candidates = lc_query.retrieve_candidates(vectorstore, question, category)
    top_results = lc_query.rerank(question, candidates)
    answer = lc_query.generate_answer(question, top_results)

    return _answer_update(answer, retrieved_docs=top_results)


def order_node(state: State) -> dict:
    question = _last_human_text(state)
    print(f"order_node: question={question!r}")

    llm = ChatOllama(model=CHAT_MODEL, reasoning=True)
    llm_with_tools = llm.bind_tools([get_order_status])

    response = llm_with_tools.invoke(question)
    print(f"order_node tool_calls: {response.tool_calls}")

    if not response.tool_calls:
        answer = response.content.strip() or "Could you share your order number so I can look that up?"
        return _answer_update(answer)

    tool_call = response.tool_calls[0]
    tool_result_raw = get_order_status.invoke(tool_call["args"])
    print(f"order_node tool_result: {tool_result_raw}")

    try:
        order_data = json.loads(tool_result_raw)
    except json.JSONDecodeError:
        # get_order_status returns a plain "no order found" string, not JSON, when it can't find a match.
        return _answer_update(tool_result_raw)

    timeline = order_data["tracking"]["timeline"]
    latest_update = timeline[-1] if timeline else {}
    items_summary = ", ".join(
        f"{item['quantity']}x {item['sku']}" for item in order_data["items"]
    )

    phrasing_llm = ChatOllama(model=CHAT_MODEL, reasoning=True)
    phrasing_chain = ORDER_PHRASING_PROMPT | phrasing_llm

    response = phrasing_chain.invoke({
        "question": question,
        "status": order_data["status"],
        "items": items_summary,
        "carrier": order_data["tracking"]["carrier"],
        "tracking_number": order_data["tracking"]["tracking_number"],
        "latest_update": f"{latest_update.get('status', 'n/a')} at {latest_update.get('location', 'n/a')} on {latest_update.get('timestamp', 'n/a')}",
    })
    answer = response.content.strip()
    print(f"order_node phrased answer: {answer!r}")

    return _answer_update(answer)


def escalation_node(state: State) -> dict:
    print("escalation_node: stub - would hand off to a human/safety team")
    answer = (
        "This sounds like a safety concern. I'm connecting you with our safety team "
        "right away - please stop using the product in the meantime."
    )
    return _answer_update(answer)


def auth_node(state: State) -> dict:
    print("auth_node: stub - would run real login/verification flow")
    answer = "Please log in to continue - authentication isn't wired up yet in this build."
    return _answer_update(answer)


def fallback_node(state: State) -> dict:
    print("fallback_node: no clear route matched")
    answer = "I'm not sure how to help with that - could you rephrase your question?"
    return _answer_update(answer)
