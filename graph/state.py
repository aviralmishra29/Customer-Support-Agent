"""Shared State schema that flows through every node in the support-agent graph."""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: Optional[str]
    is_authenticated: bool
    retrieved_docs: Optional[list]
    answer: Optional[str]
