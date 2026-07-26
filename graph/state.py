"""Shared State schema that flows through every node in the support-agent graph."""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: Optional[str]
    retrieved_docs: Optional[list]
    answer: Optional[str]

    # Auth: verified identity + expiry, not a bare bool - see docs/notes/langgraph/04-graph-enforced-auth.md
    user_id: Optional[str]
    auth_expires_at: Optional[float]

    # Auth flow's own conversational progress (email collected, awaiting OTP, etc.)
    auth_stage: Optional[str]
    pending_email: Optional[str]
    pending_otp: Optional[str]
