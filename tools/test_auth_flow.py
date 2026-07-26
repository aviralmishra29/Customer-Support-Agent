"""End-to-end test of graph-enforced auth: an unauthenticated ORDER request must be
gated to auth_node by the conditional edge (not by order_node itself), then the mock
OTP flow should authenticate the thread, after which the same ORDER request should
go through to order_node."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.build import build_graph

if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "test-auth-flow-1"}}

    print("=== Turn 1: ask for order status while NOT authenticated ===")
    result = graph.invoke(
        {"messages": [("human", "What's the status of my order BOAT-2026-001001?")]},
        config=config,
    )
    print(f"route: {result['route']}")
    print(f"answer: {result['answer']}")
    assert result["route"] == "ORDER", "router should still classify this as ORDER"

    state = graph.get_state(config)
    print(f"auth_stage after turn 1: {state.values.get('auth_stage')!r}")
    assert state.values.get("auth_stage") == "awaiting_email", "edge should have redirected to auth_node, which should ask for email (not parse the triggering message as one)"

    print("\n=== Turn 2: provide email ===")
    result = graph.invoke(
        {"messages": [("human", "my email is test.user@example.com")]},
        config=config,
    )
    print(f"answer: {result['answer']}")

    state = graph.get_state(config)
    pending_otp = state.values.get("pending_otp")
    pending_email = state.values.get("pending_email")
    print(f"(test harness reading state directly, mock only) pending_email={pending_email!r} pending_otp={pending_otp!r}")
    assert pending_otp is not None, "auth_node should have generated a mock OTP"
    assert pending_email == "test.user@example.com", f"expected the actual email to be captured, got {pending_email!r}"

    print("\n=== Turn 3: provide the (correct) OTP ===")
    result = graph.invoke(
        {"messages": [("human", f"the code is {pending_otp}")]},
        config=config,
    )
    print(f"answer: {result['answer']}")

    state = graph.get_state(config)
    print(f"user_id after verification: {state.values.get('user_id')!r}")
    print(f"auth_expires_at after verification: {state.values.get('auth_expires_at')!r}")
    assert state.values.get("user_id") == "test.user@example.com", f"expected user_id to be the real email, got {state.values.get('user_id')!r}"

    print("\n=== Turn 4: repeat the order question, now authenticated ===")
    result = graph.invoke(
        {"messages": [("human", "What's the status of my order BOAT-2026-001001?")]},
        config=config,
    )
    print(f"route: {result['route']}")
    print(f"answer: {result['answer']}")
    assert "delivered" in result["answer"].lower() or "order" in result["answer"].lower(), \
        "should now reach order_node and get a real order answer"

    print("\n=== All assertions passed ===")
