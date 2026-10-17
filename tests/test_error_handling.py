"""Verifies the structured error handling added in graph/build.py and lc_query.py:
1. _safe_node catches each of the 4 known infra exception types (confirmed real via
   test_infra_failure_baseline.py) plus a generic unexpected exception, and degrades to
   the safe INFRA_FALLBACK_ANSWER instead of crashing.
2. invoke_with_retry retries on httpx.ConnectError (fast-failing, worth a quick retry) but
   does NOT retry on httpx.TimeoutException (already waited the full timeout once - retrying
   would double the wait for an unlikely-to-change outcome).

Uses fake invocables that raise on demand rather than actually taking down Ollama/Chroma/
SQLite - the real exception types were already confirmed empirically in
test_infra_failure_baseline.py; this test verifies OUR code reacts to them correctly."""

import sqlite3
import sys
from pathlib import Path

import httpx
from chromadb.errors import InternalError

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.build import _safe_node, INFRA_FALLBACK_ANSWER
from lc_query import invoke_with_retry

if __name__ == "__main__":
    print("=== Part 1: _safe_node catches every known infra error type ===")
    exceptions_to_test = [
        httpx.ConnectError("connection refused"),
        httpx.TimeoutException("timed out"),
        InternalError("chroma down"),
        sqlite3.OperationalError("database is locked"),
        RuntimeError("some genuinely unexpected bug"),  # not a known infra type
    ]
    for exc in exceptions_to_test:
        def failing_node(state, _exc=exc):
            raise _exc

        wrapped = _safe_node(failing_node, "test_node")
        result = wrapped({})
        passed = result["answer"] == INFRA_FALLBACK_ANSWER
        print(f"{type(exc).__name__:25s} -> caught={passed}  answer={result['answer']!r}")

    print("\n=== Part 2: invoke_with_retry - retries ConnectError, does NOT retry TimeoutException ===")

    call_count = {"n": 0}

    def flaky_then_succeeds(x):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return f"succeeded on attempt {call_count['n']}"

    call_count["n"] = 0
    result = invoke_with_retry(flaky_then_succeeds, "arg")
    print(f"ConnectError-then-success: result={result!r} total_calls={call_count['n']} (expect 2)")

    def always_times_out(x):
        call_count["n"] += 1
        raise httpx.TimeoutException("timed out")

    call_count["n"] = 0
    try:
        invoke_with_retry(always_times_out, "arg")
        print("ERROR: expected TimeoutException to propagate, it didn't")
    except httpx.TimeoutException:
        print(f"TimeoutException: correctly NOT retried, total_calls={call_count['n']} (expect 1)")
