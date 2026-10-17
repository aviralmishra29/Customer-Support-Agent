"""Homegrown, fully-local eval harness - no cloud dashboard (LangSmith), nothing new to install.

Loads evals/eval_set.json, runs each case through the real compiled graph, and scores it:
- "route" cases: exact match against expected_route (deterministic, like the router's own
  structured output - no ambiguity to score).
- "content" cases: rubric-style must_mention/must_not_contain checks via plain case-insensitive
  substring matching, NOT an LLM-as-judge call. This is a deliberate choice, not a shortcut:
  every rubric item in this eval set is a literal, checkable fact (a number, a word like
  "rephrase", a banned phrase) - the same "trust code over a model's self-report" principle
  already used everywhere else in this project (auth, order-number validation, groundedness
  checks) applies here too. An LLM judge would add a new source of unreliability to verify
  something a plain string check can verify exactly.

Cases marked "known_issue" are allowed to fail without counting as a regression - they track
a documented, already-logged gap (see docs/TODO.md), not a surprise.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.build import build_graph

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def score_case(case: dict, route: str, answer: str) -> tuple[bool, str]:
    kind = case["kind"]

    if kind in ("route", "route_and_content"):
        if route != case["expected_route"]:
            return False, f"route mismatch: expected {case['expected_route']!r}, got {route!r}"

    if kind in ("content", "route_and_content"):
        for phrase in case.get("must_mention", []):
            if not _contains(answer, phrase):
                return False, f"missing required phrase: {phrase!r}"
        for phrase in case.get("must_not_contain", []):
            if _contains(answer, phrase):
                return False, f"contains banned phrase: {phrase!r}"

    return True, "ok"


def main():
    cases = json.loads(EVAL_SET_PATH.read_text())
    print(f"Loaded {len(cases)} eval cases from {EVAL_SET_PATH}")

    graph = build_graph()

    results = []
    for i, case in enumerate(cases, start=1):
        print(f"\n=== [{i}/{len(cases)}] {case['id']} ({case['category']}) ===")
        print(f"question: {case['question']!r}")

        config = {"configurable": {"thread_id": f"eval-{case['id']}"}}
        result = graph.invoke({"messages": [("human", case["question"])]}, config=config)
        route = result.get("route")
        answer = result.get("answer", "")
        print(f"route: {route}")
        print(f"answer: {answer!r}")

        passed, reason = score_case(case, route, answer)
        is_known_issue = bool(case.get("known_issue"))
        print(f"result: {'PASS' if passed else 'FAIL'} ({reason}){' [KNOWN ISSUE]' if is_known_issue and not passed else ''}")

        results.append({
            "id": case["id"],
            "category": case["category"],
            "passed": passed,
            "reason": reason,
            "known_issue": is_known_issue,
        })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    categories = sorted({r["category"] for r in results})
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        passed_count = sum(1 for r in cat_results if r["passed"])
        print(f"{cat:20s} {passed_count}/{len(cat_results)} passed")

    total_passed = sum(1 for r in results if r["passed"])
    unexpected_failures = [r for r in results if not r["passed"] and not r["known_issue"]]
    known_failures = [r for r in results if not r["passed"] and r["known_issue"]]

    print(f"\nTOTAL: {total_passed}/{len(results)} passed")
    if known_failures:
        print(f"Known-issue failures (tracked, not a regression): {[r['id'] for r in known_failures]}")
    if unexpected_failures:
        print(f"UNEXPECTED failures (real regressions to investigate): {[r['id'] for r in unexpected_failures]}")
    else:
        print("No unexpected failures.")


if __name__ == "__main__":
    main()
