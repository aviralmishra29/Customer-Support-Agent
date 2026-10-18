# Eval Results

How the eval set is shaped, what each run actually found, and what that proves (and doesn't).
See `docs/design-decisions.md` §11 for *why* it's built this way (homegrown/local, deterministic
scoring); this doc is about what came out of running it.

## The eval set — 21 cases, consolidated from real work, not invented fresh

| Category | Cases | What it checks |
|---|---|---|
| `routing` | 9 | Does the graph-level router pick the right subsystem? Reused from `tests/test_router_structured_output.py`'s original 9 adversarial messages — but now run through the *full* graph with the current prompt, not just a schema-conformance check like the original test did. |
| `guardrail` | 6 | Does off-topic/injection input get caught at `FALLBACK` before reaching the expensive RAG pipeline? Reused from the Weekend 4 injection-baseline work. |
| `unsure_handling` | 2 | Does genuinely ambiguous input get a clarifying question instead of a forced guess? |
| `rag_quality` | 2 | Does a real product answer state the correct facts (battery life, specs)? |
| `policy_quality` | 1 | Does a policy answer correctly state the return window? |
| `comparison_quality` | 1 | Does a product comparison state correct numbers for both products? |

**Scoring, concretely:** `route` cases check exact equality against an `expected_route` string.
`content` cases check `must_mention`/`must_not_contain` phrases via plain case-insensitive
substring matching — no LLM grades anything in this harness. See `evals/run_evals.py`'s
`score_case()` for the actual ~15 lines of logic; it really is that simple, deliberately.

## Run 1 — 20/21 passed

```
comparison_quality   1/1 passed
guardrail            6/6 passed
policy_quality       0/1 passed   <-- the only failure
rag_quality          2/2 passed
routing              9/9 passed
unsure_handling      2/2 passed
```

**The one failure was a genuinely new bug, not a flaky test.** `quality-policy-return-window`
("How long do I have to return a defective product?") was misrouted by the graph-level router
to `ORDER` instead of `PRODUCT_RAG` — the question never reached the policy-answering logic at
all; it got redirected to `auth_node` since `ORDER` requires login. **No test before this one
could have caught it**, because every prior policy-grounding test called `lc_query.classify_query()`
directly, bypassing the graph-level router entirely. This is the eval harness's first real run
earning its keep immediately.

**Root cause, fixed the same day:** `ORDER`'s router-prompt description was too permissive
("order-specific issues"), and `PRODUCT_RAG`'s brief "general FAQs/policies" mention wasn't
strong enough to win against return-shaped phrasing. Fix: both descriptions were tightened with
concrete examples (see `docs/design-decisions.md` for the reasoning), verified in both
directions with a dedicated test (`tests/test_order_vs_policy_routing.py`) — including the
hardest case, a message containing both "return" *and* a real order number, which still
correctly routed to `ORDER` after the fix, proving it didn't just flip the ambiguity the other
way.

## Run 2 (after the router fix) — 21/21 passed

```
comparison_quality   1/1 passed
guardrail            6/6 passed
policy_quality       1/1 passed   <-- now passing
rag_quality          2/2 passed
routing              9/9 passed
unsure_handling      2/2 passed
```

`policy_quality` went from 0/1 to 1/1 — direct confirmation the fix holds under the full 21-case
suite, not just the 4 targeted cases in the dedicated router test.

## What this does and doesn't prove

**What it proves:** the specific 21 behaviors this set checks for are currently correct, as of
the last run. It caught one real, non-obvious bug on its very first execution.

**What it doesn't prove — worth being honest about in an interview:**
- 21 cases is a thin slice of the actual possible input space. It's a *regression* suite (did
  something we already know about break), not a comprehensive correctness guarantee.
- One of the 9 routing cases (`route-sarcastic-complaint`) tracks a long-documented open bug
  (sarcasm misclassification, since Weekend 2) that **passed on both runs so far** — but that's
  not treated as "fixed." Two passes of a non-deterministic LLM aren't proof; this is logged in
  `docs/TODO.md` as "to recheck," not closed. The honest answer to "is that bug fixed?" is "it
  hasn't reproduced in 2 runs, which is weak evidence, not confirmation."
- The eval set checks *content correctness*, not *latency*. Separately (see
  `docs/cost-and-latency.md`), aggregating every timing log — not just reading live eval
  summaries — surfaced that the exact `quality-policy-return-window` case that exposed the
  router bug also took **719.18 seconds** to generate its (correct) answer during Run 2, and
  nobody noticed at the time because the eval only checks whether the answer is right, not how
  long it took. Correctness and latency are different dimensions, and this is a concrete
  example of a case passing one while silently failing the other.
- The set has no case yet for the Hinglish-generation gap, the blended-intent limitation, or
  the two latency outliers above — all known, documented gaps that simply aren't represented in
  the eval set yet (see `docs/TODO.md`'s deferred/open section for the full list).
