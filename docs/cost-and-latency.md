# Cost & Latency

Real numbers, measured from this project's own logs and traces across Weekends 1–5 — not
estimated or assumed. Source: every `[timing]` line across `logs/*.log` (excluding
`logs/query.log`, which is from the original July 5 Weekend 1 zero-framework testing — a
different session, different time period, not representative of the current pipeline) plus
direct inspection of real traces in the Phoenix UI (see `docs/notes/langgraph/03-building-and-
debugging-the-graph.md` for the original latency investigation, and `docs/TODO.md`'s
observability entry for the trace inspection).

This doc is written to be folded into the project README during Weekend 6, not a replacement
for it.

## Cost: $0 in API fees

Every model call in this project runs on local Ollama (`qwen3:1.7b` for routing/classification/
reasoning, `nomic-embed-text` for embeddings) — there is no per-token billing, no API key, no
usage cap to hit. Phoenix's own trace dashboard confirms this directly: every trace shows
**Total Cost: $0**, since Phoenix's cost tracking is wired for paid APIs and there's genuinely
nothing to meter here.

The real cost of this architecture isn't dollars — it's **latency and compute**, covered below.

## Per-stage latency, measured across the whole project

| Stage | Samples | Min | Median | Max | Avg |
|---|---|---|---|---|---|
| `retrieve_candidates` (Chroma vector search) | 34 | 0.00s | 0.01s | 2.89s | 0.41s |
| `classify_query` (`think:false`) | 49 | 0.51s | 1.03s | 14.18s | 2.81s |
| `extract_product_mentions` (`think:false`) | 9 | 0.69s | 1.51s | 3.01s | 1.51s |
| `compare_products` (`think:true`) | 4 | 32.94s | 34.44s | 37.56s | 34.55s |
| `generate_answer` (`think:true`) | 30 | 10.93s | 37.22s | 128.75s | 46.18s |
| `generate_policy_answer` (`think:true`) | 4 | 18.64s | 22.53s | **719.18s** | 195.39s |
| `rerank` (`think:true`) | 34 | 40.83s | 89.08s | 251.71s | 109.67s |

**Median is the more honest number to quote than average here** — every `think:true` stage has
a long right tail (a handful of very slow outliers pull the average far above what a typical
call actually costs), confirmed visually across every log this project has produced since
Weekend 2's original latency investigation.

### A full RAG round-trip, realistically

A single product/policy question runs `classify_query` → `retrieve_candidates` → `rerank` →
`generate_answer` sequentially. Using medians: **~1s + ~0.01s + ~89s + ~37s ≈ 127 seconds** for
a typical case. This matches what was directly observed end-to-end through the real API: the
concurrency test in `tests/test_api_concurrency.py` measured a full round-trip of **152.38
seconds** for one real product question through `POST /chat`.

## A genuinely new finding, surfaced while writing this doc

Pulling every `[timing]` line (not just the summaries we read live during testing) surfaced two
real outliers that were never flagged before:

- **`generate_policy_answer` hit 719.18 seconds once** (`logs/eval_run_2.log`, the
  `quality-policy-return-window` case — the exact question that exposed the `ORDER`/policy
  router bug) — nearly 12 minutes for one LLM call. This happened silently: the eval run was
  reported as "21/21 passed" because the *answer* was correct, but nobody checked the
  *timing* of that specific call at the time. Correctness and latency are different
  dimensions, and this is a concrete example of a case passing one check while failing the
  other unnoticed.
- **`rerank` hit 251.71 seconds once** (`logs/test_graph_e2e.log`), well above the
  192.94s figure previously called "the worst latency seen in this project" during the
  adversarial-testing work — that record has since been beaten, silently, by a legitimate
  (non-adversarial) regression-test run.

**Not fixed, logged as a new open item** (see `docs/TODO.md`): these tail-latency spikes happen
on `think:true` calls with no code-level ceiling other than the 240-second `OLLAMA_TIMEOUT_SECONDS`
safety net added during the Weekend 4 structured-error-handling work — and the 719s case is proof
that number is itself generous enough that `httpx.TimeoutException` never even fired for a call
this project would otherwise flag as unreasonably slow. The timeout protects against a truly
hung Ollama; it does nothing for "working, but absurdly slow."

## Root cause: still not conclusively identified

This was investigated directly back in Weekend 2 (see `docs/notes/langgraph/03-building-and-
debugging-the-graph.md`) and remains unresolved: structured-output overhead and pure RAM
pressure were both ruled out empirically (freeing RAM made timing *worse*, not better); CPU
contention on this CPU-only, single-machine setup is suspected but never proven. The wide
variance documented above (a `rerank` call ranging from 40.83s to 251.71s for structurally
similar work) is consistent with that theory but isn't new evidence resolving it.

## What was actually done to reduce real latency, with before/after numbers

Two concrete, measured latency wins came out of the Weekend 4 guardrail work — not
speculative, both verified directly:

1. **The `"unsure"` early-decline fix** (`rag_node`, Weekend 4) — when `classify_query` signals
   low confidence, the pipeline now skips `retrieve_candidates`/`rerank`/`generate_answer`
   entirely and returns a clarifying question immediately. Measured: **2.39 seconds total**,
   versus the 40-250+ second range above when the full pipeline runs.
2. **Router hardening against off-topic/injection input** (Weekend 4) — before the fix, 5 of 6
   adversarial/off-topic test messages reached the full `rag_node` pipeline; after broadening
   the `FALLBACK` category description, 6 of 6 were caught at the router, before any expensive
   stage ever ran. Every one of those avoided calls would otherwise have cost 40-250+ seconds
   for input that was never going to produce a useful answer anyway.

Both fixes share the same shape: **the cheapest latency win available was never "make `rerank`
faster" — it was correctly recognizing which requests shouldn't reach `rerank` at all.**

## Token counts (single-sample, from direct Phoenix trace inspection)

Real counts observed on one representative traced request (a product battery-life question) -
not aggregated across many calls, just what was directly visible in the Phoenix UI:

| Call | Observed tokens |
|---|---|
| `router_node`'s `ChatOllama` call | 824 |
| `classify_query`'s `ChatOllama` call | 186 |
| `rerank`'s `ChatOllama` call | 1,743 |
| `generate_answer`'s `ChatOllama` call | 1,035 (prompt: 285) |

`rerank` consistently handles the most tokens per call (it's shown all 10 retrieved candidates'
full text at once) — consistent with it also being the slowest and most variance-prone stage
above; more input tokens plausibly compounds with `think:true` reasoning overhead, though this
project has not isolated token count as a specific causal factor versus the still-unresolved
CPU-contention theory.

## Practical implications for anyone actually deploying this

- **This is not interactive-latency software today.** A 40-250+ second wait per real RAG
  question is acceptable for a demo/portfolio project explained as such, not for a live
  production support chat without further work (the deferred `rerank` "none relevant" escape
  hatch and the true token-streaming option discussed but not built during the FastAPI work are
  the two most direct next steps if this mattered).
- **GPU inference would likely change this story substantially** — this entire profile is
  CPU-only by the project's own hardware constraint; a `think:true` call's cost is dominated by
  sequential reasoning token generation, exactly the workload GPU acceleration targets. Not
  tested here (no GPU available), so stated as a reasoned expectation, not a verified claim.
- **The concurrency semaphore (Weekend 5 FastAPI layer) is latency-aware by design**, not
  accidental: given these real per-call durations, allowing a second heavy request to run
  concurrently on the same CPU wouldn't parallelize work, it would roughly double both
  requests' wall-clock time — serializing them with an immediate 429 instead was the
  measurement-informed choice, not a default.
