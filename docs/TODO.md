# TODO

Lightweight, scannable pending-work list. Source of truth for *why* still lives in `progress.md` (daily log) and `claude-proposed-plan.md` (the roadmap) — this file just tracks *what's left*, checkbox-style. Update whenever something is finished, added, or dropped.

---

## Immediately pending (carried over from 2026-07-11)

Both items resolved 2026-07-12 — see "Done" and "Deferred / open" below. Section kept empty as a placeholder for whatever comes up next.



## Weekend 3 — Full feature build (not started)

- [ ] Expand knowledge corpus (richer product/comparison/policy/FAQ content)
- [ ] Product comparison via structured metadata filtering (not pure similarity search)
- [ ] Order tracking: move from mock JSON lookup to a SQLite mock DB + Pydantic-schema'd tool args
- [x] Auth: real mock email/OTP flow, graph-enforced gating on order tools — 2026-07-12. Design: `State` carries `user_id`/`auth_expires_at`/`auth_stage`/`pending_email`/`pending_otp` (not a bare bool); `is_authenticated()` gate lives in `graph/edges.py`'s conditional edge (never a tool, never inside a node) so the LLM has no agency to skip it; `auth_node` is a 3-stage state machine (prompt for email -> collect email + issue mock OTP -> verify OTP via plain string comparison). Found and fixed a real bug via testing: the triggering message (not an actual email) was being misparsed as the email when auth was reached via an ORDER redirect rather than an explicit "log me in" — fixed by adding an explicit `awaiting_email` stage that always prompts fresh rather than parsing the message that caused the redirect. Also found and fixed a structural gap: the graph's `START` edge was unconditionally routing every turn through the router, which broke mid-auth-flow turns (a reply to "enter your OTP" got reclassified as a fresh intent) — fixed via a conditional edge at `START` that bypasses the router entirely while `auth_stage` is set. Verified end-to-end with real assertions (not just internal consistency) in `tests/test_auth_flow.py`. See `docs/notes/langgraph/04-graph-enforced-auth.md` for full reasoning, including why a "check auth" tool would have been the wrong shape, and how `thread_id` is not a real identity boundary (deferred to the Weekend 5 API layer).
- [ ] Escalation: real `interrupt()` human-in-the-loop mechanism + simulated escalation queue (currently just a stub message)
- [ ] Policy answers with exact-wording grounding + citations



## Weekend 4 — Production hardening, part 1 (not started)

- [ ] Eval set (25-30 questions) + scoring (LangSmith evals or LLM-as-judge)
- [ ] Input guardrails: prompt-injection resistance, off-topic deflection
- [ ] Output guardrails: groundedness check (no invented policies/specs)
- [ ] Structured error handling (LLM timeout/retry, vector DB down, tool failure)



## Weekend 5 — Production hardening, part 2 (not started)

- [ ] LangSmith observability/tracing
- [ ] FastAPI layer (streaming, session management, rate limiting)
- [ ] Frontend chat UI
- [ ] **Open decision:** deploy target — local Docker Compose demo vs. small VPS vs. local-only + README writeup (local Ollama won't run on free-tier PaaS)
- [ ] Cost/latency documentation in README



## Weekend 6 — README & interview prep (not started)

- [ ] Architecture diagram
- [ ] Design-decision writeup (why LangGraph, chunking strategy, router model choice, why auth is graph-enforced, escalation design)
- [ ] Eval results writeup
- [ ] Known limitations & next-steps section
- [ ] Interview-question rehearsal



## Deferred / open, no fixed weekend yet

- [ ] Latency root cause not conclusively identified (ruled out structured-output overhead and pure RAM pressure; CPU contention suspected) — revisit in a dedicated hardening pass, see `docs/notes/langgraph/03-building-and-debugging-the-graph.md`
- [ ] Hinglish generation follow-up: model understands Hinglish input but replies in English regardless — no instruction currently tells it to match input language (see `docs/notes/rag/06-multilingual-hindi-hinglish.md`)
- [ ] Blended-intent messages: single-route schema silently drops the secondary intent (e.g. an order-status + product-question combo message only gets the order half answered). **Decided 2026-07-12: defer, not a quick fix.** Researched production pattern (web search): "intent splitting" — detect multiple sub-intents, fan out to handle each, merge into one coherent reply. Real fix here = `routes: list[Route]` + a graph fan-out/merge mechanism — genuinely new graph-mechanics work, worth its own concept-first stage later rather than solving half-heartedly now.

---



## Done

**Weekend 1 — RAG hands-on + LangChain** (started 2026-07-04, finished 2026-07-05)

- [x] Zero-framework RAG pipeline: chunk → embed (`nomic-embed-text`) → store (ChromaDB) → classify → retrieve → rerank → generate — 2026-07-04
- [x] LangChain rebuild: `OllamaEmbeddings`, `Chroma` via `langchain-chroma`, `ChatOllama`, LCEL chains (`lc_query.py`) — 2026-07-05
- [x] Content-hash embedding cache (`ingestion/cached_embeddings.py`) since LangChain's own caching wrapper was being deprecated — 2026-07-05
- [x] `get_order_status` tool via `@tool` + `ChatOllama.bind_tools()`, verified across 3 test cases (calls tool / declines / asks for missing order number) — 2026-07-05
- [x] Verified local model stack via direct Ollama API calls (embeddings, router, generation models) — 2026-07-04

**Weekend 2 — LangGraph mental model + graph skeleton** (started 2026-07-05, finished 2026-07-11)

- [x] Two-level hierarchical router design (graph-level router + existing RAG-internal `classify_query`) — 2026-07-05
- [x] Graph-level router built with structured output (`.with_structured_output()` + Pydantic `RouterDecision`/`Route` enum) — verified reliable across 9 adversarial test cases — 2026-07-11
- [x] Regex safety-keyword hint (fed into router prompt as a signal, not a hard gate) — fixed a real false-positive risk found via adversarial testing — 2026-07-11
- [x] Real `graph/` package built: `state.py`, `nodes.py`, `edges.py`, `build.py` — compiled `StateGraph` with `MemorySaver` checkpointer — 2026-07-11
- [x] Router + RAG node + Order node working end-to-end with real multi-turn memory (checkpointed message history, human+AI messages both persisted) — 2026-07-11
- [x] Escalation and Auth nodes stubbed and verified routing correctly — 2026-07-11
- [x] Graph visualized (Mermaid) and confirmed structure matches design — 2026-07-11
- [x] Two real bugs found via E2E testing and fixed: missing AI messages in checkpointed history; `order_node` returning raw JSON instead of a phrased answer — 2026-07-11
- [x] `.gitignore` added, initial git commit made — 2026-07-11

**Docs/process** (ongoing since 2026-07-04)

- [x] `docs/notes/` written up for RAG (2026-07-04/05), LangChain (2026-07-05), and LangGraph (2026-07-05, 2026-07-11) stages
- [x] `docs/progress.md` kept as a running daily log — since 2026-07-04
- [x] This `docs/TODO.md` created for at-a-glance pending work — 2026-07-11