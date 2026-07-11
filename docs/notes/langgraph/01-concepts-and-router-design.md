# LangGraph concepts and graph-level router design (design phase, no code yet)

## Why a graph instead of a chain
Everything built so far (`prompt | llm | parser`) is a linear pipeline — step 1 always leads to step 2. Our agent needs to *branch* based on a decision made mid-flow (product question vs. order lookup vs. safety complaint), and LCEL chains have no clean way to express that. That's the actual gap LangGraph fills — not a "better" way to do what we already have, a way to do something chains structurally can't.

## Core building blocks
- **State** — a shared data structure flowing through every node, accumulating information across the whole run (and, with a checkpointer, across turns).
- **Nodes** — units of work (a function or a chain) that read state and return updates to it. Our existing `classify_query`/`retrieve_candidates`/`rerank`/`generate_answer` each naturally become a node.
- **Edges**, especially **conditional edges** — "after this node, go to X if condition A, else Y." This is what turns our informal "if classified as product, filter to type=product" into a first-class graph structure instead of an `if` statement buried in our own code.

## Checkpointers
Solve a specific mechanical problem: without one, every message starts from a blank state — zero memory of anything said before, unless we manually resend the whole history ourselves every time.

**Mechanism:** compiling a `StateGraph` with a checkpointer makes LangGraph automatically save a snapshot of the *entire* State (not just chat messages — everything we define as state) after every node runs, keyed by a `thread_id` (a conversation/session ID we provide). A new message for an existing `thread_id` causes LangGraph to load the last snapshot and continue from there, rather than starting fresh.

**Web-session analogy, made precise:** `thread_id` ≈ session ID; checkpoint backend ≈ session store; LangGraph's execution engine reads/writes it around each turn the same way session middleware wraps each HTTP request in a web app.

**Backend choices, swappable:** `MemorySaver` (in-process, lost on restart, fine for dev), `SqliteSaver` (local file, durable, single-machine), `PostgresSaver` (shared DB, durable, multiple server processes/replicas can share conversation state — the real production choice).

**Connection to `interrupt()`:** a paused graph relies on the checkpoint to preserve exactly where execution stopped, so that when a human later responds, the graph resumes from that exact point rather than restarting. The two mechanisms are linked, not just neighbors in the plan.

## Routing architecture decision: two-level (hierarchical), not one flat classifier
Decided **two separate routing decisions**, not one classifier with more categories:
1. **Graph-level router** — RAG vs. ORDER vs. ESCALATION vs. AUTH vs. FALLBACK. Decides *which subsystem* handles the request (different code paths, different auth/safety implications).
2. **RAG-internal router** (already built — `classify_query`) — product vs. policy vs. faq. Decides *which slice of the same knowledge base* to search.

**Why, for us specifically:** these are different *kinds* of decisions, and cramming both into one flat classifier risks the same failure we already saw — `classify_query` got confused with only 4 categories on novel Hinglish phrasing (see `docs/notes/rag/06-multilingual-hindi-hinglish.md`); more categories in one classifier would likely make that worse. We also already have precedent for multi-stage decisions: the deterministic safety-keyword filter runs *before* any LLM routing at all.

**Why this also matches production practice at scale, not just a convenient shortcut for a small project:** different branches have different cost/latency/reliability profiles and are often owned by different teams at a real company; production systems typically funnel cheap/deterministic checks first, then a fast coarse classifier, reaching for something more specific only where needed. Hierarchical routing is the standard pattern here, not a compromise.

## Graph-level router design (decided, not yet built)

```
[User Input] → [Intent Router]
    ├─→ AUTH         (sign-in/out — stubbed this weekend)
    ├─→ RAG          (existing classify→retrieve→rerank→generate pipeline)
    ├─→ ORDER        (get_order_status tool)
    ├─→ ESCALATION   (stubbed this weekend)
    └─→ FALLBACK     (doesn't fit anywhere, or too vague)
```

**Two-stage decision order**, extending the already-established escalation design: deterministic keyword regex runs first — if it matches, short-circuit straight to `ESCALATION`, skip the LLM router entirely (same reasoning as before: a missed safety escalation is unacceptable, and this also saves a model call for the clearest cases). Only if the regex doesn't fire does the LLM router run, choosing among all 5 categories — `ESCALATION` still exists as an LLM-classifiable option for the plan's "softer signals" case (frustration, repeated complaints) that isn't a clean keyword match.

**Draft category descriptions for the router prompt** (applying the Day 1 lesson up front — real descriptions, not bare names, since bare names are known to fail):
- AUTH: user wants to sign in, sign out, or manage account credentials
- RAG: asking about product specs/comparisons, policies (returns/warranty/shipping), or general how-to/troubleshooting help
- ORDER: asking about the status, tracking, or history of a specific order
- ESCALATION: expressing frustration, repeated unresolved complaints, or wanting to speak to a human (not for clear safety keywords — those are already caught before this point)
- FALLBACK: doesn't clearly fit any of the above

**Draft State schema:**
- `messages` — conversation so far (what the checkpointer persists across turns)
- `route` — the graph-level category the router picked, read by the conditional edge
- `is_authenticated` — bool, order lookups gated on this even though the Auth node is just a stub this weekend
- `retrieved_docs` — RAG node's intermediate output, kept for debugging/inspection
- `answer` — final response text, whichever branch produced it

## Status
Design agreed here on 2026-07-05; built, tested, and revised on 2026-07-11 — see `02-router-structured-output-and-prompt-fixes.md` and `03-building-and-debugging-the-graph.md`. Two design decisions changed shape during implementation, worth flagging since this file describes the *original* plan:
- The deterministic keyword pre-filter described above ("if it matches, short-circuit straight to ESCALATION") was **not** built this way — real adversarial testing showed hard-gating on keywords produces false positives (e.g. "these earbuds feel like fire" as a compliment). It became a *hint* fed into the LLM router's prompt instead, never a hard gate. See `02-...md` for why.
- The router's output format changed from plain-string parsing (like `classify_query`) to Pydantic + `.with_structured_output()`, for reliability reasons specific to this being a graph-routing decision rather than a RAG filter. See `02-...md`.
