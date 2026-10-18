# Architecture

Two diagrams: the full system (everything the agent depends on), and the graph's internal
routing logic (regenerated directly from the live compiled graph via `scripts/visualize_graph.py`
— not hand-drawn, so it can't silently drift from the real code).

## System overview

```mermaid
flowchart TD
    Browser["Browser<br/>static/index.html<br/>(thread_id in localStorage)"]

    subgraph API["FastAPI (main.py)"]
        Chat["POST /chat"]
        Sem["Semaphore(1)<br/>serializes requests -<br/>one CPU-bound Ollama instance"]
    end

    subgraph Agent["LangGraph (graph/)"]
        Graph["Compiled StateGraph<br/>router_node, rag_node, order_node,<br/>escalation_node, auth_node, fallback_node"]
    end

    Ollama["Ollama<br/>qwen3:1.7b (routing/reasoning)<br/>nomic-embed-text (embeddings)"]
    Chroma[("Chroma<br/>data/chroma/<br/>products + faqs + policies")]
    OrdersDB[("SQLite<br/>data/orders/orders.db<br/>orders / order_items / shipment_timeline")]
    CheckpointDB[("SQLite<br/>data/checkpoints/checkpoints.db<br/>conversation state, survives restarts")]
    Phoenix["Arize Phoenix<br/>localhost:6006<br/>local-only trace UI"]

    Browser -- "HTTP POST {thread_id, message}" --> Chat
    Chat --> Sem
    Sem --> Graph
    Graph -- "HTTP, separate calls" --> Ollama
    Graph -- "similarity + metadata filter" --> Chroma
    Graph -- "exact order lookup" --> OrdersDB
    Graph -- "read/write per thread_id" --> CheckpointDB
    Graph -. "OTel spans via observability.py" .-> Phoenix
    Chat -- "{route, answer}" --> Browser
```

**Why each arrow is the shape it is, not just that it exists:**
- Browser → FastAPI is the only network hop a user ever makes; everything else is server-side.
- The graph talks to Ollama over its *own* separate HTTP calls — FastAPI never touches Ollama directly, it only ever calls `graph.invoke()`.
- Chroma is reached via *similarity search filtered by metadata* (never a raw unfiltered scan, after the Weekend 2 fix) — two different query mechanisms depending on whether it's a RAG answer (similarity) or a comparison (exact metadata `$in` filter).
- Orders and checkpoints are two **separate** SQLite databases, not one — orders is business data seeded from JSON, checkpoints is live conversation state that grows from real usage. Conflating them would mix a regeneratable artifact with real runtime state.
- The Phoenix arrow is dashed because it's *fire-and-forget telemetry* — if Phoenix is down, tracing silently drops (confirmed empirically — a connection-refused error during testing didn't crash the actual request).

## Graph-internal routing (regenerated from the live graph, `docs/graph/router_graph.mmd`)

```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	router_node(router_node)
	rag_node(rag_node)
	order_node(order_node)
	escalation_node(escalation_node)
	auth_node(auth_node)
	fallback_node(fallback_node)
	__end__([<p>__end__</p>]):::last
	__start__ -.->|"route_from_start:<br/>mid-auth-flow?"| auth_node;
	__start__ -.-> router_node;
	router_node -.->|"route_after_router"| auth_node;
	router_node -.-> escalation_node;
	router_node -.-> fallback_node;
	router_node -.-> order_node;
	router_node -.-> rag_node;
	auth_node --> __end__;
	escalation_node --> __end__;
	fallback_node --> __end__;
	order_node --> __end__;
	rag_node --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

**The two conditional edges aren't redundant** (a common point of confusion worth being able to explain cold): `route_from_start` only checks "are we mid-OTP-exchange right now" — it bypasses the router entirely during an in-progress auth flow, since the router has no concept of conversation stage. `route_after_router` handles everything else, including the *separate* case of an unauthenticated user hitting `ORDER` for the first time, which it redirects to `auth_node` as a fresh entry, not a resumption.

### Inside `rag_node` — not visible in the graph diagram above, since it's one node's internal logic, not graph structure

```
rag_node
  └─ classify_query() → one of: comparison | policy | unsure | product | faq
       ├─ "comparison" → compare_products()       (exact metadata fetch, no similarity search)
       ├─ "unsure"     → decline early             (skips retrieval entirely - cost/confidence fix)
       ├─ "policy"     → retrieve → rerank → generate_policy_answer()  (quote-verified + cited)
       └─ "product"/"faq" → retrieve → rerank → generate_answer()     (numeric-fact-checked)
```
