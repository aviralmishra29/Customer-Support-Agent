# Building the real graph: package structure, two E2E bugs, and visualization

## Package structure

Checked against documented LangGraph conventions (web search, not assumed) before committing to a layout. Confirmed pattern: a package with `state.py` / `nodes.py` / `edges.py` / a top-level assembly file — matches what production LangGraph repos and the official docs describe, not something improvised.

- `graph/state.py` — the `State` TypedDict. Key detail: `messages: Annotated[list, add_messages]` — the `add_messages` reducer is what makes new messages *append* to history across graph invocations instead of overwriting it. Without this annotation, returning `{"messages": [...]}` from a node would replace the whole list, not add to it.
- `graph/nodes.py` — `router_node`, `rag_node` (thin wrapper around the already-built `lc_query.py` chain), `order_node` (wraps `get_order_status` via `bind_tools`), and stubs for `escalation_node`/`auth_node`/`fallback_node`.
- `graph/edges.py` — `route_after_router`, the conditional-edge function. Reads `state["route"]`, looks it up in a `{route_name: node_name}` dict, defaults to `fallback_node` if the value is ever unexpected — a safety net on top of structured output, not a replacement for it.
- `graph/build.py` — assembles `StateGraph(State)`, registers all nodes, wires `START → router_node`, adds the conditional edge, routes every leaf node to `END`, compiles with `MemorySaver()` as the checkpointer.

**Convention confirmed from LangGraph docs:** nodes must return a *new* partial state update, never mutate the existing state object in place — this is how `StateGraph` merges results, not just a style preference.

## Bug 1: message history only ever contained human messages

First real end-to-end run (`tests/test_graph_e2e.py`, 2 turns on one `thread_id`) showed `messages in thread: 2` after 2 full turns — should have been 4. Root cause: every node returned `{"answer": ...}` but never touched `state["messages"]` at all, so the checkpointer had nothing from the AI side to persist. This silently defeats a chunk of the point of using a checkpointer — the next turn's router/RAG calls would have no idea what was said before.

**Fix:** a shared helper every node routes its return value through:

```python
def _answer_update(answer: str, **extra) -> dict:
    return {"answer": answer, "messages": [AIMessage(content=answer)], **extra}
```

Reran: `messages in thread: 4`, correctly alternating `[human, ai, human, ai]`.

## Bug 2: order_node returned raw JSON as the customer-facing answer

`get_order_status` (existing tool from Weekend 1) returns a JSON string. `order_node` was passing that straight through as `answer` — a real customer would see `{"status": "delivered", "items": [...]}` instead of a sentence.

**Fix, and why parse-then-phrase beats phrase-directly:** the tool's data is already exact and correct — there's nothing to *reason about*, only to *word naturally*. Handing raw JSON to an LLM and asking it to "make this readable" makes the model do parsing and generation simultaneously, which risks it misreporting or inventing a detail. Instead: parse the JSON ourselves in Python (exact, not probabilistic), extract only the fields worth mentioning (status, items, carrier, tracking number, latest timeline event) into a plain-language context string, and only then ask the LLM to phrase *that* into a sentence. Exact same principle as `generate_answer`'s grounded generation in the RAG pipeline — the model phrases already-correct information, it doesn't extract or compute it.

## Visualizing the compiled graph

`graph.get_graph()` exposes a few renderers:
- `.draw_ascii()` / `.print_ascii()` — needs the optional `grandalf` package; failed cleanly with an `ImportError` when not installed, caught and skipped rather than crashing the whole script.
- `.draw_mermaid()` — plain text, no dependencies, always works. Written to `docs/graph/router_graph.mmd`; pasted into mermaid.live to actually view it.
- `.draw_mermaid_png()` — calls out to `mermaid.ink`'s render API; failed here with no network access from this environment. Not a code problem, just an environment constraint — the `.mmd` text file is the reliable fallback.

Visual output confirmed the structure matches the design: `__start__ → router_node`, then 5 **dashed** (conditional) edges fanning out to all 5 leaf nodes, each converging via solid edges to `__end__`.

## Latency: investigated, root cause not conclusively found, explicitly deferred

Original 2026-07-04 benchmark (`qwen3:1.7b`, `think:false` ≈ 2-4s/call) no longer holds during this session's testing — observed 20-52s/call *average*, with individual calls ranging 14-78s for what should be equivalent prompts.

Ruled out (with actual measurements, not guesses):
- **Structured output as the cause** — plain-text `think:false` calls (no `.with_structured_output()`) were *also* slow (~20s avg), so it's not primarily a structured-output/tool-binding tax.
- **RAM as the sole cause** — `free -h` showed real memory pressure (824Mi free, 2.7Gi swap in use) initially; closing memory-heavy apps (Chrome/Cursor/Postman) freed RAM up to 2.8Gi free and dropped swap to 1.8Gi, but a rerun immediately after was *slower* (52s avg), not faster. This points toward CPU contention (background processes competing for cycles) as at least as significant as RAM pressure, though not conclusively isolated either.

Checked `ollama ps`: model was loaded, 100% CPU (confirms no GPU offload, expected on this hardware), with a short keep-alive window — meaning the model likely reloads from disk between script runs that are more than a few minutes apart, adding a real but unquantified cold-start cost on top of whatever the base slowdown is.

**Decision:** stop investigating for now, treat as a known variable cost of local CPU-only inference rather than something to solve mid-router-design. Doesn't change any architecture decision made this session. Flagged in `docs/progress.md` as a to-do for a later, dedicated hardening/performance pass.
