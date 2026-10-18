# boAt Support Agent

A local-only AI customer support agent for a (fictional) boAt-style audio brand — built as a
hands-on learning project to go deep on LangGraph, RAG, and agentic system design, not as a
production product. Everything runs on-device via Ollama: no API keys, no per-token cost, no
data leaving the machine.

## What it does

- Answers product questions (specs, comparisons) and policy questions (returns, warranty) by
  retrieving from a local knowledge base, not from the model's own memory.
- Looks up real order status for authenticated users (mock OTP auth flow).
- Detects safety-relevant complaints and routes them to a human-escalation stub.
- Declines to answer when it's genuinely unsure, instead of guessing.
- Refuses off-topic and prompt-injection input before it reaches any expensive pipeline stage.

## Architecture

A six-node LangGraph `StateGraph` (`router_node`, `rag_node`, `order_node`, `escalation_node`,
`auth_node`, `fallback_node`) sits behind a FastAPI layer. A two-level routing hierarchy decides
first *which subsystem* handles a message, then (inside RAG) *which slice* of the product corpus
is relevant. Every routing decision is Pydantic-structured output, never a parsed free-text
string — this closed a real bug where free-text classifier output silently bypassed a retrieval
filter under adversarial input.

Full system diagram, the graph's internal routing diagram (auto-regenerated from the live
compiled graph, not hand-drawn), and the reasoning behind each design choice:
**[`docs/architecture.md`](docs/architecture.md)**.

## Why it's built this way

Every non-obvious decision in this project — why LangGraph over a plain chain, why two
classifiers instead of one, why auth is enforced in the graph's routing code and never exposed
as an LLM-callable tool, why policy answers require verbatim quote verification, why the eval
harness is homegrown instead of LangSmith, why this stays local instead of deploying to a VPS —
is written up with the real reasoning and the alternative that was rejected:
**[`docs/design-decisions.md`](docs/design-decisions.md)**.

## Tech stack

Python · LangChain · LangGraph · FastAPI · Chroma (vector store) · SQLite (orders +
conversation checkpoints) · Ollama (`qwen3:1.7b` for routing/reasoning, `nomic-embed-text` for
embeddings) · Arize Phoenix (local observability)

## Running it

Requires Python 3.12 and [Ollama](https://ollama.com) running locally with `qwen3:1.7b` and
`nomic-embed-text` pulled.

```bash
pip install -r requirements.txt
python ingestion/lc_ingest.py          # build the Chroma vector store
python ingestion/seed_orders_db.py     # seed the mock orders database
uvicorn main:app --reload
```

Then open `static/index.html` in a browser, or call `POST /chat` directly with
`{"thread_id": "...", "message": "..."}`.

A `Dockerfile` is included for reference but is not required — this project targets local,
single-machine use only (see [Known Limitations](docs/known-limitations.md)).

## Evaluation

A 21-case local eval harness (`evals/run_evals.py`) with deterministic, string-based scoring —
no LLM-as-judge. Its first real run caught a genuine router bug (a return-policy question
misrouted to the order-lookup path) that no earlier unit test could have found. Full results,
and an honest account of what the eval set does and doesn't prove:
**[`docs/eval-results.md`](docs/eval-results.md)**.

## Cost & latency

$0 in API fees — everything runs on local Ollama. The real cost is latency: CPU-only inference
means a full RAG round-trip typically takes ~2 minutes, with documented tail-latency outliers up
to 719 seconds on `think:true` calls. Full measured numbers, root-cause investigation, and the
two concrete latency fixes that were shipped (with before/after numbers):
**[`docs/cost-and-latency.md`](docs/cost-and-latency.md)**.

## Known limitations

An honest, non-sanitized list of what's genuinely unresolved — real unfixed bugs, deliberately
deferred fixes, and decisions placed explicitly out of scope — plus what would actually come
next if the project continued: **[`docs/known-limitations.md`](docs/known-limitations.md)**.

## Project docs

| Doc | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | System + graph-routing diagrams |
| [`docs/design-decisions.md`](docs/design-decisions.md) | Every major decision, why, and the rejected alternative |
| [`docs/eval-results.md`](docs/eval-results.md) | Eval set, results, what they do/don't prove |
| [`docs/cost-and-latency.md`](docs/cost-and-latency.md) | Real measured cost & latency numbers |
| [`docs/known-limitations.md`](docs/known-limitations.md) | Open bugs, deferred work, out-of-scope decisions |
