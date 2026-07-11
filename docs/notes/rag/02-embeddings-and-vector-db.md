# Embeddings and vector databases

## What an embedding is
A vector (list of floats — 768 numbers for `nomic-embed-text`) produced by a single deterministic forward pass through a model trained so that semantically similar text lands close together in that space. Unlike generation, there's no sampling step — same input reliably gives the same vector (verified empirically: re-embedded the same chunk twice, got bit-for-bit identical vectors). Tiny floating-point differences from parallel CPU reduction order are theoretically possible but negligible in practice.

**"Embedding" and "vector" are the same object** here — embedding describes how it was produced (meaning-encoding), vector is the general math term for the list of numbers.

## Why a dedicated embedding model instead of asking an LLM "is X similar to Y"
Embeddings turn search into arithmetic: compute each chunk's vector once at ingestion, cache it forever, then at query time just compare the query vector against already-computed vectors with cheap math (cosine/L2 distance) — no model calls needed for the search itself. Asking an LLM to judge relevance instead means one generation call per candidate chunk — doesn't scale (50 chunks = 50 slow calls per query). Embedding models are also specifically trained (contrastive learning) so meaning-closeness maps to vector-closeness; generative models are trained for fluent text, not comparable vectors.

## Batching
Ollama's `/api/embed` accepts a list of texts in one request and returns one vector per input, in the same order — confirmed via curl before building `embed.py`. Batching cuts per-call overhead vs. one-request-per-chunk. Concurrency (multiple parallel requests) does NOT help on our CPU-only local setup the way it would against a remote API — this is compute-bound (finite CPU cores), not I/O-bound (waiting on a network response), so parallel requests mostly just compete for the same cores rather than getting genuine speedup.

## Why persist embeddings to a file before loading into the vector DB
Not strictly necessary at small scale — could embed and insert into Chroma in one step. Worth doing because: (1) embedding is often the expensive step to redo (paid APIs, real compute) — persisting decouples "pay for embeddings" from "load into a serving store"; (2) lets you swap vector DBs later without recomputing; (3) makes each pipeline stage independently inspectable/debuggable.

## The "pairing" problem
A vector alone (768 floats) means nothing — it's just a coordinate. It only becomes useful paired with the source text + metadata. This pairing isn't an artifact of using an intermediate file — Chroma's own `.add()` API takes `ids`, `embeddings`, `documents`, `metadatas` as **parallel lists**, assuming position 0 in each list belongs together. Get the order wrong anywhere and you silently get semantically-valid-looking but wrong retrievals — no crash, just wrong answers.

## Chroma internals (what `PersistentClient` actually writes to disk)
- `chroma.sqlite3` — structured bookkeeping: ids, documents, metadata, collection config.
- A UUID-named folder per collection — the actual HNSW (Hierarchical Navigable Small World) vector index: `data_level0.bin` (raw vectors), `header.bin` (index config), `length.bin` (precomputed norms), `link_lists.bin` (the graph's neighbor connections). HNSW is what makes similarity search sub-linear at scale — walk a graph instead of comparing against every stored vector.
- `.add()` with an ID that already exists silently no-ops for that ID (no duplicate, no error) — confirmed empirically by re-running `store.py` and checking the count stayed the same. This also means `.add()` won't *update* an existing entry; use `.upsert()` for that.

## Distance metric: L2 vs cosine
Chroma defaults to L2 (squared Euclidean) unless you configure `hnsw:space: cosine` — we never did. Checked whether this mattered: our `nomic-embed-text` vectors are unit-normalized (length ≈ 1.0). For unit vectors, `L2² = 2 − 2×cosine` — a fixed monotonic relationship, so L2 and cosine produce the **identical nearest-neighbor ranking**. Real gap between design intent and config, but harmless here specifically because of this normalization property — not something to assume in general.
