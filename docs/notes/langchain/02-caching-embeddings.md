# Caching embeddings to avoid recomputation

## The problem
`Chroma`'s public insert API (`add_texts`/`add_documents`/`from_documents`) always recomputes embeddings internally by calling the embedding function itself — there's no parameter to hand it precomputed vectors. Verified via source: no `embeddings=` passthrough exists on the LangChain wrapper's public methods (native `chromadb`'s own `collection.add()` does support this, but that's bypassing LangChain entirely). This mattered because we wanted to keep our Day 1 "avoid re-embedding unchanged content" discipline.

## LangChain's own answer — and why we didn't use it
`CacheBackedEmbeddings.from_bytes_store(underlying_embedder, document_embedding_cache, namespace=...)` — wraps any `Embeddings` object, hashes input text, checks a key-value store (`ByteStore`, e.g. `LocalFileStore`) before calling the real model. Solves the problem *inside* the normal `Chroma.from_documents()` flow, no bypassing needed.

**Real, current finding (verified, not assumed):** in LangChain v1 (what we have, `langchain==1.3.11`), this was moved out of the core `langchain` package into a separate `langchain-classic` package — and the other place it used to live, `langchain-community`, was archived/sunset in June 2026 and is no longer maintained. LangChain deliberately shrank its core package to focus on essential agent-building blocks. Decided not to add a dependency the library authors are visibly moving away from — hand-rolled the same idea instead.

## Hand-rolled version (`ingestion/cached_embeddings.py`)
`CachedEmbeddings` subclasses `langchain_core.embeddings.Embeddings` (the same ABC `OllamaEmbeddings` implements — confirmed its abstract methods are `embed_documents(texts) -> list[list[float]]` and `embed_query(text) -> list[float]`), wraps a real embeddings object, and:
- Hashes each text (SHA256) → checks a persisted JSON file (`data/embedding_cache/<namespace>.json`) for that hash.
- Only calls the real model for texts not already cached (batched, one call for however many are missing).
- **Critically preserves input order in its return value** (`return [self.cache[h] for h in hashes]`, iterating over the *original* order) — this is what makes it a transparent drop-in for `Chroma.from_documents()`, which trusts `embed_documents()`'s output order to match its input order (see 01-lcel-and-rebuild.md's pairing note).

## The cache is not a backup — a real gap worth remembering
The cache file only contains `{hash: vector}` — no id, text, or metadata. If we lost the Chroma DB, we could cheaply rebuild it *using the cache* — but only because `data/processed/chunks.jsonl` still exists to re-derive (id, text, metadata) and re-hash each chunk to look up its vector. If we lost `chunks.jsonl` instead, the cache alone would be useless — hashes are one-way, no way to recover the original text or which document a vector belonged to. `chunks.jsonl` is the real source of truth; the cache is purely a compute-avoidance layer, never a substitute for it. This is the same "a vector means nothing without its pairing" lesson from Day 1, showing up in a new form.

## What "no caching" would have looked like
Practically identical on the *first* run (nothing's cached either way). The difference shows up on a second run after changing even one chunk: without caching, all 131 chunks get re-embedded from scratch; with caching, only the changed one does. Negligible cost at our scale (free local compute, seconds of difference) — the value is understanding the pattern that matters a lot at real scale (paid embedding APIs, millions of records).

## Real production scale, beyond just this cache
- Use the cache to make "avoid recomputation" invisible inside the normal LangChain flow (what we did), rather than bypassing the vector store's insert API.
- For genuinely large bulk loads, use the vector DB's **native** bulk-insert API (optimized for high-throughput batch writes) for the ingestion job specifically, while still using LangChain for the serving/query side.
- Distributed processing (Spark/Airflow) instead of one sequential script; a durable staging layer (Parquet on S3) instead of a local JSON file; incremental/change-data-capture re-indexing so a nightly job only touches what actually changed.

## A related, real concurrency bug class (discussed, not built)
If someone parallelizes embedding calls and aggregates results by *completion* order (e.g. naively via `as_completed()` or a shared list appended to by multiple workers) rather than *submission* order, positional pairing breaks silently — worker 3 finishing before workers 1 and 2 means its vector lands at the wrong index. Tools like `ThreadPoolExecutor.map()` and `asyncio.gather()` specifically guard against this by preserving input order regardless of completion order; a manual aggregation must carry the original index through the parallel execution to stay correct. Our sequential-batch design (chosen for CPU-bound-hardware reasons on Day 1) gets immunity to this entire bug class for free, as a bonus we hadn't originally intended.
