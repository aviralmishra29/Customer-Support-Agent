# RAG pipeline and chunking

## Why RAG exists
An LLM only knows its training data plus whatever's in the current prompt. Three ways to add domain knowledge: fine-tuning (expensive, slow to update, imprecise for exact facts), stuffing everything into the prompt (doesn't scale, "lost in the middle" problem), or RAG (store knowledge externally, retrieve only what's relevant per query). RAG is fundamentally a search problem — get retrieval right and generation is easy; get it wrong and no prompt engineering saves you (garbage in, garbage out).

## The pipeline
**Ingestion (once, or whenever corpus changes):** load docs → chunk → embed each chunk → store (vector + text + metadata) in a vector DB.
**Query time (every message):** embed the query (same model as chunks) → retrieve top-k by similarity → stuff into prompt → generate.

## Chunking
Don't embed whole documents — a chunk covering multiple topics produces a blurry, averaged embedding that matches nothing well. Chunk so each vector stays focused on one coherent idea. This matters even for small corpora — it's a precision problem, not a scale/speed problem (speed only becomes the dominant concern at real scale, thousands+ of docs).

**Structural chunking > fixed character-count splitting when the source has natural boundaries.** For our corpus: JSON arrays already give one-object-per-record boundaries (one product = one chunk, one FAQ = one chunk); Markdown policy docs split cleanly on `##` headers. Fixed-size splitting is more of a fallback for unstructured prose without natural breaks.

**Chunks must be self-describing.** A chunk shouldn't just be "Manufacturing defects in materials..." with no indication of which policy it's from — prefix with doc title + section header (e.g. "Warranty Policy — Coverage: ...").

**Convert non-natural-language source formats to prose before embedding.** Raw JSON syntax (braces, quotes, `"anc": false`) and Markdown tables/bold markers are out-of-distribution noise for an embedding model trained on natural text — a boolean `false` carries no meaning on its own. Convert to sentences: `{"anc": false}` → "no ANC"; a markdown table row → "Audio warranty: 1 year."

## What actually belongs in a vector DB (vs. a real DB)
Vector DBs are for content searched *by meaning*: product descriptions, policies, FAQs — relatively static, fuzzy-query-friendly text. They're bad at exact matches, numeric filters, and fast-changing data (price, stock, order status) — that stays in a relational/document DB, accessed via direct queries or tool calls. Embeddings also go stale when source data changes — production systems need a re-embedding pipeline to keep vectors in sync.

**Not everything that "seems textual" belongs in RAG either.** Escalation/safety detection isn't a retrieval task (there's no document to find) — it's classification on the incoming message. We use a deterministic keyword regex for that, not embeddings or LLM classification, because a missed safety escalation is the one failure mode this project can't tolerate — see [[project_boat_support_agent]].
