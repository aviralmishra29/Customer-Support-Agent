# LangChain rebuild: LCEL, Runnables, and what actually changed vs. our scratch build

## Why rebuild at all, given we already hand-built everything
Not a new concept — a new standard tool for concepts we already understand deeply. Value: less boilerplate, standardized swappable interfaces (`ChatOllama`/`OllamaEmbeddings` could be swapped for another provider's equivalent with minimal code change), pre-built implementations of common patterns. Cost: reduced transparency into what's happening under the hood, and dependence on whatever a library's maintainers chose to expose — see the `classify_query` example below for a concrete instance of this, not just a theoretical claim.

## Core abstraction: `Runnable` and LCEL
Almost everything (prompt templates, chat models, output parsers, retrievers) implements the same `Runnable` interface — invokable, and composable with `|`. `chain = prompt | llm | parser` reads left-to-right like Unix pipes: each stage's output becomes the next stage's input.

## Kept our own chunking, swapped everything else
Compared `RecursiveCharacterTextSplitter`'s generic paragraph/character-based splitting against our structural chunking (JSON records, Markdown headers) — concluded ours is genuinely better-suited to our data: JSON objects have no natural blank-line boundaries the splitter could exploit, and its "greedily repack adjacent small pieces" behavior would merge distinct policy sections together, undoing the "one chunk = one topic" design. Also relevant: LangChain's own more specialized `MarkdownHeaderTextSplitter` exists for exactly our markdown case, closer to what we built by hand — the generic splitter isn't even LangChain's own best answer for structured markdown.

Kept `chunks.jsonl` from Day 1, wrapped each entry as a `Document` (`page_content`, `metadata`, `id` — confirmed via `inspect.signature` that `Document` accepts `id` directly).

## The concrete abstraction-cost example (not just theory)
Built `classify_query` as `prompt | ChatOllama(reasoning=False) | StrOutputParser()` — clean, minimal. But piping through `StrOutputParser()` throws away the full `AIMessage`, including `response_metadata` (which holds `total_duration` etc.). Result: when we added timing instrumentation, `classify_query` showed `ollama=n/a` — we lost the ability to distinguish real model compute time from any surrounding overhead for that one call, purely because we chose the convenient one-line chain. `rerank` and `generate_answer` were deliberately built as `prompt | llm` (no parser) specifically to keep the full message and its `reasoning_content`/`response_metadata` — and sure enough, timing worked cleanly for those. Same tradeoff, now measured rather than assumed.

## `ChatOllama`'s field mapping — verified via source, not guessed
Researched (not assumed) what survives using `ChatOllama` vs. raw `requests.post`: essentially everything. `response_metadata` contains `total_duration`, `load_duration`, `prompt_eval_count`, `eval_count`, `done_reason`, etc. — same fields as the raw API, just under a named attribute instead of a flat dict. The one real config detail: Ollama's `thinking` field only lands cleanly in `additional_kwargs["reasoning_content"]` if you pass `reasoning=True` to `ChatOllama` — the LangChain-level name for Ollama's `think` parameter, confirmed via `ChatOllama.model_fields`. Left at the default (`None`), `<think>` tags stay embedded inside `.content` unextracted.

## Vector store insertion mechanics — same order-based pairing we built by hand, just automated
Verified from `langchain_chroma`'s source: `Chroma.add_texts()`/`from_documents()` extracts `texts`/`ids`/`metadatas` from the documents (same order), calls `embedding_function.embed_documents(texts)` **once** for the whole batch, then pairs `embeddings[idx]` with `texts[idx]`/`ids[idx]`/`metadatas[idx]` purely by index — exactly the zip-by-position pairing we hand-wrote in `store.py` on Day 1. LangChain isn't smarter here, just automated. One real difference: it calls the native collection's `.upsert()`, not `.add()` — meaning re-running ingestion with the same IDs *updates* existing entries, unlike Day 1's raw `chromadb.add()`, which silently no-ops on duplicate IDs.

## Metadata filter passthrough — verified, not assumed
`vectorstore.as_retriever(search_kwargs={"k": 10, "filter": {"type": "product"}})` does correctly exclude wrong-type chunks, confirmed by re-running the exact "good battery life" query with and without the filter — same fix for the same original problem as Day 1's raw `where=` clause, just through the retriever abstraction.
