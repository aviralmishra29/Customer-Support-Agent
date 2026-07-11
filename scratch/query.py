"""Zero-framework RAG query: classify -> retrieve top-10 (filtered) -> LLM rerank to top-3."""

import re
import time

import chromadb
import requests

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "boat_support"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:1.7b"
CANDIDATE_K = 10
FINAL_K = 3

CLASSIFY_PROMPT = """Classify the customer question into exactly one category:
- product: asking about product specs, features, comparisons, or recommendations
- policy: asking about returns, warranty, or shipping rules
- faq: asking how to do something, or troubleshooting an issue
- unsure: does not clearly fit any of the above

Question: {question}

Respond with ONLY the category word, nothing else."""

RERANK_PROMPT = """A customer asked: "{question}"

Here are {n} candidate pieces of information, numbered 1 to {n}:
{numbered_candidates}

Pick the {final_k} numbers that BEST answer the customer's question, ordered from most to least relevant.
Respond with ONLY the numbers separated by commas, e.g. "2,7,1". Do not explain."""

GENERATE_PROMPT = """You are a customer support assistant for boAt, a consumer electronics brand.
Answer the customer's question using ONLY the information in the context below.
If the context does not fully answer the question, say so honestly instead of guessing or making anything up.

Context:
{context}

Customer question: {question}

Answer:"""

TIMINGS = []


def record_timing(stage: str, wall_s: float, ollama_s: float | None = None) -> None:
    TIMINGS.append({"stage": stage, "wall_s": wall_s, "ollama_s": ollama_s})
    ollama_str = f", ollama={ollama_s:.2f}s" if ollama_s is not None else ""
    print(f"[timing] {stage}: wall={wall_s:.2f}s{ollama_str}")


def embed_query(text: str) -> list[float]:
    start = time.perf_counter()
    response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": [text]})
    response.raise_for_status()
    wall_s = time.perf_counter() - start

    response_json = response.json()
    vector = response_json["embeddings"][0]
    ollama_s = response_json.get("total_duration", 0) / 1e9
    record_timing("embed_query", wall_s, ollama_s)

    print(f"embed_query returned vector of length {len(vector)}")
    return vector


def classify_query(question: str) -> str:
    start = time.perf_counter()
    response = requests.post(OLLAMA_CHAT_URL, json={
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": CLASSIFY_PROMPT.format(question=question)}],
        "think": False,
        "stream": False,
    })
    response.raise_for_status()
    wall_s = time.perf_counter() - start

    response_json = response.json()
    category = response_json["message"]["content"].strip().lower()
    ollama_s = response_json.get("total_duration", 0) / 1e9
    record_timing("classify_query", wall_s, ollama_s)

    print(f"classify_query returned: {category}")
    return category


def retrieve_candidates(query: str, category: str, k: int = CANDIDATE_K) -> list[dict]:
    start = time.perf_counter()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    print(f"Connected to Chroma at '{CHROMA_DIR}'")

    collection = client.get_collection(COLLECTION_NAME)
    print(f"Retrieved collection '{COLLECTION_NAME}' with {collection.count()} items")

    query_vector = embed_query(query)

    where = {"type": category} if category in ("product", "policy", "faq") else None
    chroma_start = time.perf_counter()
    results = collection.query(query_embeddings=[query_vector], n_results=k, where=where)
    record_timing("chroma_query_only", time.perf_counter() - chroma_start)
    record_timing("retrieve_candidates_total (incl. embed)", time.perf_counter() - start)

    candidates = [
        {"id": id_, "text": doc, "metadata": meta, "distance": dist}
        for id_, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
    print(f"retrieve_candidates returned {len(candidates)} candidates (filter: {where})")
    return candidates


def rerank(question: str, candidates: list[dict], final_k: int = FINAL_K) -> list[dict]:
    numbered = "\n".join(f"{i + 1}. {c['text']}" for i, c in enumerate(candidates))
    prompt = RERANK_PROMPT.format(question=question, n=len(candidates), numbered_candidates=numbered, final_k=final_k)

    start = time.perf_counter()
    response = requests.post(OLLAMA_CHAT_URL, json={
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "think": True,
        "stream": False,
    })
    response.raise_for_status()
    wall_s = time.perf_counter() - start

    response_json = response.json()
    ollama_s = response_json.get("total_duration", 0) / 1e9
    record_timing("rerank", wall_s, ollama_s)

    print(f"rerank thinking: {response_json.get('message', {}).get('thinking', '')!r}")
    raw = response_json["message"]["content"]
    print(f"rerank raw response: {raw!r}")

    numbers = [int(n) for n in re.findall(r"\d+", raw)]
    valid_indices = [n - 1 for n in numbers if 1 <= n <= len(candidates)]

    seen = set()
    ordered_indices = [i for i in valid_indices if not (i in seen or seen.add(i))][:final_k]

    if len(ordered_indices) < final_k:
        print(f"WARNING: rerank parsing only found {len(ordered_indices)} valid picks, falling back to vector order for the rest")
        for i in range(len(candidates)):
            if i not in ordered_indices:
                ordered_indices.append(i)
            if len(ordered_indices) == final_k:
                break

    reranked = [candidates[i] for i in ordered_indices]
    print(f"rerank selected candidates: {[c['id'] for c in reranked]}")
    return reranked


def generate_answer(question: str, top_results: list[dict]) -> str:
    context = "\n".join(f"- {c['text']}" for c in top_results)
    prompt = GENERATE_PROMPT.format(context=context, question=question)

    start = time.perf_counter()
    response = requests.post(OLLAMA_CHAT_URL, json={
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "think": True,
        "stream": False,
    })
    response.raise_for_status()
    wall_s = time.perf_counter() - start

    response_json = response.json()
    ollama_s = response_json.get("total_duration", 0) / 1e9
    record_timing("generate_answer", wall_s, ollama_s)

    print(f"generate_answer thinking: {response_json.get('message', {}).get('thinking', '')!r}")

    answer = response_json["message"]["content"].strip()
    print(f"generate_answer returned: {answer!r}")
    return answer


if __name__ == "__main__":
    overall_start = time.perf_counter()

    question = "which earbuds have good battery life"
    print(f"Query: {question}")

    category = classify_query(question)
    candidates = retrieve_candidates(question, category)

    print("\nAll candidates before rerank:")
    for i, c in enumerate(candidates):
        print(f"{i + 1}. [distance={c['distance']:.4f}] {c['text']}")

    top_results = rerank(question, candidates)

    answer = generate_answer(question, top_results)
    overall_wall_s = time.perf_counter() - overall_start

    print("\nResponse:")
    print(answer)

    print("\n--- Timing summary ---")
    for t in TIMINGS:
        ollama_str = f"{t['ollama_s']:.2f}s" if t["ollama_s"] is not None else "n/a"
        print(f"{t['stage']:38s} wall={t['wall_s']:.2f}s  ollama={ollama_str}")
    print(f"{'TOTAL (end-to-end)':38s} wall={overall_wall_s:.2f}s")
