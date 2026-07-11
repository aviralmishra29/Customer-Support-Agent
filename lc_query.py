"""LangChain rebuild of scratch/query.py: classify -> retrieve (filtered) -> LCEL rerank -> LCEL generate."""

import re
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

sys.path.insert(0, str(Path(__file__).parent / "ingestion"))
from cached_embeddings import CachedEmbeddings

CHROMA_DIR = Path(__file__).parent / "data" / "chroma"
COLLECTION_NAME = "boat_support_prod_db"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:1.7b"
CANDIDATE_K = 10
FINAL_K = 3

CLASSIFY_PROMPT = ChatPromptTemplate.from_template("""Classify the customer question into exactly one category:
- product: asking about product specs, features, comparisons, or recommendations
- policy: asking about returns, warranty, or shipping rules
- faq: asking how to do something, or troubleshooting an issue
- unsure: does not clearly fit any of the above

Question: {question}

Respond with ONLY the category word, nothing else.""")

RERANK_PROMPT = ChatPromptTemplate.from_template("""A customer asked: "{question}"

Here are {n} candidate pieces of information, numbered 1 to {n}:
{numbered_candidates}

Pick the {final_k} numbers that BEST answer the customer's question, ordered from most to least relevant.
Respond with ONLY the numbers separated by commas, e.g. "2,7,1". Do not explain.""")

GENERATE_PROMPT = ChatPromptTemplate.from_template("""You are a customer support assistant for boAt, a consumer electronics brand.
Answer the customer's question using ONLY the information in the context below.
If the context does not fully answer the question, say so honestly instead of guessing or making anything up.

Context:
{context}

Customer question: {question}

Answer:""")

TIMINGS = []


def record_timing(stage: str, wall_s: float, ollama_s: float | None = None) -> None:
    TIMINGS.append({"stage": stage, "wall_s": wall_s, "ollama_s": ollama_s})
    ollama_str = f", ollama={ollama_s:.2f}s" if ollama_s is not None else ""
    print(f"[timing] {stage}: wall={wall_s:.2f}s{ollama_str}")


def get_vectorstore() -> Chroma:
    base_embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    embeddings = CachedEmbeddings(base_embeddings, namespace=EMBED_MODEL)
    print(f"Connecting to Chroma collection '{COLLECTION_NAME}' at '{CHROMA_DIR}'")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    print(f"Connected. Collection count: {vectorstore._collection.count()}")
    return vectorstore


def classify_query(question: str) -> str:
    llm = ChatOllama(model=CHAT_MODEL, reasoning=False)
    chain = CLASSIFY_PROMPT | llm | StrOutputParser()

    start = time.perf_counter()
    category = chain.invoke({"question": question}).strip().lower()
    record_timing("classify_query", time.perf_counter() - start)

    print(f"classify_query returned: {category}")
    return category


def retrieve_candidates(vectorstore: Chroma, question: str, category: str, k: int = CANDIDATE_K) -> list:
    where = {"type": category} if category in ("product", "policy", "faq") else None
    retriever = vectorstore.as_retriever(search_kwargs={"k": k, "filter": where} if where else {"k": k})

    start = time.perf_counter()
    candidates = retriever.invoke(question)
    record_timing("retrieve_candidates", time.perf_counter() - start)

    print(f"retrieve_candidates returned {len(candidates)} candidates (filter: {where})")
    return candidates


def rerank(question: str, candidates: list, final_k: int = FINAL_K) -> list:
    numbered = "\n".join(f"{i + 1}. {c.page_content}" for i, c in enumerate(candidates))

    llm = ChatOllama(model=CHAT_MODEL, reasoning=True)
    chain = RERANK_PROMPT | llm

    start = time.perf_counter()
    response = chain.invoke({"question": question, "n": len(candidates), "numbered_candidates": numbered, "final_k": final_k})
    wall_s = time.perf_counter() - start
    ollama_s = response.response_metadata.get("total_duration", 0) / 1e9
    record_timing("rerank", wall_s, ollama_s)

    print(f"rerank reasoning: {response.additional_kwargs.get('reasoning_content', '')!r}")
    raw = response.content
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
    print(f"rerank selected candidates: {[c.id for c in reranked]}")
    return reranked


def generate_answer(question: str, top_results: list) -> str:
    context = "\n".join(f"- {c.page_content}" for c in top_results)

    llm = ChatOllama(model=CHAT_MODEL, reasoning=True)
    chain = GENERATE_PROMPT | llm

    start = time.perf_counter()
    response = chain.invoke({"context": context, "question": question})
    wall_s = time.perf_counter() - start
    ollama_s = response.response_metadata.get("total_duration", 0) / 1e9
    record_timing("generate_answer", wall_s, ollama_s)

    print(f"generate_answer reasoning: {response.additional_kwargs.get('reasoning_content', '')!r}")
    answer = response.content.strip()
    print(f"generate_answer returned: {answer!r}")
    return answer


if __name__ == "__main__":
    overall_start = time.perf_counter()

    question = "which earbuds have good battery life"
    print(f"Query: {question}")

    vectorstore = get_vectorstore()

    category = classify_query(question)
    candidates = retrieve_candidates(vectorstore, question, category)

    print("\nAll candidates before rerank:")
    for i, c in enumerate(candidates):
        print(f"{i + 1}. {c.page_content}")

    top_results = rerank(question, candidates)
    answer = generate_answer(question, top_results)
    overall_wall_s = time.perf_counter() - overall_start

    print("\nResponse:")
    print(answer)

    print("\n--- Timing summary ---")
    for t in TIMINGS:
        ollama_str = f"{t['ollama_s']:.2f}s" if t["ollama_s"] is not None else "n/a"
        print(f"{t['stage']:20s} wall={t['wall_s']:.2f}s  ollama={ollama_str}")
    print(f"{'TOTAL (end-to-end)':20s} wall={overall_wall_s:.2f}s")
