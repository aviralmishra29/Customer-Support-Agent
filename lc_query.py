"""LangChain rebuild of scratch/query.py: classify -> retrieve (filtered) -> LCEL rerank -> LCEL generate."""

import re
import sys
import time
from enum import Enum
from pathlib import Path

import httpx
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

sys.path.insert(0, str(Path(__file__).parent / "ingestion"))
from cached_embeddings import CachedEmbeddings

CHROMA_DIR = Path(__file__).parent / "data" / "chroma"
COLLECTION_NAME = "boat_support_prod_db"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:1.7b"

# Generous enough to never cut off a legitimate slow reasoning call (observed up to ~193s on
# confusing input during adversarial testing), but bounded - without this, a hung/unreachable
# Ollama would hang a node forever instead of failing in a way that can be caught and handled.
OLLAMA_TIMEOUT_SECONDS = 240


def make_llm(reasoning: bool) -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL, reasoning=reasoning, client_kwargs={"timeout": OLLAMA_TIMEOUT_SECONDS})


@retry(
    # Only retry a fast-failing connection refusal (Ollama not up yet / restarting) - NOT a
    # timeout, since a timeout already means we waited the full OLLAMA_TIMEOUT_SECONDS once;
    # retrying that would double the user's wait for a very unlikely-to-change outcome. Better
    # to fail fast to a safe fallback than make them wait twice.
    retry=retry_if_exception_type(httpx.ConnectError),
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    reraise=True,
)
def invoke_with_retry(invocable, *args, **kwargs):
    return invocable(*args, **kwargs)


CANDIDATE_K = 10
FINAL_K = 3

class QueryCategory(str, Enum):
    PRODUCT = "product"
    COMPARISON = "comparison"
    POLICY = "policy"
    FAQ = "faq"
    UNSURE = "unsure"


class ClassifyDecision(BaseModel):
    category: QueryCategory = Field(description="Which category this question falls into.")


CLASSIFY_PROMPT = ChatPromptTemplate.from_template("""Classify the customer question into exactly one category:
- product: asking about product specs, features, or recommendations for a SINGLE product or general product questions
- comparison: asking to compare TWO OR MORE specific named products against each other (e.g. "X vs Y", "which is better, X or Y", "difference between X and Y")
- policy: asking about returns, warranty, or shipping rules
- faq: asking how to do something, or troubleshooting an issue
- unsure: does not clearly fit any of the above (this node should only ever be reached for
  genuine boAt product/order support questions - the graph-level router already filters out
  off-topic and instruction-override messages before they reach here)

Question: {question}""")

PRODUCT_MENTION_PROMPT = ChatPromptTemplate.from_template("""Extract the product names mentioned in this customer question, exactly as written by the customer.

Question: {question}

Respond with ONLY the product names, one per line, nothing else.""")

COMPARISON_PROMPT = ChatPromptTemplate.from_template("""You are a customer support assistant for boAt, a consumer electronics brand.
A customer wants to compare these products. Use ONLY the exact specs given below - do not invent,
guess, or adjust any number. If a spec is missing for a product, say it's not available rather than guessing.

{product_specs}

Customer question: {question}

Give a short comparison answer, calling out the key differences relevant to the question.""")

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

POLICY_GENERATE_PROMPT = ChatPromptTemplate.from_template("""You are a customer support assistant for boAt, a consumer electronics brand.
Answer the customer's question using ONLY the information in the context below.
If the context does not fully answer the question, say so honestly instead of guessing or making anything up.

For any specific number, deadline, or condition (e.g. "7 days", "48 hours", "unused and in original
packaging"), quote the exact sentence from the context in double quotes - do not paraphrase or
reword these specifics. You may write normal prose around the quotes.

Context:
{context}

Customer question: {question}

Answer:""")

POLICY_UNGROUNDED_FALLBACK = (
    "I want to make sure I give you accurate policy information, and I'm not fully confident "
    "in the answer I was about to give. Let me connect you with our support team for this one."
)

PRODUCT_UNGROUNDED_FALLBACK = (
    "I want to make sure I only give you accurate specs, and I'm not fully confident in the "
    "answer I was about to give. Could you rephrase, or ask about the product directly?"
)

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> set[str]:
    return set(NUMBER_PATTERN.findall(text))

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
    """Structured output (not StrOutputParser) - a malformed/off-script string here used to
    silently disable retrieve_candidates's type filter entirely (fell through its `in (...)`
    check to "search everything, unfiltered"). An enum makes that structurally impossible,
    same fix already applied to the graph-level router in graph/nodes.py."""
    llm = make_llm(reasoning=False)
    structured_llm = llm.with_structured_output(ClassifyDecision)
    chain = CLASSIFY_PROMPT | structured_llm

    start = time.perf_counter()
    decision = invoke_with_retry(chain.invoke, {"question": question})
    category = decision.category.value
    record_timing("classify_query", time.perf_counter() - start)

    print(f"classify_query returned: {category}")
    return category


def extract_product_mentions(question: str) -> list[str]:
    """Free-text extraction only - these names are NOT trusted as real catalog identifiers.
    resolve_product_mentions() below does the actual exact matching, in code."""
    llm = make_llm(reasoning=False)
    chain = PRODUCT_MENTION_PROMPT | llm | StrOutputParser()

    start = time.perf_counter()
    raw = invoke_with_retry(chain.invoke, {"question": question})
    record_timing("extract_product_mentions", time.perf_counter() - start)

    mentions = [line.strip() for line in raw.splitlines() if line.strip()]
    print(f"extract_product_mentions returned: {mentions}")
    return mentions


def _normalize_product_name(name: str) -> str:
    return re.sub(r"\bboat\b", "", name.lower()).strip()


def resolve_product_mentions(vectorstore: Chroma, mentions: list[str]) -> tuple[list[dict], list[str]]:
    """Resolves free-text mentions to real catalog SKUs by substring match against the
    actual product catalog - never by trusting the LLM's spelling/exact wording. Same
    not-trust-the-model's-output principle as order_number validation: the LLM is only
    good enough to point roughly at a product, code decides what that product actually is.
    """
    catalog = vectorstore.get(where={"type": "product"})["metadatas"]

    resolved_skus = []
    unresolved = []
    for mention in mentions:
        norm_mention = _normalize_product_name(mention)
        matches = [
            m for m in catalog
            if norm_mention and (
                norm_mention in _normalize_product_name(m["name"])
                or _normalize_product_name(m["name"]) in norm_mention
                or norm_mention in m["sku"].lower()
            )
        ]
        if matches:
            best = max(matches, key=lambda m: len(m["name"]))
            if best["sku"] not in resolved_skus:
                resolved_skus.append(best["sku"])
            print(f"resolve_product_mentions: {mention!r} -> {best['sku']!r} ({best['name']!r})")
        else:
            unresolved.append(mention)
            print(f"resolve_product_mentions: {mention!r} -> no match")

    if not resolved_skus:
        return [], unresolved

    # Exact structured fetch by SKU - zero similarity search involved, same principle as
    # the order-lookup tool reading a DB row directly instead of guessing from prose.
    exact = vectorstore.get(where={"sku": {"$in": resolved_skus}})["metadatas"]
    return exact, unresolved


def compare_products(question: str, vectorstore: Chroma) -> str:
    mentions = extract_product_mentions(question)
    products, unresolved = resolve_product_mentions(vectorstore, mentions)

    if len(products) == 0:
        return "I couldn't find those products in our catalog. Could you double-check the exact names?"
    if len(products) == 1:
        return (
            f"I could only find '{products[0]['name']}' in our catalog. "
            "Could you give the exact name of the other product you'd like to compare it to?"
        )

    spec_keys = sorted({key for p in products for key in p if key not in ("type",)})
    lines = []
    for p in products:
        details = ", ".join(f"{key}={p[key]}" for key in spec_keys if key in p)
        lines.append(f"{p['name']}: {details}")
    product_specs = "\n".join(lines)
    print(f"compare_products specs block:\n{product_specs}")

    llm = make_llm(reasoning=True)
    chain = COMPARISON_PROMPT | llm

    start = time.perf_counter()
    response = invoke_with_retry(chain.invoke, {"product_specs": product_specs, "question": question})
    wall_s = time.perf_counter() - start
    ollama_s = response.response_metadata.get("total_duration", 0) / 1e9
    record_timing("compare_products", wall_s, ollama_s)

    answer = response.content.strip()
    print(f"compare_products returned: {answer!r}")
    return answer


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

    llm = make_llm(reasoning=True)
    chain = RERANK_PROMPT | llm

    start = time.perf_counter()
    response = invoke_with_retry(chain.invoke, {"question": question, "n": len(candidates), "numbered_candidates": numbered, "final_k": final_k})
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

    llm = make_llm(reasoning=True)
    chain = GENERATE_PROMPT | llm

    start = time.perf_counter()
    response = invoke_with_retry(chain.invoke, {"context": context, "question": question})
    wall_s = time.perf_counter() - start
    ollama_s = response.response_metadata.get("total_duration", 0) / 1e9
    record_timing("generate_answer", wall_s, ollama_s)

    print(f"generate_answer reasoning: {response.additional_kwargs.get('reasoning_content', '')!r}")
    answer = response.content.strip()
    print(f"generate_answer returned: {answer!r}")

    # Numeric groundedness check: lighter than policy's exact-quote verification (product/FAQ
    # answers are meant to read as natural prose, not verbatim-quoted) - every number the
    # answer states (battery hours, price, IP rating digits, etc.) must actually appear
    # somewhere in the retrieved context, or it's treated as a possibly fabricated spec.
    context_numbers = _extract_numbers(context)
    answer_numbers = _extract_numbers(answer)
    fabricated = answer_numbers - context_numbers
    if fabricated:
        print(f"generate_answer: UNGROUNDED numbers {fabricated} not found in retrieved context - falling back")
        return PRODUCT_UNGROUNDED_FALLBACK
    print(f"generate_answer: {len(answer_numbers)} number(s) in answer, all verified against context")

    return answer


def _normalize_for_quote_check(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(".:,")


def generate_policy_answer(question: str, top_results: list) -> str:
    """Policy answers need exact-wording grounding, not just topical relevance: a paraphrase
    can silently drop a condition ("unused and in original packaging") or soften a number
    ("7 days" -> "about a week"). POLICY_GENERATE_PROMPT asks the model to quote specifics
    verbatim; the check below verifies it actually did, in code, since a prompt instruction
    alone is never trusted to guarantee compliance - same principle as every other exact-fact
    check in this project (auth, order lookups, comparison specs).
    """
    context = "\n".join(f"- {c.page_content}" for c in top_results)

    llm = make_llm(reasoning=True)
    chain = POLICY_GENERATE_PROMPT | llm

    start = time.perf_counter()
    response = invoke_with_retry(chain.invoke, {"context": context, "question": question})
    wall_s = time.perf_counter() - start
    ollama_s = response.response_metadata.get("total_duration", 0) / 1e9
    record_timing("generate_policy_answer", wall_s, ollama_s)

    print(f"generate_policy_answer reasoning: {response.additional_kwargs.get('reasoning_content', '')!r}")
    answer = response.content.strip()
    print(f"generate_policy_answer raw answer: {answer!r}")

    # Groundedness check: every quoted span the model produced must actually appear in the
    # real retrieved context - not fuzzy-matched, an exact (whitespace/punctuation-normalized)
    # substring check. A quote that doesn't check out means the model claimed to cite
    # something that isn't there, which is treated as an ungrounded answer, not a near-miss.
    normalized_context = _normalize_for_quote_check(context)
    quoted_spans = re.findall(r'"([^"]+)"', answer)
    for quote in quoted_spans:
        if _normalize_for_quote_check(quote) not in normalized_context:
            print(f"generate_policy_answer: UNGROUNDED quote {quote!r} not found in retrieved context - falling back")
            return POLICY_UNGROUNDED_FALLBACK
    print(f"generate_policy_answer: {len(quoted_spans)} quoted span(s), all verified against context")

    # Citation attached by code from chunk metadata, never self-reported by the LLM -
    # it could misname or invent a source doc, so it's never asked to. Only attached when
    # the answer actually used a verified quote - a real bug found via adversarial testing
    # (a "the context doesn't contain that information" refusal was still getting a
    # "Source: ..." line listing docs that were retrieved but never actually used).
    if quoted_spans:
        doc_titles = sorted({c.metadata["doc"] for c in top_results if "doc" in c.metadata})
        if doc_titles:
            answer = f"{answer}\n\nSource: {', '.join(doc_titles)}"

    print(f"generate_policy_answer returned: {answer!r}")
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
