# Multilingual support: Hindi/Hinglish — tested, not assumed

## Why this came up
Our FAQ corpus (`data/seed/faqs.json`) has 12 non-English entries: 2 pure Hindi (Devanagari script), 10 "Hinglish" (Hindi phrased in Latin script, e.g. "Mera order kaise track karun?" — extremely common register in real Indian D2C support). Realized our entire pipeline (`classify_query`, `rerank`, `generate` prompts) is written in English with zero language-handling instructions, and had never been tested against this content despite it sitting in our corpus since Day 1.

## Test setup
Ran the real `lc_query.py` pipeline (not a mock) against two Hinglish questions:
1. *"Mera order kaise track karun?"* — closely matches an already-indexed FAQ.
2. *"Mera order abhi tak nahi aaya, kaise pata karu?"* — same underlying intent (order tracking), fresh/novel phrasing not indexed verbatim.

## Finding 1 (encouraging): embeddings show real cross-lingual understanding
For query 1, the top-3 retrieved candidates included **both** the Hinglish FAQ *and* its English equivalent ("How can I track my order?") — `nomic-embed-text` placed the same underlying concept close together across languages, not just within one. Reranking correctly picked the genuinely relevant 3 out of 10 and excluded clearly unrelated ones (battery, warranty, COD), reasoning coherently in English about Hinglish input.

## Finding 2 (real gap): generation defaults to English regardless of input language
Despite understanding the Hinglish question correctly (visible in its own reasoning trace, which translates it accurately), `generate_answer` replied in English — it even echoed the original Hinglish question back as a heading, then answered entirely in English underneath. This is a missing instruction, not a model limitation: nothing in `GENERATE_PROMPT` says to match the input language, so the model has no signal to do so, and — at least in this test — didn't do it on its own either. Cheapest concrete fix identified: explicitly instruct generation to reply in the same language as the question.

## Finding 3 (real gap): classification/retrieval is fragile for novel Hinglish phrasing
Query 2 (same intent as query 1, different wording) **misclassified as `"policy"` instead of `"faq"`.** This one wrong label cascaded through the entire pipeline: the metadata filter then restricted retrieval to policy-only chunks (return/warranty/safety-SOP), **excluding the correct FAQ entirely before similarity search even ran** — same "a wrong filter can hurt more than no filter" risk flagged back on Day 1, now observed for real. Reranking's own reasoning shows visible uncertainty ("I'm not entirely sure... but I think...") because it was forced to choose among candidates none of which were actually relevant. Generation, to its credit, handled the resulting bad context honestly — it recognized the context didn't answer the question and said so, rather than hallucinating.

## Root cause, both gaps
Everything upstream of generation (`classify_query`, `rerank`) reasons in English about English-described categories, with no Hinglish examples or awareness baked into the prompts — same fix pattern that already worked once before (Day 1's bare-category-names classification failure was fixed by adding real category *descriptions*; Hinglish robustness would likely need the analogous fix — few-shot Hinglish examples in the classify prompt, or a fallback to unfiltered search when classification confidence looks shaky).

## Status
Logged as a follow-up, not fixed yet. Concrete next steps if/when picked up: (1) add an explicit language-matching instruction to `GENERATE_PROMPT` — cheap, likely high-value; (2) add Hinglish examples to `CLASSIFY_PROMPT` or add a lower-confidence fallback (e.g. don't filter if the model's classification seems uncertain) to reduce the "wrong filter excludes the right answer" risk.
