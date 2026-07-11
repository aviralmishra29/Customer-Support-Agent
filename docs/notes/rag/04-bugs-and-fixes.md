# Bugs and problems hit today, and how they were resolved

Kept separate from the concept notes because "what broke and how you fixed it" is its own valuable thing to be able to explain — this is exactly what interview questions like "tell me about a bug you debugged" are probing for.

## 1. Markdown bold markers survived asymmetrically (`**Stop using**` → `Stop using**`)
**Symptom:** policy chunk text had stray `**` left in it, but only trailing ones, not leading.
**Root cause:** the bullet-marker stripper used one overloaded character-strip (`lstrip("-*0123456789. ")`) to remove list markers like `"- "` or `"1. "`. Since `*` was in that strip set, it also silently ate a *leading* `**` from bold text, but had no way to catch the *trailing* `**` — asymmetric by construction.
**Fix:** replaced the overloaded strip with two dedicated regexes — one specifically for bold markers (`\*\*(.+?)\*\*` → captured inner text), one specifically for bullet markers (`^(?:[-*]|\d+\.)\s+`) — applied as separate, purpose-built passes instead of one catch-all character strip.

## 2. Copied data files changed unexpectedly after copying
**Symptom:** `return-policy.md` and `products.json` in `data/` differed from their source in the sibling project, with `return-policy.md` showing garbled/spliced text (a title merged with unrelated text from a different policy file).
**Investigation:** diffed every copied file against source, checked mtimes to confirm the destination files were modified *after* the copy — ruled out that any script/edit tool had touched them.
**Resolution:** turned out to be the user's own manual edits, not corruption. Lesson: always ask before assuming a change is a bug and "fixing" (i.e. overwriting) something someone else touched intentionally.

## 3. Query classification failed on an easy case
**Symptom:** "which earbuds have good battery life" — an unambiguous product question — classified as `"unsure"`.
**Root cause:** the prompt listed category *names* only (`product, policy, faq, unsure`) with no description of what belongs in each — the model had nothing to actually reason from.
**Fix:** added a one-line description per category to the prompt. Re-verified against 5 hand-picked examples spanning all 3 real categories — 5/5 correct after the fix. General lesson (same one as the `gemma3` router failure from earlier): bare category names are not a reliable classification prompt; description/few-shot context is load-bearing, not optional.

## 4. Reranking silently skipped the actual best answer
**Symptom:** for "good battery life," a tied-*worst*-battery product (35h) got ranked into the top-3, while the actual best (68h) was excluded entirely.
**Root cause:** reranking used `qwen3:1.7b` with `think:false` — fast, shallow pattern-matching, not real reasoning. Comparing 10 numeric values and finding the true max needs deliberate step-by-step reasoning, which `think:false` skips for speed.
**Fix:** switched reranking specifically to `think:true`. Its chain-of-thought explicitly listed every candidate's battery number and sorted them, correctly surfacing the 68h product as #1 on the next run. Not perfect — one run still mis-picked among a 3-way tie at 50h — but the major miss (excluding the actual best answer) was fixed. Tradeoff accepted: ~20-25s extra latency for that one call, worth it for correctness on this project.

## 5. Retrieval ranked complaints above the actual good product
**Symptom:** top vector-similarity results for "good battery life" were two FAQ entries *complaining* about battery drain, with the genuinely good-battery product ranked 3rd.
**Root cause:** embeddings measure topical closeness, not sentiment/intent — "battery drains quickly" and "good battery life" are both heavily about the same topic, so they land close in vector space regardless of polarity. Confirmed this wasn't fixable by switching the distance metric (see #6) — the vectors themselves, not the comparison formula, encode this closeness.
**Fix:** added intent classification before retrieval, filtering to `type == "product"` via Chroma's `where` clause — excludes FAQ/policy chunks from the candidate pool entirely for product-type queries, rather than trying to out-rank them after the fact.

## 6. Suspected distance-metric mismatch — investigated, turned out to be a non-issue
**Symptom:** noticed Chroma was never explicitly configured with `hnsw:space: cosine`, despite having earlier reasoned that cosine was the "right" choice.
**Investigation:** checked whether this actually mattered by measuring the L2 norm of our stored vectors — found they're unit-normalized (length ≈ 1.0).
**Conclusion:** for unit-length vectors, L2 and cosine distance are related by a fixed monotonic formula, so they produce identical rankings. A real gap between stated design intent and actual config, but empirically harmless here — worth verifying rather than either assuming it was fine *or* assuming it needed fixing.
