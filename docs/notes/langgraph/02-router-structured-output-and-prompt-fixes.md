# Graph-level router: structured output, why keyword-gating was wrong, and prompt-quality fixes

## Why the router needs structured output, not plain-string parsing

`classify_query` (RAG-internal) gets away with "answer with exactly one word" + `StrOutputParser()` because if the model slips format (extra word, wrong case), the cost is a slightly worse retrieval filter. Annoying, not dangerous.

The graph-level router's output decides which **entire subsystem** runs, via a conditional edge that does a plain dict lookup (`{"ORDER": order_node, ...}`). A malformed string here either throws `KeyError` or silently misroutes — a categorically worse failure than a bad RAG filter, including for the one route (`ESCALATION`) where a miss has real safety stakes.

**Fix:** constrain the output at the schema level, not just via prompt discipline.

```python
class Route(str, Enum):
    PRODUCT_RAG = "PRODUCT_RAG"
    ORDER = "ORDER"
    ESCALATION = "ESCALATION"
    AUTH = "AUTH"
    FALLBACK = "FALLBACK"

class RouterDecision(BaseModel):
    route: Route = Field(description="Which subsystem should handle this user message.")
```

`Route(str, Enum)` makes non-conformant values structurally impossible rather than just discouraged. `RouterDecision` wraps it in a `BaseModel` because `.with_structured_output()` expects a schema, and it leaves room to add fields later (e.g. `confidence`) without changing the mechanism.

**Verified, not assumed:** `qwen3:1.7b` + `.with_structured_output(RouterDecision)` was tested against 9 deliberately adversarial prompts (`tests/test_router_structured_output.py`) — every single call returned a valid, schema-conformant `RouterDecision`, no exceptions, no out-of-enum values. The mechanism holds up on this model. What it does *not* guarantee is that the classification itself is *correct* — see below, that's a separate, prompt-quality problem.

## Why regex-only escalation gating is wrong (caught via an adversarial example)

Original design (see `01-...md`) planned a deterministic keyword regex that would **hard-route** straight to `ESCALATION` before any LLM saw the message. Real test case that broke this: *"These earbuds are so good they really feel like fire, suggest me more like this"* — a keyword match on "fire" with zero actual hazard.

Researched how production systems actually handle this (web search, not assumed) rather than guessing. Consistent pattern found across guardrail/moderation literature:
- Regex/keyword matches should be a **signal fed into the classifier's prompt**, never a standalone hard gate — the same false-positive problem shows up in content moderation generally (e.g. "kill" in "you're killing it").
- Confidence-banded escalation (auto-handle above a threshold / auto-escalate to human below it / a middle band that still answers but flags) is the broader pattern — noted for later, not built this session.
- Cost-aware escalation — cheap deterministic/fast checks handle the obvious majority, expensive reasoning only runs on genuinely uncertain cases — matches our existing `think:false` (router) vs. `think:true` (generation) split, just not yet exploited for this purpose.

**What we actually built:** the regex scan still runs first, but only produces a boolean hint (`safety_keyword_hit`) injected into the router's prompt as context ("you may be told whether a safety-related keyword was detected... use it as a signal, not a verdict"). The LLM reads the *whole message* and decides. Verified this correctly distinguishes the figurative earbuds case (`safety_keyword_hit=True` → `PRODUCT_RAG`) from a real hazard with the exact same keyword hit (`safety_keyword_hit=True` → `ESCALATION`).

## Classification-quality bugs found via adversarial testing (separate from the format problem above)

Structured output fixes *format* reliability; it says nothing about whether the category chosen is *correct*. Two real misclassifications surfaced, same root cause pattern already seen once before (Hinglish, `rag/06-multilingual-hindi-hinglish.md`): **bare/narrow category descriptions aren't enough for edge-case phrasing.**

1. **Figurative "fire" + recommendation request → misfired to `FALLBACK`.** The `PRODUCT_RAG` description only mentioned "questions about products," not recommendation requests or figurative praise. Fixed by explicitly adding both to the description. Reran, fixed.
2. **Sarcastic negative complaint ("explosive quality" as sarcasm about a dead battery) → misfired to `FALLBACK`.** Correctly avoided false-`ESCALATION`, but overcorrected into "doesn't fit any category" instead of recognizing it as a product complaint. **Not yet fixed** — noted as a real, live gap. Likely same fix shape: explicitly state that negative/sarcastic product complaints are still `PRODUCT_RAG`, not just praise.

## Known, accepted gap: single-route schema can't express blended intent

Real test message: *"My order #778 hasn't arrived yet, and also is the Airdopes 141 good for gym use?"* — genuinely two intents (ORDER + PRODUCT_RAG). The schema forces exactly one `route`, so the router picked only `ORDER`, silently dropping the product question.

**Decision (user's call, 2026-07-11): acceptable for now.** Fixing this properly would mean either `routes: list[Route]` plus a fan-out/merge mechanism in the graph, or a primary+secondary field with the primary node checking for a secondary intent before finishing. Not solved this session — noted as a structural gap to revisit, not a bug to silently patch around.
