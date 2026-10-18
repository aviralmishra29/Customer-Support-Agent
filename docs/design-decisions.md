# Design Decisions

Reference material for interview prep. Each section: the decision, the real reasoning behind
it (pulled from this project's own history, not reconstructed after the fact), and the
alternative that was considered and rejected. Rehearsal questions are at the bottom — try
answering them cold before checking back against the sections above, then update this doc if
your own phrasing is better or you spot a gap.

---

## 1. Why LangGraph over a plain LCEL chain

**Decision:** build the agent as a `StateGraph` (nodes/edges/conditional routing), not a single
LCEL pipeline (`prompt | llm | parser`).

**Why:** an LCEL chain is linear — step 1 always leads to step 2. This agent needs to *branch*:
a product question goes to RAG, an order question goes to a DB lookup, a safety complaint goes
to escalation. That's a structural requirement a chain can't express at all, not a style
preference. LangGraph adds three things a chain doesn't have: **State** (shared data flowing
through the whole run), **nodes** (functions that read state and return partial updates), and
**conditional edges** (first-class branching logic instead of an informal `if` buried in code).

**Alternative considered:** keep everything as LCEL chains with Python `if/else` gluing them
together manually. Rejected because the routing logic itself needed to be a first-class,
inspectable part of the system (visualizable, testable in isolation) — not scattered
conditional logic a reader has to reconstruct from call sites.

---

## 2. Two-level hierarchical routing

**Decision:** a graph-level router (`PRODUCT_RAG`/`ORDER`/`ESCALATION`/`AUTH`/`FALLBACK`) decides
*which subsystem* handles a message; a separate RAG-internal classifier (`classify_query`:
`product`/`comparison`/`policy`/`faq`/`unsure`) decides *which slice of the product corpus*.

**Why:** these are different *kinds* of decisions, not the same decision asked twice. Cramming
both into one flat classifier risks the confusion already seen with Hinglish phrasing on just
4 categories — adding a 9th category to one classifier makes every decision harder, not easier.
This also matches real production practice: different branches have different cost/ownership/
safety profiles (a misrouted `ESCALATION` is a safety issue; a misrouted RAG category is a
quality issue), so cheap/deterministic checks should funnel first, with the LLM only deciding
what it actually needs to.

**Alternative considered:** one flat classifier with all categories. Rejected for the reason
above, and because `PRODUCT_RAG` vs `ORDER` vs `ESCALATION` genuinely need different enforcement
levels (auth-gating only applies to `ORDER`; `ESCALATION` has safety stakes none of the others
do) — one classifier flattens those differences away.

---

## 3. Structured output over free-text parsing

**Decision:** the graph-level router and `classify_query` both use `.with_structured_output()` +
a Pydantic enum, not `StrOutputParser()` returning a bare string. Same principle applied again
to tool arguments (`OrderLookupArgs` with a regex-constrained `order_number`).

**Why:** a malformed route string breaks a dict-based conditional edge — either a crash or a
silent misroute, categorically worse than a slightly-off RAG filter. This wasn't theoretical:
`classify_query`'s original `StrOutputParser()` version, under adversarial testing, once echoed
`"injection_successful"` back as a "category" — which fell outside the retrieval filter's
allow-list and silently disabled type-filtering entirely, searching the whole corpus unfiltered.
An enum makes that structurally impossible instead of merely unlikely.

**Alternative considered:** keep the router's output loose and defensively `.get(route,
fallback)` in the edge function. Partially done anyway as a second safety net, but the real fix
is preventing the bad value from being producible at all, not just catching it after the fact.

---

## 4. Chunking strategy — structural, not fixed-size

**Decision:** one JSON record = one product chunk; Markdown `##` headers = policy section
boundaries. Raw JSON/Markdown syntax converted to natural-language sentences before embedding
(`{"anc": false}` → "no ANC").

**Why:** a generic fixed-size/overlap splitter (`RecursiveCharacterTextSplitter`) would merge
distinct policy sections or split a product's specs mid-sentence, since it doesn't know where
the real topic boundaries are — JSON has no blank-line boundaries to exploit, and a return
policy's "Non-returnable items" section is a genuinely different topic from "Refund timeline"
even though they sit in the same file. Raw JSON syntax is also out-of-distribution noise for a
text embedding model — `{"anc": false}` means nothing to it, "no ANC" does.

**Alternative considered:** LangChain's `RecursiveCharacterTextSplitter`, evaluated directly
during the Weekend 1b LangChain rebuild and rejected — the hand-rolled structural chunking was
kept deliberately rather than replaced, one of the only pieces of the original zero-framework
build that survived the framework migration unchanged.

---

## 5. Why auth is graph-enforced, never a tool

**Decision:** `is_authenticated(state)` is called from `graph/edges.py`'s conditional edge — never
exposed as an LLM-callable tool, never checked inside a node the model's output flows through.

**Why:** a tool is still something the LLM *decides* whether to call. If "check auth" were a
tool, the model could skip calling it, be prompt-injected into skipping it, or simply fail to
reason that it should — the same class of fragility as putting a safety rule in a system prompt
and hoping it's followed. Putting the gate in the conditional edge means it runs on **every**
request as part of the graph's wiring — the only way to bypass it is to change the graph's code
itself. This is the interview-gold framing: model-enforced auth is prompt-injection-bypassable,
code/graph-enforced is not.

**Alternative considered:** a `check_is_authenticated()` tool bound alongside `get_order_status`.
Rejected outright during the design phase, before any code was written, for the reason above.

---

## 6. Escalation design — signal not gate, then later descoped

**Decision (Weekend 2):** a safety-keyword regex doesn't hard-route to `ESCALATION` — it's fed
into the router's prompt as a boolean *signal*, and the LLM makes the final call reading the
whole message.

**Why:** a pure keyword hard-gate broke on a real adversarial example — *"these earbuds are so
good they feel like fire"* — a keyword hit with zero actual hazard. Research into production
guardrail patterns confirmed this is a known anti-pattern (the same false-positive class as
"you're killing it" in content moderation): regex should inform the classifier, never replace
its judgment.

**Decision (Weekend 4):** real `interrupt()`-based human-in-the-loop escalation handling was
explicitly **descoped**, not built. The reasoning: this project is the AI-agent part only, not a
full support-ops platform with a human side to hand off to — building real pause/resume +
simulated queue infrastructure for a human team that doesn't exist here would be building for a
hypothetical. What exists (correct detection + a clear stub handoff message) was judged
sufficient for the project's actual scope.

---

## 7. Product comparison via structured metadata filtering

**Decision:** comparing two named products pulls their exact specs via a Chroma metadata filter
(`sku: {"$in": [...]}`), never via similarity-ranked retrieval.

**Why:** similarity search ranks by *meaning closeness*, not *exact identity* — it's the wrong
tool for "give me these two specific products' real numbers." Building this required a real
fix first: Chroma's product metadata originally only had `type`/`category`/`name`/`sku`; the
actual specs (battery life, ANC, IP rating) existed only as prose text, not as filterable
fields. Fixed by flattening each product's `specs` dict into scalar Chroma metadata at
chunk-build time. Product name resolution (matching "Rockerz 450" in a user's message to the
real SKU) is done by **code**, substring-matching against the real catalog — never trusted
directly from the LLM's extraction, same principle as order-number validation.

**Alternative considered:** route comparison questions through the same retrieve→rerank→
generate pipeline as everything else. Rejected — that pipeline is built for *finding* relevant
content, not *fetching exact known facts about named entities*, a different problem shape
entirely.

---

## 8. Policy grounding — quote verification + citation

**Decision:** policy answers must quote specific numbers/conditions verbatim; a code-level check
verifies every quoted span actually appears in the retrieved source text, falling back to a
safe response if not. A citation (`Source: <doc title>`) is appended by code from chunk
metadata, never self-reported by the LLM.

**Why:** found via direct testing — a real answer once stated *"...the refund process will
follow the QC pass timeline"*, a true fact, but one that was never in the context retrieved
that turn. A prompt instruction alone ("answer only from context") doesn't *guarantee*
compliance; only a post-generation check does. The citation is code-appended because the LLM
could misname or invent a source doc if asked to self-report it.

**Alternative considered (and explicitly rejected for a different answer type):** the same
exact-quote mechanism was deliberately **not** reused for product/FAQ answers — forcing
verbatim quoting onto natural conversational prose would make it read robotically. A lighter
numeric fact-check (every number the answer states must appear in the retrieved context) was
built instead for that case — same principle, different strictness calibrated to the answer
type.

---

## 9. Input/output guardrails — what was actually built, and why it isn't a generic fix

**Decision:** broadened the router's `FALLBACK` category description with concrete off-topic/
injection examples, plus a non-binding keyword signal — not a hard block.

**Why:** tested empirically before building anything (`tests/test_prompt_injection_baseline.py`)
— only 1 of 6 adversarial/off-topic messages was caught early; the rest reached the full RAG
pipeline. Importantly, none of the 6 actually *succeeded* as attacks — the final answers held
safe via `generate_answer`'s pre-existing anti-hallucination framing, an *incidental* side
effect, not a real guardrail. That distinction matters: "the attack didn't work" and "we built
a defense" are different claims, and conflating them would overstate what was actually done.

**Output side:** a numeric fact-check for product/FAQ answers (see #8) — deliberately lighter
than policy's exact-quote mechanism, calibrated to the answer type.

---

## 10. Checkpointer choice — `SqliteSaver`, and deliberately not solving returning-user resolution

**Decision:** `MemorySaver` → `SqliteSaver` (durability across process restarts), but the harder
half of the original gap — resolving a *returning* user to their *same* old conversation across
separate app launches — was explicitly **not built**.

**Why the swap:** nearly free (same interface, zero added complexity) and matches the
already-decided local-only deploy target (single-machine durability, not `PostgresSaver`'s
multi-replica sharing, which this project isn't doing).

**Why the descope:** user_email→thread_id resolution is general backend session-management
plumbing, not an agentic-AI skill, and genuinely depends on a session/cookie layer that didn't
exist yet. The actual plan, decided but not built: push `thread_id` to the **client** (browser
`localStorage` generates and remembers it) — sidesteps the hard server-side identity-resolution
problem entirely for a project at this scope.

---

## 11. Eval harness — homegrown/local, deterministic scoring

**Decision:** built `evals/eval_set.json` + `evals/run_evals.py` — a local, zero-dependency
harness — instead of LangSmith. Scoring is plain substring matching (`must_mention`/
`must_not_contain`), not LLM-as-judge.

**Why LangSmith was rejected:** it's a hosted cloud platform. Production traces (and even some
eval cases) would include real data shapes (emails, order numbers) — sending that off-machine
is a genuine data-exposure decision, not just a philosophy one, worth deciding on purpose.

**Why deterministic scoring over LLM-as-judge:** every rubric item in this eval set is a
literal, checkable fact (a number, a route name, a banned phrase) — trusting code over a
model's self-report whenever something can be checked exactly is this project's own
established principle, applied consistently (auth, order-number validation, groundedness
checks all do the same). An LLM judge would add a new source of unreliability to verify
something a plain string check verifies perfectly.

**What it actually found, unprompted:** on its first real run, the harness caught a genuine
router bug (a general return-policy question misrouted to `ORDER`) that no prior test could
have found, because every earlier policy test called `classify_query` directly, bypassing the
graph-level router entirely. This is the concrete answer to "why build an eval set at all" —
it found something nothing else was testing for.

---

## 12. Deploy target — local-only, and why the "obvious" cloud options don't fit

**Decision:** local-only + README + a `docker-compose.yml` for one-command reproducibility. No
public hosting.

**Why a VPS was rejected:** this project has repeatedly measured real LLM calls taking
40-250+ seconds (one `rerank` call hit 251.71s, one `generate_policy_answer` call hit 719.18s).
Cheaper infrastructure doesn't fix that, it worsens it. A VPS also means real ongoing cost, real
maintenance burden, and a genuine break from local-only — the LLM inference itself moving to
rented compute, not just a data-exposure nuance like the LangSmith/eval decisions.

**Why Vercel/Netlify are structurally impossible, not just a worse fit:** both are serverless —
functions spin up per-request with no persistent background process, which conflicts
fundamentally with Ollama needing to stay resident in memory across calls. Their execution-time
caps (roughly 10-300s depending on plan) would kill requests given this project's documented
latency profile.

---

# Rehearsal Questions

Try answering these cold, in your own words, before checking back against the sections above.
Update this file afterward if your phrasing is clearer or you spot something missing.

1. Why did you use LangGraph instead of just chaining LLM calls together?
2. Walk me through your routing design. Why two classifiers instead of one?
3. What's the risk of letting an LLM return a plain string for a routing decision, and how did you actually observe that risk, not just anticipate it?
4. Why did you write your own chunking logic instead of using LangChain's text splitter?
5. Why isn't your auth check implemented as a tool the LLM can call?
6. Walk me through your escalation detection. Why not just hard-match on safety keywords?
7. You built escalation detection but not a real human-handoff system — why, and how did you decide where to draw that line?
8. How does your product comparison feature avoid hallucinating specs?
9. How do you prevent your policy-answering RAG system from inventing details?
10. Why does your product/FAQ answer path use a different groundedness check than your policy path?
11. What did your own testing find when you tried to break your system with prompt injection? What actually happened, precisely?
12. Why SQLite for your checkpointer instead of just keeping everything in memory, and why didn't you build full user-session resolution?
13. Why did you build your own eval harness instead of using LangSmith?
14. Why is your eval scoring deterministic string matching instead of using an LLM to grade answers?
15. What's the most interesting bug your eval harness found, and why didn't any earlier test catch it?
16. Why didn't you deploy this to a VPS or Vercel? What specifically breaks?
17. What's the single biggest unresolved issue in this project right now, and why haven't you fixed it?
18. If you had another two weeks, what would you build next and why that, specifically, over everything else still open?
