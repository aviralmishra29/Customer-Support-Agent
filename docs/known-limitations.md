# Known Limitations & Next Steps

An honest list of what's genuinely unresolved in this project, pulled directly from
`docs/TODO.md`'s deferred/open section and scattered decisions made throughout — not a sanitized
summary. Organized by what kind of gap each one actually is, since "we haven't built X" and "we
found a real bug and haven't fixed it" are different categories worth being able to distinguish
out loud.

## Real, unresolved bugs

- **Latency root cause not conclusively identified.** Structured-output overhead and pure RAM
  pressure were both ruled out empirically back in Weekend 2 (freeing RAM made timing *worse*,
  not better); CPU contention on this CPU-only setup is suspected but never proven.
- **Two latency outliers found while writing `docs/cost-and-latency.md`, not while they
  happened:** `generate_policy_answer` hit 719.18s once, `rerank` hit 251.71s once — both from
  legitimate, non-adversarial runs. The 240-second Ollama timeout built during Weekend 4
  apparently didn't fire for the 719s case, which is itself an open question worth investigating
  (does the timeout actually apply to this call path, or did the call genuinely complete in
  719s without the timeout mechanism engaging at all?) — not yet answered, flagged honestly as
  unknown rather than assumed.
- **The sarcasm-misclassification bug (documented since 2026-07-11) has passed its last 2 eval
  runs, but that is not being treated as "fixed."** Non-deterministic LLM output means 2 passes
  is weak evidence, not proof. Needs several more runs (or an isolated repeated test) before its
  status changes.

## Deliberately deferred — a real fix exists, scoped out by choice

- **Blended-intent messages** — a single message carrying two genuine intents (order status +
  product question) only gets the first one handled; the schema forces exactly one `Route`.
  Real fix (`routes: list[Route]` + a graph fan-out/merge mechanism) was researched and
  identified back in Weekend 2, deliberately deferred as its own future concept-first stage
  rather than solved half-heartedly.
- **Hinglish generation gap** — the pipeline demonstrably understands Hinglish input (confirmed:
  a Hinglish query's top retrieval results include both the Hinglish and English versions of the
  same FAQ), but `generate_answer` always replies in English regardless of input language. A fix
  (add a language-matching instruction to the generation prompt) was identified, never applied.
- **`rerank`'s "none relevant" escape hatch** — it's structurally forced to always pick exactly
  3 candidates, even for genuinely irrelevant input, which is part of what caused the 192.94s/
  251.71s outliers above. Deferred as defense-in-depth, since the router hardening and `"unsure"`
  early-decline fixes (Weekend 4) now keep most junk input from ever reaching `rerank` at all —
  "most" is not "all," and this is the obvious next layer if a case is found that still slips
  through.

## Explicitly out of scope — a deliberate project-boundary decision, not an oversight

- **No real escalation human-in-the-loop handoff.** The router correctly *detects* escalation-
  worthy messages; `escalation_node` returns a stub handoff message rather than a real
  `interrupt()`-based pause/resume + simulated queue. Decision: this project is the AI-agent
  part only, not a full support-ops platform with a human team to hand off to — building real
  infrastructure for a human side that doesn't exist here would be building for a hypothetical.
- **Returning-user thread resolution is unsolved, on purpose.** Nothing maps a verified identity
  back to their previous `thread_id` across separate app launches. Judged to be general backend
  session-management plumbing, not an agentic-AI skill worth building for a project at this
  scope. The actual plan if it's ever needed: push `thread_id` to the client (browser
  `localStorage`), sidestepping server-side identity resolution entirely — simpler than the
  server-side mapping mechanism originally considered.
- **No token-level response streaming.** The FastAPI layer deliberately ships plain blocking
  request/response first; stage-progress SSE or true token streaming were both discussed as
  real options and explicitly deferred to a later pass, given this project's architecture (nodes
  run several internal LLM calls before a user-facing one) makes true streaming more invasive
  than it looks.
- **Single-instance, not horizontally scalable.** The deploy-target decision (local-only,
  `SqliteSaver`, a single CPU-bound Ollama instance behind a concurrency semaphore) is correct
  for this project's actual scope but doesn't scale past one machine serving one request at a
  time. `PostgresSaver` and a queue/worker model would be the real next step for anything beyond
  a demo, not attempted here on purpose.

## What would actually be next, if continuing this project

In rough priority order, reasoning included (useful for "what would you do with 2 more weeks"):

1. **Investigate the two new latency outliers** — highest priority because it's an open question
   about whether a safety mechanism (the timeout) is actually working as designed, not just a
   performance nice-to-have.
2. **Re-run the eval suite several more times** to get real confidence on the sarcasm bug's
   status, rather than leaving it in an ambiguous "passed twice" state indefinitely.
3. **The `rerank` escape hatch** — directly addresses the worst tail-latency cases, and is a
   contained, well-understood fix at this point (the design was already worked out, just not
   built).
4. **Hinglish reply-language fix** — small, well-scoped, already has an identified solution.
5. **Blended-intent handling** — the largest remaining piece of genuinely new mechanism work,
   intentionally saved for last since it's a real architectural addition, not a patch.
