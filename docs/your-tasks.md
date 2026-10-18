# Your Tasks — Weekend 6 Interview Prep

Things that genuinely need *your* input/practice, not mine — I can compile facts and draft
material, but these specifically require you to articulate, decide, or rehearse something
yourself. Update the referenced docs after each one if your own answer is better than what's
there, or if you spot a gap.

---

## 1. Architecture diagram — explain it back

File: `docs/architecture.md`

Answer these in your own words (no notes):
- Why are `orders.db` and `checkpoints.db` two separate SQLite databases instead of one shared file? What would actually break if they were merged?
- Two different conditional edges (`route_from_start`, `route_after_router`) can both send a user to `auth_node`. If asked "why not just one auth-check at the start of every turn," what's the answer?
- Why isn't `compare_products` just another branch inside the same retrieve→rerank→generate pipeline as everything else?

## 2. Design-decision writeup — rehearse cold

File: `docs/design-decisions.md` (18 questions at the bottom)

Answer all 18 without looking at the write-ups above them first. Then check your answers against
the doc, and update it wherever your own phrasing is clearer or you notice something missing or
wrong.

## 3. Eval results writeup — ready to rehearse

File: `docs/eval-results.md`

Answer these in your own words, no notes:
- Why did the eval set catch the `ORDER`/policy router bug when every earlier test missed it?
- What does "20/21 passed, then 21/21 passed" actually prove, and what does it *not* prove?
- The `route-sarcastic-complaint` case passed on both runs — is that bug fixed? Why or why not is that a trustworthy conclusion?
- A case can pass the eval (correct answer) and still be a real problem (the 719.18s policy-answer case). What does that tell you about the difference between correctness testing and performance testing?

## 4. Known limitations & next-steps — ready to review

File: `docs/known-limitations.md`

Read it critically: does it honestly reflect what you understand to be the real open gaps, or
does anything read like it's glossing over something? One specific thing worth your own
judgment call: do you agree with the priority order in the "what's actually next" section
(timeout investigation first, blended-intent last), or would you reorder it?

## 5. Interview-question rehearsal — a dedicated live session

This is the one that can't be done by writing docs — a mock-interview session where I ask real
questions (not just the 18 listed above) and give you direct feedback on your answers live.
Schedule this as its own separate pass, after you've gone through #1–#4 at least once on your
own, not before.

---

**Suggested order:** #2 first (it's the meatiest and most likely to surface gaps in your own
understanding), then #1, then #3–#4 once they exist, then #5 last as the real-feeling dry run.
