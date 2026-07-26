# Graph-enforced auth: why a tool is the wrong shape, and where thread_id fits (or doesn't)

## Traditional session-based auth, recapped for reference

Login → server generates a random, unguessable session ID → session record (`session_id → {user_id, expires_at}`) stored server-side (session store: Redis, DB, or in-memory for dev) → session ID sent to the client, typically as an `HttpOnly` cookie so client-side JS can't read/leak it → every subsequent request carries that ID automatically (cookie) → server middleware looks it up *before* any business logic runs, attaches the resolved user, or rejects the request if missing/expired. Two real gotchas: **session fixation** (always issue a fresh ID at login, never reuse a pre-login one) and **CSRF** (cookies auto-attach, so a malicious site can ride a valid session without stealing it).

The load-bearing idea: **the check happens in middleware, in front of the business logic, as code — never as something the business logic itself decides to do.**

## Why a "check_is_authenticated" tool would be the wrong shape here

First instinct was: give the LLM a tool it can call to check auth status. That's actually the same class of mistake as putting the auth rule in a prompt instruction — a tool is still something the **LLM decides whether and when to invoke**. A cleverly-worded message could convince the model the user is "obviously" already logged in and skip calling the tool, or the model could call it, get `False` back, and still generate an answer as if it were `True` — models don't reliably act on tool results the way code reliably acts on an `if` statement.

**The fix has to remove the LLM's agency over the check entirely.** That means the gate lives in the **conditional edge** (`graph/edges.py`), not inside a node the LLM's output flows through, and definitely not as a tool:

```python
AUTH_REQUIRED_ROUTES = {"ORDER"}

def route_after_router(state: State) -> str:
    route = state["route"]
    if route in AUTH_REQUIRED_ROUTES and not is_authenticated(state):
        return "auth_node"
    ...
```

This runs as part of the graph's wiring, automatically, on every single request — nothing about the LLM's output can skip it. The LLM can decide `route == "ORDER"` all it wants; the edge still refuses to let `order_node` execute if `is_authenticated(state)` is `False`. The only way to bypass this check is to change the graph's actual code — which is exactly the property we want.

## State design: a bare bool wasn't enough

Original design had `is_authenticated: bool`. Replaced with:

- `user_id: Optional[str]` — *who*, specifically, is authenticated (not authenticated in the abstract)
- `auth_expires_at: Optional[float]` — a real expiry timestamp, mirroring a session's TTL
- `auth_stage: Optional[str]` — where the conversational auth flow currently is (`None` / `"awaiting_otp"`)
- `pending_email`, `pending_otp` — the in-flight values for the *current* verification attempt

`is_authenticated(state)` (in `graph/nodes.py`, imported by `graph/edges.py`) is the single gate function: `user_id` present AND `time.time() < auth_expires_at`. Anything gating sensitive nodes should call this, never read a bare flag directly — keeps the "what counts as authenticated" definition in exactly one place.

## auth_node: a 2-turn conversational state machine, LLM only for parsing

1. **No `auth_stage` yet** → regex-extract an email from the message (falls back to the raw message if no email-shaped substring found), generate a mock OTP (`random.randint`, printed to console — no real email service exists), store it as `pending_otp`/`pending_email`, set `auth_stage="awaiting_otp"`.
2. **`auth_stage == "awaiting_otp"`** → regex-extract a numeric code from the reply, **plain string comparison** against `pending_otp`. Match → set `user_id`, `auth_expires_at = now + 30min`, clear the pending fields and stage. No match → ask again, stay in the same stage.

The LLM is never in this path at all, actually — both steps are regex + plain Python, deliberately, since even "parsing" doesn't need an LLM call here and every call we skip is also a real latency win given how slow generation is on this hardware.

## Known MVP limitation, not yet solved

After successful OTP verification, the node just confirms login and asks the user to *repeat* their original order question — it doesn't automatically re-run the order lookup that originally triggered the auth requirement. Fixing that (remembering "what did they actually want" across the auth interruption) is a real UX gap, noted but out of scope for this pass.

## Item 3: `thread_id` is not a substitute for a real session/identity boundary

`thread_id` is **conversation identity** — literally just whatever string we pass as `config["configurable"]["thread_id"]` to key the checkpointer's storage. Nothing about it inherently ties to a *verified person*. Two consequences worth being explicit about:

- **For this project's current scope (mock, CLI-driven, single developer testing):** we generate a fresh `thread_id` per conversation (e.g. `uuid4()`) and treat "this thread's `is_authenticated(state)` is true" as good enough, because there's no real multi-user traffic yet.
- **This does NOT generalize to a real deployed system as-is.** If two different real users' requests ever shared a `thread_id` — e.g. a bug in how a frontend generates/reuses thread IDs, a caching layer that hands out the same ID twice, a naive implementation that derives `thread_id` from something guessable — user A's verified login would silently authenticate user B on that same thread. The checkpointer has no concept of "which human is this," only "which conversation is this," and conflating the two is the actual security bug waiting to happen.

**How this should actually work once Weekend 5's FastAPI layer exists:** the real identity boundary belongs at the **web session layer**, one level above the graph — the same cookie-based session model recapped above. The web server verifies the session cookie *first* (proper middleware, proper session store), resolves a real `user_id` from it, and only *then* derives a `thread_id` from that already-verified identity (e.g. `thread_id = f"user:{verified_user_id}:{conversation_id}"`, or simply looks up/creates a conversation scoped to that user). The graph's checkpointer's job is conversation memory, full stop — it should never be the thing a real system trusts to prove who someone is. Treat this as a placeholder architecture for now, and revisit it for real when the API layer gets built.
