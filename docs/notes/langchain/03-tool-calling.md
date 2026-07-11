# Tool calling with `@tool` and `bind_tools()`

## The mechanism
`@tool` (from `langchain_core.tools`) turns a plain Python function into something an LLM can be offered as an option — it auto-derives a schema from the function's type hints and uses the **docstring as the tool's description**, which is what the model actually reads to decide relevance (same lesson as the classify prompt needing real category descriptions, not just bare names — a vague docstring would be just as unreliable as a bare-names classification prompt).

`llm.bind_tools([get_order_status])` attaches the tool's schema to the model. Calling `.invoke(question)` on the bound model returns an `AIMessage` with a `.tool_calls` list — populated if the model decided to call something, empty if it decided to answer directly.

**Important mechanic, easy to miss:** `bind_tools()` + `.invoke()` only makes the *decision* — it does not execute the tool function. `get_order_status`'s own print statements never fired during the test, confirming the function itself was never actually called; only `response.tool_calls` (name + args the model wants to use) was produced. Actually running the tool and feeding its result back to the model for a final answer is a separate step your own code (or a higher-level agent construct, like what LangGraph will provide next weekend) is responsible for.

## Test setup
Built `get_order_status(order_number: str) -> str` (`tools/order_tools.py`) against real mock data in `data/seed/orders.json`. Bound it to `ChatOllama(model="qwen3:1.7b", reasoning=True)`. Ran 3 test cases per the plan's explicit checkpoint ask: "write down why the LLM chose tool vs direct answer."

## Case 1: "What's the status of my order BOAT-2026-001001?"
**Tool called:** `get_order_status(order_number='BOAT-2026-001001')`.
**Why:** the question contains an exact, unambiguous order number, and the tool's description says exactly when to use it ("Only call this when the customer has provided a specific order number"). Straightforward match between intent and tool applicability.

## Case 2: "Which earbuds have good battery life?"
**Tool not called.** Answered directly, explaining it can't help with product specs, and proactively offered order-status help instead.
**Why:** the model's own reasoning explicitly named the mismatch — the only available tool is order-lookup, and this question has nothing to do with orders. Correctly recognized "no available tool applies" rather than forcing a bad-fit tool call or hallucinating an answer.

## Case 3: "I placed an order yesterday, when will it arrive?" — the genuinely open test
**Tool not called** — and this was the interesting, previously-unknown outcome. The order-related *intent* is clearly present (unlike case 2), but no order number was given, and `get_order_status` requires one. Two plausible failure modes existed going in: (a) hallucinate a plausible-looking order number just to have something to call the tool with, or (b) recognize the missing required argument and ask for it. Got (b): *"they didn't provide the exact order number. The tool requires the order number to look up the status... I should ask them to provide the specific order number."* Responded with a clear clarifying question, including an example format. This is the safer outcome and wasn't guaranteed — worth having tested rather than assumed, since a hallucinated order number silently fed into a real lookup would be a much worse failure mode than just asking.

## Takeaway
All 3 cases showed correct, well-calibrated behavior: call when intent + required info are both present, decline gracefully when no tool applies, and ask for missing required info rather than guessing when intent is present but the argument isn't. This is exactly the tool-vs-direct-answer judgment the plan's checkpoint wanted verified, not assumed.
