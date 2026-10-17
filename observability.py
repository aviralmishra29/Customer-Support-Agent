"""Local-only tracing via Arize Phoenix (self-hosted, OpenTelemetry-based) - chosen over
LangSmith specifically because production traces would include real customer data (emails
from auth, order numbers, shipping cities), not just synthetic test questions like the eval
set. Sending that to a third-party cloud service is a real data-exposure decision, not just a
"local-only philosophy" one - Phoenix keeps every trace on this machine.

Usage: call setup_tracing() once, before building/invoking the graph. Requires a Phoenix
server already running locally (see README/run instructions - `phoenix serve`), listening on
the default port 6006.
"""

from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

PHOENIX_ENDPOINT = "http://localhost:6006/v1/traces"
PROJECT_NAME = "boat-support-agent"

_instrumented = False


def setup_tracing():
    global _instrumented
    if _instrumented:
        print("setup_tracing: already instrumented, skipping")
        return

    print(f"setup_tracing: registering OTel tracer pointed at {PHOENIX_ENDPOINT}")
    tracer_provider = register(
        project_name=PROJECT_NAME,
        endpoint=PHOENIX_ENDPOINT,
    )

    print("setup_tracing: instrumenting LangChain/LangGraph calls")
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    _instrumented = True
    print("setup_tracing: done - every graph.invoke() from here on will send traces to Phoenix")
