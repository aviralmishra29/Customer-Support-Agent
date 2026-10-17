"""FastAPI layer wrapping the compiled graph in a real HTTP API.

Session identity is entirely client-managed: the caller generates and remembers its own
thread_id (e.g. browser localStorage), sending it with every request. No server-side identity
resolution is attempted here - that was deliberately descoped (see docs/TODO.md's checkpointer-
durability entry) as general session-management plumbing, not something worth building for a
learning-scoped local project.

Concurrency is limited to 1 in-flight graph execution at a time. This is NOT a generic
abuse-prevention rate limit (a time-window limiter like slowapi would solve a different
problem) - it's a direct match to the real hardware constraint: one CPU-bound Ollama instance
that doesn't meaningfully parallelize two heavy requests, it just makes both slower. A second
request while one is running gets a clear 429 instead of silently queuing or degrading.

No streaming in this pass (deliberate choice) - plain blocking request/response first.
"""

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from observability import setup_tracing

setup_tracing()

from graph.build import build_graph

_graph = None
_busy_semaphore = threading.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    print("startup: building graph (opens the SQLite checkpointer connection once)")
    _graph = build_graph()
    print("startup: graph ready")
    yield
    print("shutdown")


app = FastAPI(title="boAt Support Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, description="Client-managed conversation identity.")
    message: str = Field(min_length=1, description="The customer's message.")


class ChatResponse(BaseModel):
    route: str
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # A plain `def` route - FastAPI/Starlette runs these in a worker thread pool automatically,
    # which is exactly right for our blocking graph.invoke() call, with no manual
    # asyncio.to_thread plumbing needed. The semaphore below serializes across those worker
    # threads - without it, two concurrent requests could each grab their own thread and both
    # call graph.invoke() at once, defeating the whole point of limiting concurrency.
    acquired = _busy_semaphore.acquire(blocking=False)
    if not acquired:
        print(f"chat: rejected (already busy) thread_id={request.thread_id!r}")
        raise HTTPException(
            status_code=429,
            detail="The agent is busy processing another request - please try again shortly.",
        )

    try:
        print(f"chat: thread_id={request.thread_id!r} message={request.message!r}")
        config = {"configurable": {"thread_id": request.thread_id}}
        result = _graph.invoke({"messages": [("human", request.message)]}, config=config)
        print(f"chat: route={result['route']!r}")
        return ChatResponse(route=result["route"], answer=result["answer"])
    finally:
        _busy_semaphore.release()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def frontend():
    # Served directly by this same FastAPI app - no separate frontend server/build step,
    # same-origin so no CORS config needed, no Node/React toolchain required.
    return FileResponse(Path(__file__).parent / "static" / "index.html")
