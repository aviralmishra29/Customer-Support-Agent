"""Verifies the FastAPI layer: a basic /chat call works, and critically, that the concurrency
semaphore actually rejects a second overlapping request with 429 instead of silently queuing
it or letting both run at once. Requires the API already running (uvicorn main:app)."""

import threading
import time

import requests

BASE_URL = "http://localhost:8000"
results = {}


def send_request(label, thread_id, message):
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"thread_id": thread_id, "message": message},
            timeout=300,
        )
        results[label] = (resp.status_code, resp.json(), time.perf_counter() - start)
    except Exception as e:
        results[label] = (None, str(e), time.perf_counter() - start)


if __name__ == "__main__":
    print("=== Health check ===")
    health = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"status={health.status_code} body={health.json()}")

    print("\n=== Firing two overlapping /chat requests ===")
    t1 = threading.Thread(
        target=send_request,
        args=("first (slow, real pipeline)", "test-concurrency-1", "Which earbuds have good battery life?"),
    )
    t2 = threading.Thread(
        target=send_request,
        args=("second (should be rejected)", "test-concurrency-2", "What's the capital of France?"),
    )

    t1.start()
    time.sleep(1.0)  # give the first request time to acquire the semaphore before firing the second
    t2.start()

    t1.join()
    t2.join()

    for label, (status, body, elapsed) in results.items():
        print(f"\n{label}:")
        print(f"  status={status}  elapsed={elapsed:.2f}s")
        print(f"  body={body}")
