"""Baseline (no error handling yet): deliberately break each of the 3 external dependencies
(Ollama, Chroma, SQLite) and observe the REAL exception type/message each one actually raises.
Purpose is purely observational - design the structured error handling around real behavior,
not guessed exception types, same empirical-first approach used for every other guardrail
built in this project so far."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_ollama_down():
    print("\n=== 1. Ollama unreachable (wrong port) ===")
    from langchain_ollama import ChatOllama
    try:
        llm = ChatOllama(model="qwen3:1.7b", base_url="http://localhost:1", reasoning=False)
        llm.invoke("hello")
        print("no exception raised (unexpected)")
    except Exception as e:
        print(f"exception type: {type(e).__module__}.{type(e).__name__}")
        print(f"exception message: {e}")


def test_chroma_unreachable():
    print("\n=== 2. Chroma pointed at a garbage/nonexistent path ===")
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    try:
        vs = Chroma(
            persist_directory="/nonexistent/path/that/does/not/exist",
            collection_name="boat_support_prod_db",
            embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        )
        vs.get(limit=1)
        print("no exception raised (unexpected)")
    except Exception as e:
        print(f"exception type: {type(e).__module__}.{type(e).__name__}")
        print(f"exception message: {e}")


def test_sqlite_missing():
    print("\n=== 3. SQLite pointed at a nonexistent DB path ===")
    try:
        conn = sqlite3.connect("/nonexistent/dir/orders.db")
        conn.execute("SELECT * FROM orders WHERE order_number = ?", ("BOAT-2026-001001",))
        print("no exception raised (unexpected)")
    except Exception as e:
        print(f"exception type: {type(e).__module__}.{type(e).__name__}")
        print(f"exception message: {e}")


def test_sqlite_locked():
    print("\n=== 4. SQLite DB locked by another connection (real contention, not just a bad path) ===")
    db_path = Path(__file__).parent.parent / "data" / "orders" / "orders.db"
    try:
        blocker = sqlite3.connect(str(db_path))
        blocker.execute("BEGIN EXCLUSIVE")
        contender = sqlite3.connect(str(db_path), timeout=1.0)
        contender.execute("SELECT * FROM orders WHERE order_number = ?", ("BOAT-2026-001001",))
        print("no exception raised (unexpected)")
    except Exception as e:
        print(f"exception type: {type(e).__module__}.{type(e).__name__}")
        print(f"exception message: {e}")
    finally:
        blocker.rollback()
        blocker.close()


if __name__ == "__main__":
    test_ollama_down()
    test_chroma_unreachable()
    test_sqlite_missing()
    test_sqlite_locked()
