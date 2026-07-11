"""Load embedded chunks into a persistent local ChromaDB collection."""

import json
from pathlib import Path

import chromadb

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "boat_support"


def load_embedded_chunks() -> list[dict]:
    path = DATA_DIR / "processed" / "embeddings.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    chunks = load_embedded_chunks()
    print(f"Loaded {len(chunks)} embedded chunks")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} items")
    print(f"Persisted at {CHROMA_DIR}")
