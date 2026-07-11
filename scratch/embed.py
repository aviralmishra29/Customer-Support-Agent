"""Embed chunks with nomic-embed-text via Ollama, in sequential batches of 20."""

import json
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
BATCH_SIZE = 20


def load_chunks() -> list[dict]:
    path = DATA_DIR / "processed" / "chunks.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "input": texts})
    response.raise_for_status()
    return response.json()["embeddings"]


def embed_all_chunks(chunks: list[dict]) -> list[dict]:
    embedded = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vectors = embed_batch([c["text"] for c in batch])

        if len(vectors) != len(batch):
            raise RuntimeError(f"Batch at index {i}: sent {len(batch)} texts, got back {len(vectors)} vectors")

        for chunk, vector in zip(batch, vectors):
            embedded.append({**chunk, "embedding": vector})

        print(f"Embedded chunks {i} to {i + len(batch) - 1}")

    return embedded


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    embedded = embed_all_chunks(chunks)

    out_path = DATA_DIR / "processed" / "embeddings.jsonl"
    with out_path.open("w") as f:
        for item in embedded:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(embedded)} embedded chunks to {out_path}")
