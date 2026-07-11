"""Hand-rolled content-hash embedding cache, wrapping any LangChain Embeddings implementation."""

import hashlib
import json
from pathlib import Path

from langchain_core.embeddings import Embeddings

CACHE_DIR = Path(__file__).parent.parent / "data" / "embedding_cache"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedEmbeddings(Embeddings):
    def __init__(self, underlying: Embeddings, namespace: str):
        self.underlying = underlying
        self.cache_path = CACHE_DIR / f"{namespace}.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        print(f"CachedEmbeddings loaded {len(self.cache)} cached vectors from {self.cache_path}")

    def _save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        hashes = [_hash_text(t) for t in texts]
        to_embed_texts = [texts[i] for i, h in enumerate(hashes) if h not in self.cache]
        to_embed_indices = [i for i, h in enumerate(hashes) if h not in self.cache]

        print(f"embed_documents: {len(texts) - len(to_embed_texts)} cache hits, {len(to_embed_texts)} cache misses")

        if to_embed_texts:
            new_vectors = self.underlying.embed_documents(to_embed_texts)
            for idx, vector in zip(to_embed_indices, new_vectors):
                self.cache[hashes[idx]] = vector
            self._save_cache()
            print(f"embed_documents: computed and cached {len(new_vectors)} new vectors")

        return [self.cache[h] for h in hashes]

    def embed_query(self, text: str) -> list[float]:
        h = _hash_text(text)
        if h in self.cache:
            print("embed_query: cache hit")
            return self.cache[h]

        vector = self.underlying.embed_query(text)
        self.cache[h] = vector
        self._save_cache()
        print("embed_query: cache miss, computed and cached")
        return vector
