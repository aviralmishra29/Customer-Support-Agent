"""LangChain rebuild of ingestion: reuse our own chunking output, wrap as Documents, embed + store via LangChain."""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from cached_embeddings import CachedEmbeddings

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "boat_support_prod_db"
EMBED_MODEL = "nomic-embed-text"


def load_documents() -> list[Document]:
    path = DATA_DIR / "processed" / "chunks.jsonl"
    chunks = [json.loads(line) for line in path.read_text().splitlines()]

    documents = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"], id=chunk["id"])
        for chunk in chunks
    ]
    print(f"load_documents wrapped {len(documents)} chunks into Document objects")
    return documents


if __name__ == "__main__":
    documents = load_documents()
    sample = documents[0]
    print(f"Sample document id: {sample.id}")
    print(f"Sample document metadata: {sample.metadata}")
    print(f"Sample document page_content: {sample.page_content}")

    base_embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    embeddings = CachedEmbeddings(base_embeddings, namespace=EMBED_MODEL)
    print(f"Using embedding model: {EMBED_MODEL} (cached)")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(f"Chroma.from_documents completed for collection '{COLLECTION_NAME}'")
    print(f"Collection count: {vectorstore._collection.count()}")
